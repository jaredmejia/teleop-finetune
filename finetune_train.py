import argparse
from functools import partial

import librosa
import librosa.display
import numpy as np
import torch
import tqdm
import yaml

import wandb

# relative imports
from finetune_models import AvidR3M, MultiSensoryAttention
from finetune_utils import (
    backbone_transforms,
    multi_collate,
    prepare_data,
    set_seeds,
    vis_batch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model, optimizer, criterion, dataloader, debug=False):
    """Train model on training set

    Args:
        model (nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for training set
        debug (bool, optional): whether to run in debug mode. Defaults to False.

    Returns:
        float: average loss over training set
    """
    model.set_train()
    running_loss = 0
    for _, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds.squeeze(), data["target"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data["target"].size(0)

        if debug:
            break

    return running_loss / len(dataloader.sampler)


def eval_loop(model, criterion, dataloader, debug=False):
    """Evaluate model on validation set

    Args:
        model (nn.Module): model to evaluate
        criterion (nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for validation set
        debug (bool, optional): whether to run in debug mode. Defaults to False.

    Returns:
        float: average loss on validation set
    """
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for _, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            preds = model(data)
            loss = criterion(preds.squeeze(), data["target"])
            running_loss += loss.item() * data["target"].size(0)

            if debug:
                break

    return running_loss / len(dataloader.sampler)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_file", type=str, required=True, help="path to model config file"
    )
    parser.add_argument(
        "--extract_dir", type=str, default="./prep_data/3s_window_train"
    )
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=47)
    parser.add_argument("--log_freq", type=int, default=4)
    parser.add_argument("--num_samples_log", type=int, default=15)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "vis"])
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_arguments()

    # print args with names
    print("args:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    # load model configs
    model_config = yaml.safe_load(open(args.model_config_file, "rb"))
    model_args = model_config["model"]["args"]
    model_arch = model_config["model"]["arch"]
    model_name = model_config["model_name"]
    dataset_cfg = model_config["dataset"]
    training_cfg = model_config["training"]

    # print model configs
    print("model configs:")
    for arg in model_args:
        print(f"\t{arg}: {model_args[arg]}")

    # print training configs
    print("training configs:")
    for arg in training_cfg:
        print(f"\t{arg}: {training_cfg[arg]}")

    # get model specific transforms
    transforms = backbone_transforms(model_name, model_config)

    # data / misc args
    extract_dir = args.extract_dir
    validation_split = args.validation_split
    random_seed = args.random_seed
    mode = args.mode
    log_freq = args.log_freq
    num_samples_log = args.num_samples_log
    debug = bool(args.debug)

    # training args
    BATCH_SIZE = training_cfg["batch_size"]
    MLP_LR = training_cfg["mlp_learning_rate"]
    BACKBONE_LR = training_cfg["backbone_learning_rate"]
    FROZEN_START = training_cfg["frozen_start"]
    UNFROZEN_EPOCHS = training_cfg["unfrozen_epochs"]
    FROZEN_END = training_cfg["frozen_end"]
    DISCRIMINATIVE_LR = training_cfg["discriminative_lr"]

    # initialize wandb
    if mode == "train":
        wandb.init(project="manimo-teleop", entity="contact-mic", name=model_name)
    elif mode == "vis":
        wandb.init(
            project="manimo-teleop", entity="contact-mic", name=f"{model_name}_vis"
        )
    wandb.config.update(args)
    wandb.config.update(model_config)
    wandb.config.update(training_cfg)
    wandb.define_metric("val/loss", summary="min")

    # set random seed
    set_seeds(random_seed)

    # MODEL
    if model_arch == "AvidR3M":
        model = AvidR3M(**model_args).to(DEVICE)
        frozen_backbone = model_args["frozen_backbone"]
    elif model_arch == "MultiSensoryAttention":
        model = MultiSensoryAttention(**model_args).to(DEVICE)
        frozen_backbone = False

        assert (
            model_args["seq_len"] == dataset_cfg["seq_len"]
        ), "seq_len in model config must match seq_len in dataset config"

    else:
        raise NotImplementedError(f"arch {model_arch} not implemented")

    include_audio = model_args["modality"] == "audio-video"
    include_orig_audio = mode == "vis" and model_args["modality"] == "audio-video"
    include_image = model_args["goal_conditioned"]

    wandb.watch(model, log="all", log_freq=100)

    # DATA
    print("\n##### DATA #####")
    collate_func = partial(
        multi_collate,
        device=DEVICE,
        include_image=include_image,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    train_loader, val_loader = prepare_data(
        dataset_cfg,
        transforms,
        BATCH_SIZE,
        collate_func,
        include_image=include_image,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
        num_img_frames=dataset_cfg["seq_len"],
    )

    # reset seed due to model differences in random initialization
    set_seeds(random_seed)

    if mode == "train":
        # TRAIN

        # define loss
        if model_config["dataset"]["target_type"] == "discrete":
            from finetune_utils import multi_discrete_loss

            criterion = multi_discrete_loss(
                num_dim=dataset_cfg["num_dim"],
                output_per_dim=dataset_cfg["output_per_dim"],
            )

        else:
            criterion = torch.nn.MSELoss()

        # if frozen, only train the last layer, otherwise use discriminative learning rates
        if frozen_backbone:
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        else:
            if DISCRIMINATIVE_LR:
                print("Using discriminative learning rates")
                print("Backbone LR:", BACKBONE_LR)
                print("MLP LR:", MLP_LR)
                trainable_params = [
                    {"params": model.batchnorm.parameters(), "lr": MLP_LR},
                    {"params": model.feat_fusion.parameters(), "lr": MLP_LR},
                ]
                if model_arch == "AvidR3M":
                    trainable_params.append(
                        {"params": model.avid_backbone.parameters(), "lr": BACKBONE_LR}
                    )
                if include_image:
                    trainable_params.append(
                        {"params": model.r3m_backbone.parameters(), "lr": BACKBONE_LR}
                    )

            else:
                trainable_params = model.parameters()

        if training_cfg["optimizer"] == "adam":
            optimizer = torch.optim.Adam(trainable_params, lr=MLP_LR)
        elif training_cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(trainable_params, lr=MLP_LR)
        elif training_cfg["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=MLP_LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, threshold=1e-2, verbose=True
        )

        best_val_loss = np.inf
        for epoch in range(FROZEN_START + UNFROZEN_EPOCHS + FROZEN_END):
            if epoch == FROZEN_START and frozen_backbone:
                model.unfreeze_backbone()

                # discriminative learning rates
                trainable_params = [
                    {"params": model.avid_backbone.parameters(), "lr": BACKBONE_LR},
                    {"params": model.r3m_backbone.parameters(), "lr": BACKBONE_LR},
                    {"params": model.batchnorm.parameters(), "lr": MLP_LR},
                    {"params": model.feat_fusion.parameters(), "lr": MLP_LR},
                ]

                if training_cfg["optimizer"] == "adam":
                    optimizer = torch.optim.Adam(trainable_params, lr=MLP_LR)
                elif training_cfg["optimizer"] == "adamw":
                    optimizer = torch.optim.AdamW(trainable_params, lr=MLP_LR)
                elif training_cfg["optimizer"] == "sgd":
                    optimizer = torch.optim.SGD(trainable_params, lr=MLP_LR)

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=3, threshold=1e-2, verbose=True
                )
                print("##### UNFREEZING BACKBONE #####")

            elif epoch == FROZEN_START + UNFROZEN_EPOCHS:
                model.freeze_backbone()

            train_loss = train_loop(
                model, optimizer, criterion, train_loader, debug=debug
            )
            print(f"Epoch: {epoch}, Training Loss: {train_loss}")

            val_loss = eval_loop(model, criterion, val_loader, debug=debug)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.state_dict(), f"./trained_models/best_model_{model_name}.pth"
                )
            scheduler.step(val_loss)

            # log visualizations / preds
            if epoch % log_freq == 0:
                vis_batch(
                    val_loader,
                    model=model,
                    model_arch=model_arch,
                    max_samples=num_samples_log,
                    include_image=include_image,
                    include_audio=include_audio,
                    include_orig_audio=include_orig_audio,
                    output_dim=model_args["output_dim"],
                )

    elif mode == "vis":
        # VISUALIZE

        if debug:
            model = None
            d_loader = train_loader
        else:
            model.load_state_dict(
                torch.load(model_config["model"]["checkpoint"], map_location=DEVICE)
            )
            model.eval()
            d_loader = val_loader

        vis_batch(
            d_loader,
            model=model,
            model_arch=model_arch,
            max_samples=num_samples_log,
            include_image=include_image,
            include_audio=include_audio,
            include_orig_audio=include_orig_audio,
            output_dim=model_args["output_dim"],
        )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
