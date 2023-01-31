import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import yaml

from functools import partial
from torchvision import transforms as T
from PIL import Image

# relative imports
from finetune_models import AvidR3M
from finetune_utils import (
    invNormalize,
    set_seeds,
    backbone_transforms,
    multi_collate,
)
from teleop_prop_data import TeleopCompletionDataset, load_target_data, get_dataloaders


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(model, optimizer, criterion, dataloader):
    """Train model on training set

    Args:
        model (nn.Module): model to train
        optimizer (torch.optim.Optimizer): optimizer
        criterion (nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for training set

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

    return running_loss / len(dataloader.sampler)


def eval_loop(model, criterion, dataloader):
    """Evaluate model on validation set

    Args:
        model (nn.Module): model to evaluate
        criterion (nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for validation set

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

    return running_loss / len(dataloader.sampler)


def vis_preds(
    model, dataloader, max_samples=15, include_audio=False, include_orig_audio=False
):
    """Visualize a few samples from the dataloader

    Args:
        model (torch.nn.Module): model to visualize
        dataloader (torch.utils.data.DataLoader): dataloader to visualize
        max_samples (int, optional): max number of samples to visualize. Defaults to 10.

    Returns:
        None
    """
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        batch_size = sample_batch["target"].size(0)
        preds = model(sample_batch).detach().cpu().numpy().squeeze()
        target = sample_batch["target"].detach().cpu().numpy().squeeze()
        for sample_idx in range(min(batch_size, max_samples)):
            log_dict = {
                "sample_idx": sample_idx,
                "pred": wandb.Histogram(preds),
                "target": wandb.Histogram(target),
            }

            # GOAL IMAGE
            # log video, image, audio, and target
            image = (
                sample_batch["image"][sample_idx]
                .detach()
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
            )
            # convert to PIL image
            image = Image.fromarray((image * 255).astype(np.uint8))
            log_dict["goal_image"] = wandb.Image(image)

            # INPUT VIDEO
            video = sample_batch["video"][sample_idx].detach().cpu()
            # unnormalize video frames and convert to numpy
            video_frames = [video[:, i, :, :] for i in range(video.shape[1])]
            video_frames = [
                invNormalize(frame).numpy().transpose(1, 2, 0) for frame in video_frames
            ]
            video_frames = [
                Image.fromarray((frame * 255).astype(np.uint8))
                for frame in video_frames
            ]
            video_frames[0].save(
                "./temp_imgs/sample_vid.gif",
                save_all=True,
                append_images=video_frames[1:],
                optimize=False,
                duration=500,
                loop=0,
            )
            log_dict["input_video"] = wandb.Video(
                "./temp_imgs/sample_vid.gif",
                fps=30,
                format="gif",
                caption=f"preds: {preds[sample_idx]:.2f}, target: {target[sample_idx]:.2f}",
            )

            # INPUT AUDIO
            if include_audio:
                audio = sample_batch["audio"][sample_idx].detach().cpu().numpy()
                # display melspec of audio
                plt.figure()
                s_db = librosa.amplitude_to_db(np.abs(audio[0]), ref=np.max)
                librosa.display.specshow(s_db, sr=16000, x_axis="time", y_axis="linear")
                plt.colorbar()
                plt.savefig("./temp_imgs/spec.png")
                plt.clf()
                plt.close()
                log_dict["audio"] = wandb.Image("./temp_imgs/spec.png")

            # ORIG AUDIO
            if include_orig_audio:
                orig_audio = (
                    sample_batch["orig_audio"][sample_idx].detach().cpu().numpy()
                )
                # display untransformed audio
                plt.figure()
                plt.plot(list(range(orig_audio.shape[0])), orig_audio)
                plt.ylim([1600, 2400])
                plt.savefig("./temp_imgs/orig_audio.png")
                plt.clf()
                plt.close()
                log_dict["orig_audio"] = wandb.Image("./temp_imgs/orig_audio.png")

            # save to wandb
            wandb.log(log_dict)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_file", type=str, default="./configs/av-avid-p-c.yaml"
    )
    parser.add_argument("--extract_dir", type=str, default="./prep_data/avid_prep")
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=47)
    parser.add_argument("--frozen_backbone", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frozen_learning_rate", type=float, default=0.001)
    parser.add_argument("--unfrozen_learning_rate", type=float, default=0.0001)
    parser.add_argument("--frozen_epochs", type=int, default=10)
    parser.add_argument("--unfrozen_epochs", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=4)
    parser.add_argument("--num_samples_log", type=int, default=15)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "vis"])
    return parser.parse_args()


def main():
    args = parse_arguments()

    # load model configs
    model_config = yaml.safe_load(open(args.model_config_file, "r"))
    model_args = model_config["model"]["args"]
    model_name = model_config["model_name"]

    # data / misc args
    extract_dir = args.extract_dir
    validation_split = args.validation_split
    random_seed = args.random_seed
    mode = args.mode
    log_freq = args.log_freq
    num_samples_log = args.num_samples_log

    # training args
    BATCH_SIZE = args.batch_size
    FROZEN_LR = args.frozen_learning_rate
    UNFROZEN_LR = args.unfrozen_learning_rate
    FROZEN_EPOCHS = args.frozen_epochs
    UNFROZEN_EPOCHS = args.unfrozen_epochs

    if mode == "train":
        wandb.init(project="finetune-teleop", entity="contact-mic", name=model_name)
    elif mode == "vis":
        wandb.init(
            project="finetune-teleop", entity="contact-mic", name=f"{model_name}_vis"
        )
    wandb.config.update(args)
    wandb.config.update(model_config)

    # set random seed
    set_seeds(random_seed)

    # MODEL
    model = AvidR3M(**model_args).to(DEVICE)
    wandb.watch(model)

    # DATA
    print(f"\n##### DATA #####")
    include_audio = False if model_args["modality"] == "video" else True
    include_orig_audio = (
        False if mode == "train" or model_args["modality"] == "video" else True
    )
    collate_func = partial(
        multi_collate,
        device=DEVICE,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    # get dataset
    image_paths, curr_target_idxs, traj_ids_idxs = load_target_data(extract_dir)
    transforms = backbone_transforms(
        model_args["avid_name"],
        model_args["avid_cfg_path"],
        audio_transform=model_config["dataset"]["audio_transform"],
    )
    dataset = TeleopCompletionDataset(
        image_paths,
        curr_target_idxs,
        transforms,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    train_loader, val_loader = get_dataloaders(
        dataset,
        traj_ids_idxs=traj_ids_idxs,
        collate_fn=collate_func,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
    )
    print(
        f"Train size: {len(train_loader.sampler)}, Val size: {len(val_loader.sampler)}"
    )

    # reset seed due to model differences in random initialization
    set_seeds(random_seed)

    if mode == "train":
        # TRAIN
        criterion = torch.nn.MSELoss()

        # set optimizer and scheduler for unfrozen parameters
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=FROZEN_LR
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=4
        )

        best_val_loss = np.inf
        for epoch in range(FROZEN_EPOCHS + UNFROZEN_EPOCHS):
            if epoch == FROZEN_EPOCHS:
                model.unfreeze_backbone()
                optimizer = torch.optim.SGD(model.parameters(), lr=UNFROZEN_LR)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", patience=4
                )
                print(f"##### UNFREEZING BACKBONE #####")

            train_loss = train_loop(model, optimizer, criterion, train_loader)
            print(f"Epoch: {epoch}, Training Loss: {train_loss}")

            val_loss = eval_loop(model, criterion, val_loader)
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
                vis_preds(
                    model,
                    val_loader,
                    max_samples=num_samples_log,
                    include_audio=include_audio,
                    include_orig_audio=include_orig_audio,
                )

    elif mode == "vis":
        # VISUALIZE
        model.load_state_dict(
            torch.load(model_config["model"]["checkpoint"], map_location=DEVICE)
        )
        model.eval()
        vis_preds(
            model,
            val_loader,
            max_samples=num_samples_log,
            include_audio=include_audio,
            include_orig_audio=include_orig_audio,
        )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
