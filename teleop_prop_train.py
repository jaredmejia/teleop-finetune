import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import tqdm
import wandb
import yaml

from teleop_prop_data import ConditionalTeleopDataset, load_target_data, get_dataloaders
from torch import optim, nn, utils
from torchvision import transforms as T
from PIL import Image

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import load_encoder, load_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

avid_cfg_path = "/home/vdean/jared_contact_mic/avid-glove/config/gloveaudio/avcat-avid-ft-jointloss.yaml"
avid_video_cfg_path = "/home/vdean/jared_contact_mic/avid-glove/config/gloveaudio/avcat-avid-ft-video.yaml"

# transforms for video to unnormalize image frames
invNormalize = T.Compose(
    [
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.22803, 1 / 0.22145, 1 / 0.216989]),
        T.Normalize(mean=[0 - 0.43216, 0 - 0.394666, 0 - 0.37645], std=[1.0, 1.0, 1.0]),
    ]
)

# function to set all seeds for torch and numpy
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_prep(avid_name, avid_cfg):
    avid_backbone = load_encoder(avid_name, avid_cfg)
    r3m_backbone = load_encoder("r3m", None)
    return avid_backbone, r3m_backbone


def backbone_transforms(avid_cfg):
    avid_transforms = load_transforms("avid-no-ft", avid_cfg)
    r3m_transforms = load_transforms("r3m", None)

    data_types = ["video", "audio", "image", "target", "orig_audio"]

    def backbone_transforms_f(data):
        data_t = {}
        for data_type in data_types:
            if data_type in ["video", "audio"]:
                data_t[data_type] = avid_transforms(data)[data_type]
            elif data_type in "image":
                data_t["image"] = r3m_transforms(data)["image"]
            else:
                data_t[data_type] = torch.tensor(data[data_type])
        return data_t

    return backbone_transforms_f


def multi_collate(batch):
    video_batch, img_batch, audio_batch, target_batch, orig_audio_batch = [], [], [], [], []
    include_video, include_img, include_audio = True, True, True
    for data in batch:
        if "video" not in data.keys() or not include_video:
            include_video = False
        else:
            video_batch.append(data["video"])

        if "image" not in data.keys() or not include_img:
            include_img = False
        else:
            img_batch.append(data["image"])

        if "audio" not in data.keys() or not include_audio:
            include_audio = False
        else:
            audio_batch.append(data["audio"])

        target_batch.append(data["target"])

        orig_audio_batch.append(data["orig_audio"])


    batched_data = {}
    if include_video:
        video_batch = torch.stack(video_batch, dim=0).to(device)
        batched_data["video"] = video_batch
    if include_img:
        img_batch = torch.stack(img_batch, dim=0).to(device)
        batched_data["image"] = img_batch
    if include_audio:
        audio_batch = torch.stack(audio_batch, dim=0).to(device)
        batched_data["audio"] = audio_batch

    target_batch = torch.stack(target_batch, dim=0).to(device)
    batched_data["target"] = target_batch

    orig_audio_batch = torch.stack(orig_audio_batch, dim=0)
    batched_data["orig_audio"] = orig_audio_batch

    return batched_data


class PropReg(nn.Module):
    def __init__(
        self,
        avid_backbone,
        r3m_backbone,
        frozen_backbone=True,
        avid_emb_dim=1024,
        r3m_emb_dim=512,
        hidden_dim=512,
        disable_backbone_dropout=True,
        modality="audio-video"
    ):
        super().__init__()
        self.avid_backbone = avid_backbone
        self.r3m_backbone = r3m_backbone
        self.frozen_backbone = frozen_backbone
        self.feat_fusion = nn.Sequential(
            nn.Linear(avid_emb_dim + r3m_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.modality = modality
        self.disable_backbone_dropout = disable_backbone_dropout

        if self.frozen_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def freeze_backbone(self):
        """Freeze the backbone parameters and disable dropout and batchnorm
        for the backbone.
        """
        # freeze parameters of the backbone
        for param in self.avid_backbone.parameters():
            param.requires_grad = False
        for param in self.r3m_backbone.parameters():
            param.requires_grad = False

        # disable dropout and batchnorm for backbone
        self.avid_backbone.eval()
        self.r3m_backbone.eval()

        self.frozen_backbone = True

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters and enable dropout and batchnorm
        for the backbone.
        """
        # unfreeze parameters of the backbone
        for param in self.avid_backbone.parameters():
            param.requires_grad = True
        for param in self.r3m_backbone.parameters():
            param.requires_grad = True

        # enable dropout and batchnorm for backbone
        self.avid_backbone.train()
        self.r3m_backbone.train()

        if self.disable_backbone_dropout:
            if self.modality == "audio-video":
                self.avid_backbone.avid_model.dropout.eval()
            else:
                self.avid_backbone.dropout.eval()

        self.frozen_backbone = False

    def set_train(self):
        """Set model to train mode. If frozen_backbone is True, only the
        feature fusion layer is set to train mode. Otherwise, all layers are
        set to train mode.
        """
        if self.frozen_backbone:
            self.feat_fusion.train()
        else:
            self.train()

    def forward(self, data):
        if self.frozen_backbone:
            with torch.no_grad():
                vid_aud_emb = self.avid_backbone(data)
                img_emb = self.r3m_backbone(data["image"])
        else:
            vid_aud_emb = self.avid_backbone(data)
            img_emb = self.r3m_backbone(data["image"])

        cat_emb = torch.cat((vid_aud_emb, img_emb), dim=1)
        pred = self.feat_fusion(cat_emb)

        return pred


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
    for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds, data["target"])
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data["target"].size(0)

    return running_loss / len(dataloader.sampler)


def eval_loop(model, criterion, dataloader, log_media=False):
    """Evaluate model on validation set
    
    Args:
        model (nn.Module): model to evaluate
        criterion (nn.Module): loss function
        dataloader (torch.utils.data.DataLoader): dataloader for validation set
        log_media (bool, optional): whether to log media. Defaults to False.
        
    Returns:
        float: average loss on validation set
    """
    model.eval()
    running_loss = 0
    with torch.no_grad():
        for idx, data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            preds = model(data)
            loss = criterion(preds, data["target"])
            running_loss += loss.item() * data["target"].size(0)

            if idx == 0 and log_media:
                sample_idx = np.random.randint(0, data["target"].size(0))
                # log video, image, audio, and target
                video = (
                    data["video"][sample_idx].detach().cpu()
                )  
                image = (
                    data["image"][sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
                )
                audio = data["audio"][sample_idx].detach().cpu().numpy()
                target = data["target"].detach().cpu().numpy().squeeze()
                preds = preds.detach().cpu().numpy().squeeze()

                # convert to PIL image
                image = Image.fromarray((image * 255).astype(np.uint8))

                # unnormalize video frames and convert to numpy
                video_frames = [video[:, i, :, :] for i in range(video.shape[1])]
                video_frames = [
                    invNormalize(frame).numpy().transpose(1, 2, 0)
                    for frame in video_frames
                ]
                video_frames = [
                    Image.fromarray((frame * 255).astype(np.uint8))
                    for frame in video_frames
                ]
                video_frames[0].save(
                    "./sample_vid.gif",
                    save_all=True,
                    append_images=video_frames[1:],
                    optimize=False,
                    duration=500,
                    loop=0,
                )

                # display melspec of audio
                plt.figure()
                s_db = librosa.amplitude_to_db(np.abs(audio[0]), ref=np.max)
                librosa.display.specshow(s_db, sr=16000, x_axis="time", y_axis="linear")
                plt.colorbar()
                plt.savefig("./spec.png")
                plt.clf()

                # save to wandb
                wandb.log(
                    {
                        "video": wandb.Video(
                            "./sample_vid.gif",
                            fps=30,
                            format="gif",
                            caption=f"preds: {preds[sample_idx]:.2f}, target: {target[sample_idx]:.2f}",
                        ),
                        "goal image": wandb.Image(image),
                        "audio": wandb.Image("./spec.png"),
                        "preds": wandb.Histogram(preds),
                        "target": wandb.Histogram(target),
                    }
                )

    return running_loss / len(dataloader.sampler)


def vis_loop(model, dataloader, max_samples=15):
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
                # log video, image, audio, and target
                video = (
                    sample_batch["video"][sample_idx].detach().cpu()
                )  
                image = (
                    sample_batch["image"][sample_idx].detach().cpu().numpy().transpose(1, 2, 0)
                )
                audio = sample_batch["audio"][sample_idx].detach().cpu().numpy()
                orig_audio = sample_batch["orig_audio"][sample_idx].detach().cpu().numpy()

                # convert to PIL image
                image = Image.fromarray((image * 255).astype(np.uint8))

                # unnormalize video frames and convert to numpy
                video_frames = [video[:, i, :, :] for i in range(video.shape[1])]
                video_frames = [
                    invNormalize(frame).numpy().transpose(1, 2, 0)
                    for frame in video_frames
                ]
                video_frames = [
                    Image.fromarray((frame * 255).astype(np.uint8))
                    for frame in video_frames
                ]
                video_frames[0].save(
                    "./sample_vid.gif",
                    save_all=True,
                    append_images=video_frames[1:],
                    optimize=False,
                    duration=500,
                    loop=0,
                )

                # display melspec of audio
                plt.figure()
                s_db = librosa.amplitude_to_db(np.abs(audio[0]), ref=np.max)
                librosa.display.specshow(s_db, sr=16000, x_axis="time", y_axis="linear")
                plt.colorbar()
                plt.savefig("./spec.png")
                plt.clf()

                # display untransformed audio
                plt.figure()
                plt.plot(list(range(orig_audio.shape[0])), orig_audio)
                plt.ylim([1600, 2400])
                plt.savefig("./orig_audio.png")
                plt.clf()
                plt.close()

                # save to wandb
                wandb.log(
                    {
                        "video": wandb.Video(
                            "./sample_vid.gif",
                            fps=30,
                            format="gif",
                            caption=f"preds: {preds[sample_idx]:.2f}, target: {target[sample_idx]:.2f}",
                        ),
                        "goal image": wandb.Image(image),
                        "audio": wandb.Image("./spec.png"),
                        "orig_audio": wandb.Image("./orig_audio.png"),
                        "preds": wandb.Histogram(preds),
                        "target": wandb.Histogram(target),
                        "sample_idx": sample_idx,
                    }
                )


def main():
    # argparse for extract_dir, validation_split, random_seed, frozen_backbone, BATCH_SIZE, LR, EPOCHS, model_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_dir", type=str, default="./prep_data/shf_prep")
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=47)
    parser.add_argument("--frozen_backbone", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--frozen_learning_rate", type=float, default=0.001)
    parser.add_argument("--unfrozen_learning_rate", type=float, default=0.0001)
    parser.add_argument("--frozen_epochs", type=int, default=10)
    parser.add_argument("--unfrozen_epochs", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="no_window_overlap")
    parser.add_argument("--log_freq", type=int, default=4)
    parser.add_argument("--disable_backbone_dropout", type=bool, default=True)
    parser.add_argument("--modality", type=str, default="audio-video", choices=["audio", "video", "audio-video"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "vis"])
    args = parser.parse_args()

    extract_dir = args.extract_dir
    validation_split = args.validation_split
    random_seed = args.random_seed
    frozen_backbone = args.frozen_backbone
    model_name = args.model_name
    disable_backbone_dropout = args.disable_backbone_dropout
    modality = args.modality
    mode = args.mode

    BATCH_SIZE = args.batch_size
    FROZEN_LR = args.frozen_learning_rate
    UNFROZEN_LR = args.unfrozen_learning_rate
    FROZEN_EPOCHS = args.frozen_epochs
    UNFROZEN_EPOCHS = args.unfrozen_epochs

    if mode == "train":
        wandb.init(project="finetune-teleop", entity="contact-mic", name=model_name)
        wandb.config.update(args)
    elif mode == "vis":
        wandb.init(project="finetune-teleop", entity="contact-mic", name=f"{model_name}_vis")
        wandb.config.update(args)

    if modality == "audio-video":
        avid_name = 'avid-no-ft'
        avid_cfg = yaml.safe_load(open(avid_cfg_path))
        avid_emb_dim = 1024
    elif modality == "video":
        avid_name = 'avid-no-ft-video'
        avid_cfg = yaml.safe_load(open(avid_video_cfg_path))
        avid_emb_dim = 512
    else:
        raise NotImplementedError

    set_seeds(random_seed)

    # MODEL
    avid_backbone, r3m_backbone = model_prep(avid_name, avid_cfg)
    model = PropReg(
        avid_backbone=avid_backbone,
        r3m_backbone=r3m_backbone,
        frozen_backbone=frozen_backbone,
        disable_backbone_dropout=disable_backbone_dropout,
        modality=modality,
        avid_emb_dim=avid_emb_dim,
    ).to(device)
    wandb.watch(model)


    # DATA
    print(f"\n##### DATA #####")
    image_paths, curr_target_idxs, traj_ids_idxs = load_target_data(extract_dir)
    transforms = backbone_transforms(avid_cfg)
    dataset = ConditionalTeleopDataset(image_paths, curr_target_idxs, transforms)
    train_loader, val_loader = get_dataloaders(
        dataset,
        traj_ids_idxs=traj_ids_idxs,
        collate_fn=multi_collate,
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

            if epoch % args.log_freq == 0:
                log_media = True
            else:
                log_media = False

            train_loss = train_loop(model, optimizer, criterion, train_loader)
            print(f"Epoch: {epoch}, Training Loss: {train_loss}")

            val_loss = eval_loop(model, criterion, val_loader, log_media=log_media)
            print(f"Epoch: {epoch}, Validation Loss: {val_loss}")

            wandb.log({"train/loss": train_loss, "val/loss": val_loss, "epoch": epoch})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"./models/best_model_{model_name}.pth")
            scheduler.step(val_loss)
    
    elif mode == "vis":
        # VISUALIZE
        model.load_state_dict(torch.load(f"./models/best_model_{model_name}.pth"))
        model.eval()
        vis_loop(model, val_loader)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
