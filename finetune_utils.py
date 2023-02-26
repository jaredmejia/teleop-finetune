import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

import wandb

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import load_transforms

# define __all__ for all functions and classes in this file
__all__ = [
    "set_seeds",
    "backbone_transforms",
    "multi_collate",
    "invNormalize",
]


# transforms for video to unnormalize image frames
invNormalize = T.Compose(
    [
        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1 / 0.22803, 1 / 0.22145, 1 / 0.216989]),
        T.Normalize(mean=[0 - 0.43216, 0 - 0.394666, 0 - 0.37645], std=[1.0, 1.0, 1.0]),
    ]
)

# function to set all seeds for torch and numpy
def set_seeds(seed):
    """Set all seeds for torch and numpy.

    Args:
        seed (int): seed to set
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backbone_transforms(model_name, cfg):
    """Load transforms for backbone model.

    Args:
        model_name (str): name of model
        cfg (dict): configuration for model

    Returns:
        backbone_transforms_f (function): function to apply transforms to data
    """
    transforms = load_transforms(model_name, cfg)

    def backbone_transforms_f(data, inference=False, device="cuda:0"):
        data_t = transforms(data)

        for data_type in data.keys():
            if data_type not in data_t:
                data_t[data_type] = torch.tensor(data[data_type])

        if inference:
            for data_type in data_t:
                if data_type != "target":
                    data_t[data_type] = data_t[data_type].unsqueeze(0).to(device)

        return data_t

    return backbone_transforms_f


def multi_discrete_loss(num_dim=2, output_per_dim=2):
    if output_per_dim == 2:
        functional_criterion = F.binary_cross_entropy_with_logits
    else:
        functional_criterion = F.cross_entropy

    def criterion(pred, target):
        for i in range(num_dim):
            if i == 0:
                loss = functional_criterion(pred[:, i], target[:, i])
            else:
                loss += functional_criterion(pred[:, i], target[:, i])
        return loss

    return criterion


### PREPARING DATA FOR TRAINING ###


def multi_collate(
    batch, device=None, include_image=True, include_audio=True, include_orig_audio=False
):
    """Collate data for training.

    Args:
        batch (list): list of data
        device (torch.device): device to move data to
        include_image (bool): whether to include image data
        include_audio (bool): whether to include audio data
        include_orig_audio (bool): whether to include original audio data

    Returns:
        batched_data (dict): batched data
    """
    video_batch, img_batch, audio_batch, target_batch, orig_audio_batch = (
        [],
        [],
        [],
        [],
        [],
    )
    for data in batch:
        video_batch.append(data["video"])
        target_batch.append(data["target"])

        if include_image:
            img_batch.append(data["image"])

        if include_audio:
            audio_batch.append(data["audio"])

        if include_orig_audio:
            orig_audio_batch.append(data["orig_audio"])

    batched_data = {}
    video_batch = torch.stack(video_batch, dim=0).to(device)
    batched_data["video"] = video_batch

    target_batch = torch.stack(target_batch, dim=0).to(device)
    batched_data["target"] = target_batch

    if include_image:
        img_batch = torch.stack(img_batch, dim=0).to(device)
        batched_data["image"] = img_batch

    if include_audio:
        audio_batch = torch.stack(audio_batch, dim=0).to(device)
        batched_data["audio"] = audio_batch

    if include_orig_audio:
        orig_audio_batch = torch.stack(orig_audio_batch, dim=0)
        batched_data["orig_audio"] = orig_audio_batch

    return batched_data


def prepare_data(
    dataset_cfg,
    transforms,
    batch_size,
    collate_func,
    include_image=True,
    include_audio=False,
    include_orig_audio=False,
    window_dur=1,
    video_hz=30,
    num_img_frames=6,
    num_valid_trajs=20,
):
    """Prepare data for training.

    Args:
        dataset_cfg (dict): dataset configuration
        transforms (dict): transforms for data
        batch_size (int): batch size
        collate_func (function): function to collate data
        include_image (bool, optional): whether to include image data. Defaults to True.
        include_audio (bool, optional): whether to include audio data. Defaults to False.
        include_orig_audio (bool, optional): whether to include original audio data. Defaults to False.
        window_dur (int, optional): window duration for audio data. Defaults to 1.
        video_hz (int, optional): video frame rate. Defaults to 30.
        num_img_frames (int, optional): number of image frames. Defaults to 6.
        num_valid_trajs (int, optional): number of validation trajectories. Defaults to 20.

    Returns:
        train_dataloader (DataLoader): dataloader for training data
        val_dataloader (DataLoader): dataloader for validation data
    """
    dataset_name = dataset_cfg["name"]
    if dataset_name == "manimo-teleop":
        from teleop_datasets.manimo_teleop_data import get_dataloaders

        demo_dir = dataset_cfg["demo_dir"]
        train_loader, val_loader = get_dataloaders(
            demo_dir,
            transforms,
            collate_func,
            batch_size=batch_size,
            include_image=include_image,
            include_audio=include_audio,
            include_orig_audio=include_orig_audio,
            window_dur=window_dur,
            video_hz=video_hz,
            num_img_frames=num_img_frames,
        )

    else:
        from teleop_datasets.teleop_prop_data import get_dataloaders, load_target_data

        # get dataset
        extract_dir = dataset_cfg["extract_dir"]
        (
            image_paths,
            curr_target_idxs,
            traj_ids_idxs,
            actions,
            traj_ids,
        ) = load_target_data(extract_dir)

        if dataset_name == "teleop-completion":
            from teleop_datasets.teleop_prop_data import TeleopCompletionDataset

            dataset = TeleopCompletionDataset(
                image_paths,
                curr_target_idxs,
                transforms,
                include_audio=include_audio,
                include_orig_audio=include_orig_audio,
            )
        elif dataset_name == "teleop-action":
            from teleop_datasets.teleop_prop_data import TeleopActionDataset

            dataset = TeleopActionDataset(
                image_paths,
                curr_target_idxs,
                traj_ids,
                actions,
                transforms,
                include_image=include_image,
                include_audio=include_audio,
                include_orig_audio=include_orig_audio,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        train_loader, val_loader = get_dataloaders(
            dataset,
            traj_ids_idxs=traj_ids_idxs,
            collate_fn=collate_func,
            batch_size=batch_size,
            num_valid_trajs=num_valid_trajs,
        )

    print(
        f"Train size: {len(train_loader.sampler)}, Val size: {len(val_loader.sampler)}"
    )

    return train_loader, val_loader


### VISUALIZATION FUNCTIONS ###


def vis_batch(
    dataloader,
    model=None,
    model_arch="AvidR3M",
    max_samples=15,
    include_image=True,
    include_audio=False,
    include_orig_audio=False,
    output_dim=1,
):
    """Visualize a batch of data

    Args:
        dataloader (torch.utils.data.DataLoader): dataloader for dataset
        model (nn.Module, optional): model to use for predictions. Defaults to None.
        model_arch (str, optional): model architecture. Defaults to "AvidR3M".
        max_samples (int, optional): max number of samples to visualize. Defaults to 15.
        include_image (bool, optional): whether to include image in visualization. Defaults to True.
        include_audio (bool, optional): whether to include audio in visualization. Defaults to False.
        include_orig_audio (bool, optional): whether to include original audio in visualization. Defaults to False.
        output_dim (int, optional): output dimension. Defaults to 1.
    """
    if model:
        model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        batch_size = sample_batch["target"].size(0)
        target = sample_batch["target"].detach().cpu().numpy().squeeze()

        if model:
            preds = model(sample_batch).detach().cpu().numpy().squeeze()

        for sample_idx in range(min(batch_size, max_samples)):
            if output_dim == 1:
                log_dict = {
                    "sample_idx": sample_idx,
                    "target": wandb.Histogram(target),
                }
                caption = f"target: {target[sample_idx]:.2f}"

                if model:
                    log_dict["pred"] = wandb.Histogram(preds)
                    caption += f", pred: {preds[sample_idx]:.2f}"

            elif output_dim == 2:  # 2-dim delta eef
                log_dict = {}
                caption = f"delta_y: {target[sample_idx][0]:.2f}, delta_z: {target[sample_idx][1]:.2f}"
                if sample_idx == 0:
                    for dim in range(output_dim):
                        if model:
                            log_dict[f"(target-pred)_{dim}"] = wandb.Histogram(
                                np.abs(target[:, dim] - preds[:, dim])
                            )
                        else:
                            log_dict[f"target_{dim}"] = wandb.Histogram(target[:, dim])

                if model:
                    caption += f"\npred_delta_y: {preds[sample_idx][0]:.2f}, pred_delta_z: {preds[sample_idx][1]:.2f}"

            else:
                log_dict = {"sample_idx": sample_idx}
                caption = ""
                if sample_idx == 0 and model:
                    # for each dim in pred and target log wandb histogram
                    for dim in range(output_dim):
                        target_pred_diff = target[:, dim] - preds[:, dim]
                        angle_diff = np.arctan2(
                            np.sin(target_pred_diff), np.cos(target_pred_diff)
                        )
                        log_dict[f"(target-pred)_{dim}"] = wandb.Histogram(angle_diff)

            # INPUT VIDEO
            video = sample_batch["video"][sample_idx].detach().cpu()
            log_dict["input_video"] = write_video(
                video,
                model_arch=model_arch,
                name="./temp_imgs/sample_vid.gif",
                fps=30,
                duration=500,
                caption=caption,
                wandb_log=True,
            )

            if include_image:
                # GOAL IMAGE
                image = sample_batch["image"][sample_idx].detach().cpu()
                # convert to PIL image
                log_dict["image"] = write_image(image, caption=caption, wandb_log=True)

            # INPUT AUDIO
            if include_audio:
                audio = sample_batch["audio"][sample_idx].detach().cpu()
                log_dict["audio"] = write_audio_spec(
                    audio, name="./temp_imgs/spec.png", caption=caption, wandb_log=True
                )

            # ORIG AUDIO
            if include_orig_audio:
                orig_audio = sample_batch["orig_audio"][sample_idx].detach().cpu()
                log_dict["orig_audio"] = write_audio_waveform(
                    orig_audio,
                    name="./temp_imgs/orig_audio.png",
                    caption=caption,
                    wandb_log=True,
                )

            # save to wandb
            wandb.log(log_dict)


def write_video(
    video,
    model_arch="AvidR3M",
    name="./temp_imgs/sample_vid.gif",
    fps=30,
    duration=500,
    caption=None,
    wandb_log=False,
):
    """Write video to file

    Args:
        video (torch.Tensor): video tensor
        model_arch (str, optional): model architecture. Defaults to "AvidR3M".
        name (str, optional): name of file to save. Defaults to "./temp_imgs/sample_vid.gif".
        fps (int, optional): frames per second. Defaults to 30.
        duration (int, optional): duration of video in milliseconds. Defaults to 500.
        caption (str, optional): caption for video. Defaults to None.
        wandb_log (bool, optional): whether to log to wandb. Defaults to False.

    Returns:
        wandb.Video: wandb video object
    """
    if model_arch == "MultiSensoryAttention":
        video_frames = [
            video[i, :, :, :].numpy().transpose(1, 2, 0) for i in range(video.shape[0])
        ]
    else:
        video_frames = [video[:, i, :, :] for i in range(video.shape[1])]

        # unnormalize video frames and convert to numpy
        video_frames = [
            invNormalize(frame).numpy().transpose(1, 2, 0) for frame in video_frames
        ]

    video_frames = [
        Image.fromarray((frame * 255).astype(np.uint8)) for frame in video_frames
    ]
    video_frames[0].save(
        name,
        save_all=True,
        append_images=video_frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )

    if wandb_log:
        return wandb.Video(name, fps=fps, format="gif", caption=caption)


def write_image(image, caption=None, name=None, wandb_log=False):
    """Write image to file

    Args:
        image (torch.Tensor): image tensor
        caption (str, optional): caption for image. Defaults to None.
        name (str, optional): name of file to save. Defaults to None.
        wandb_log (bool, optional): whether to log to wandb. Defaults to False.

    Returns:
        wandb.Image: wandb image object
    """
    image = image.numpy().transpose(1, 2, 0)
    image = Image.fromarray((image).astype(np.uint8))

    if wandb_log:
        return wandb.Image(image, caption=caption)
    else:
        image.save(name)


def write_audio_spec(
    audio_spec, name="./temp_imgs/spec.png", caption=None, wandb_log=False
):
    """Write audio spectrogram to file

    Args:
        audio_spec (torch.Tensor): audio spectrogram tensor
        name (str, optional): name of file to save. Defaults to "./temp_imgs/spec.png".
        caption (str, optional): caption for image. Defaults to None.
        wandb_log (bool, optional): whether to log to wandb. Defaults to False.

    Returns:
        wandb.Image: wandb image object
    """
    audio_spec = audio_spec.numpy()
    plt.figure()
    s_db = librosa.amplitude_to_db(np.abs(audio_spec[0]), ref=np.max)
    librosa.display.specshow(s_db, sr=16000, x_axis="time", y_axis="linear")
    plt.colorbar()
    plt.savefig(name)
    plt.clf()
    plt.close()

    if wandb_log:
        return wandb.Image(name, caption=caption)


def write_audio_waveform(
    audio_waveform,
    x_vals=None,
    name="./temp_imgs/waveform.png",
    caption=None,
    wandb_log=False,
):
    """Write audio waveform to file

    Args:
        audio_waveform (torch.Tensor): audio waveform tensor
        x_vals (list, optional): x values for audio waveform. Defaults to None.
        name (str, optional): name of file to save. Defaults to "./temp_imgs/waveform.png".
        caption (str, optional): caption for image. Defaults to None.
        wandb_log (bool, optional): whether to log to wandb. Defaults to False.

    Returns:
        wandb.Image: wandb image object
    """
    audio_waveform = audio_waveform.numpy().squeeze()
    if x_vals is None:
        x_vals = list(range(audio_waveform.shape[0]))

    plt.figure()
    plt.plot(x_vals, audio_waveform)
    plt.ylim([1600, 2400])
    plt.savefig(name)
    plt.clf()
    plt.close()

    if wandb_log:
        return wandb.Image(name, caption=caption)
