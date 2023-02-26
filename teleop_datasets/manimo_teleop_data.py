import glob
import os

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

TRAINING_DEMOS = [1, 2, 3, 6, 8, 9, 12, 14, 15, 16, 17, 19]
VALIDATION_DEMOS = [5, 11, 18]


class ManimoTeleopDataset(Dataset):
    def __init__(
        self,
        demo_fnames,
        transforms,
        include_image=False,
        include_audio=False,
        include_orig_audio=False,
        window_dur=1,
        video_hz=30,
        num_img_frames=6,
        target_dims=None,
        cat_per_dim=2,
        action_thresh=1e-4,
    ):
        """Dataset for Manimo teleoperation data.

        Args:
            demo_fnames (list): List of paths to demo files.
            transforms (dict): Dictionary of transforms to apply to each modality.
            include_image (bool): Whether to include image data.
            include_audio (bool): Whether to include audio data.
            include_orig_audio (bool): Whether to include original audio data.
            window_dur (float): Duration of window to extract from each demo.
            video_hz (int): Video frame rate.
            num_img_frames (int): Number of image frames to extract from each window.
        """
        self.demo_fnames = demo_fnames
        self.transforms = transforms
        self.include_image = include_image
        self.include_audio = include_audio
        self.include_orig_audio = include_orig_audio
        self.window_dur = window_dur
        self.video_hz = video_hz
        self.num_img_frames = num_img_frames
        self.video_frame_stride = int(window_dur * video_hz / num_img_frames)
        self.num_video_frames = int(window_dur * video_hz)

        # discretize action space
        if target_dims is None:
            target_dims = [1, 2]
        self.target_dims = target_dims
        self.output_dim = len(target_dims)
        self.cat_per_dim = cat_per_dim
        self.action_thresh = action_thresh

        # pre-load all data
        self.demo_to_traj, self.idx_to_demo = self._pre_load_data(demo_fnames)

    def _pre_load_data(self, demo_fnames):
        """Pre-load all data from demo files.

        Args:
            demo_fnames (list): List of paths to demo files.

        Returns:
            list_of_dicts (list): List of dictionaries, each containing data for a single demo.
            idx_to_demo (list): List of demo indices for each observation.
            num_obs (int): Total number of observations.
        """
        demo_to_traj = {}  # maps demo_fname to dict of data
        idx_to_demo = []  # each idx is a tuple of (demo_fname, obs_idx)
        for demo_fname in demo_fnames:

            traj_dict = {}
            with h5py.File(demo_fname, "r") as f:
                # VIDEO DATA FOR ENTIRE EPISODE
                video_arr = self._get_video_arr(f["videos"]["cam0c"])
                traj_dict["video"] = video_arr

                # AUDIO DATA FOR ENTIRE EPISODE
                if self.include_audio:
                    audio = f["audio"][:]
                    traj_dict["audio"] = audio

                # EEF POSITION DATA FOR ENTIRE EPISODE
                eef_pos = f["eef_pos"][:]
                target_action = eef_pos[1:] - eef_pos[:-1]
                target_action = target_action[
                    :, self.target_dims
                ]  # keep only target dimensions

                # discretize action ([left, none], [done, none]) to .3mm increments
                target_action = self._discretize_actions(target_action)
                traj_dict["target_action"] = target_action.astype(float)

                # hold entire trajectory
                demo_to_traj[demo_fname] = traj_dict

                # matching indices to demo file
                demo_len = video_arr.shape[0] - 1
                idx_to_demo += [(demo_fname, i) for i in range(1, demo_len)]

        return demo_to_traj, idx_to_demo

    def _discretize_actions(self, target_action):
        """Discretize action space.

        Args:
            target_action (np.array): Array of shape (num_obs, self.output_dim) containing target actions.

        Returns:
            target_action (np.array): Array of shape (num_obs, self.output_dim) containing discretized target actions.
        """
        if self.cat_per_dim == 2:
            # discretize action ([left, none], [down, none]) to .3mm increments
            target_action = np.stack(
                (
                    np.where(target_action[:, 0] > self.action_thresh, 1, 0),
                    np.where(target_action[:, 1] < -self.action_thresh, 1, 0),
                ),
                axis=1,
            )
        elif self.cat_per_dim == 3:
            # discretize action ([left, right, none], [up, down, none]) to .3mm increments
            target_action = np.where(
                target_action > self.action_thresh, 1, 0
            ) + np.where(target_action < self.action_thresh, -1, 0)
        else:
            raise ValueError("Invalid number of categories per dimension.")
        return target_action

    def _get_video_arr(self, video_obj):
        serialized_video = np.array(video_obj)
        video_arr = iio.imread(serialized_video.tobytes(), index=None, extension=".mp4")
        return video_arr

    def __len__(self):
        """Get number of observations in dataset."""
        return len(self.idx_to_demo)

    def _get_video_clip(self, video_arr, video_clip_end):
        """Get video clip and goal image from pre-loaded video array.

        Args:
                video_arr (np.array): Array of video frames.
                video_clip_end (int): End index of video clip.

        Returns:
            video_clip (list): List of PIL images.
            goal_image (PIL.Image): Goal image.
        """
        video_clip_start = max(0, video_clip_end - self.num_video_frames)
        video_clip = list(
            video_arr[video_clip_start : video_clip_end : self.video_frame_stride]
        )

        # convert frames to PIL images
        video_clip = [ToPILImage()(frame[:, :, ::-1]) for frame in video_clip]

        if self.include_image:
            goal_image = video_arr[-1][:, :, ::-1]
            goal_image = ToPILImage()(goal_image)
            return video_clip, goal_image

        return video_clip, None

    def __getitem__(self, idx):
        """Get data for a single observation.

        Args:
            idx (int): Index of observation.

        Returns:
            data_t (dict): Dictionary of transformed data.
        """
        demo_fname, obs_idx = self.idx_to_demo[idx]
        data = {}

        # get video clip and goal image
        video_clip, goal_image = self._get_video_clip(
            self.demo_to_traj[demo_fname]["video"], obs_idx + 1
        )
        data["video"] = video_clip

        # goal image
        if self.include_image:
            data["image"] = goal_image

        # get audio
        if self.include_audio:
            audio = self.demo_to_traj[demo_fname]["audio"][obs_idx]
            data["audio"] = audio

            if self.include_orig_audio:
                orig_audio = np.mean(audio, axis=0)
                data["orig_audio"] = orig_audio

        # get target action
        target_action = self.demo_to_traj[demo_fname]["target_action"][obs_idx]
        data["target"] = target_action

        data_t = self.transforms(data)

        return data_t


def get_dataloaders(
    demo_dir,
    transforms,
    collate_func,
    batch_size=32,
    include_image=False,
    include_audio=False,
    include_orig_audio=False,
    window_dur=1,
    video_hz=30,
    num_img_frames=6,
):
    """Returns train and validation dataloaders for the Manimo Teleop dataset.

    Args:
        demo_dir (str): Path to directory containing the Manimo Teleop dataset.
        transforms (torchvision.transforms.Compose): Transforms to apply to the data.
        batch_size (int, optional): Batch size. Defaults to 32.
        include_image (bool, optional): Whether to include the goal image in the data. Defaults to False.
        include_audio (bool, optional): Whether to include the audio in the data. Defaults to False.
        include_orig_audio (bool, optional): Whether to include the original audio in the data. Defaults to False.
        window_dur (int, optional): Duration of the window to extract from the video. Defaults to 1.
        video_hz (int, optional): Video frame rate. Defaults to 30.
        num_img_frames (int, optional): Number of image frames to extract from the video. Defaults to 6.

    Returns:
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
    """
    demo_fnames = demodir_to_fpaths(demo_dir)
    train_demo_fnames = [demo_fnames[demo_num] for demo_num in TRAINING_DEMOS]
    val_demo_fnames = [demo_fnames[demo_num] for demo_num in VALIDATION_DEMOS]

    train_dataset = ManimoTeleopDataset(
        train_demo_fnames,
        transforms,
        include_image=include_image,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
        window_dur=window_dur,
        video_hz=video_hz,
        num_img_frames=num_img_frames,
    )
    val_dataset = ManimoTeleopDataset(
        val_demo_fnames,
        transforms,
        include_image=include_image,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
        window_dur=window_dur,
        video_hz=video_hz,
        num_img_frames=num_img_frames,
    )
    # shuffle validation once for visualization variety
    val_dataset = torch.utils.data.Subset(
        val_dataset, np.random.permutation(len(val_dataset))
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func
    )

    return train_dataloader, val_dataloader


def demodir_to_fpaths(demo_dir):
    """Returns a dictionary mapping demo numbers to their filepaths.

    Args:
        demo_dir (str): Path to directory containing the Manimo Teleop dataset.

    Returns:
        fpath_dict (dict): Dictionary mapping demo numbers to their filepaths.
    """
    demo_fpaths = glob.glob(os.path.join(demo_dir, "*.h5"))
    fpath_dict = {}
    for demo_fpath in demo_fpaths:
        demo_fname = os.path.basename(demo_fpath)
        demo_num = int(demo_fname.split(".")[0].split("-")[-1])
        fpath_dict[demo_num] = demo_fpath
    return fpath_dict
