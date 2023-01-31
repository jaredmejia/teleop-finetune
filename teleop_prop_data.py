import argparse
import numpy as np
import os
import pickle

import PIL.Image as Image

from torch import utils

CAM_FPS = 30
NUM_IMG_FRAMES = 6
WINDOW_DUR = 3


class TeleopCompletionDataset(utils.data.Dataset):
    def __init__(
        self,
        img_paths,
        curr_target_idxs,
        transforms,
        include_audio=False,
        include_orig_audio=False,
    ):
        self.img_paths = img_paths
        self.curr_target_idxs = curr_target_idxs
        self.transforms = transforms
        self.include_audio = include_audio
        self.include_orig_audio = include_orig_audio

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        data = {}

        # INPUT VIDEO
        video_data = [
            Image.open(img_path)
            for img_path in self.img_paths[idx][
                :: WINDOW_DUR * CAM_FPS // NUM_IMG_FRAMES
            ]
        ]
        data["video"] = video_data

        # TARGET IMAGE / TARGET VALUE
        curr_idx, target_idx = self.curr_target_idxs[idx]
        target_img_idx = idx + target_idx - curr_idx
        target_img_path = self.img_paths[target_img_idx][
            -1
        ]  # get last image from list of images
        target_img = Image.open(target_img_path)
        target_val = curr_idx / target_idx
        data["image"] = target_img
        data["target"] = target_val

        if self.include_audio:
            # AUDIO
            txt_list = []
            for i, img_path in enumerate(self.img_paths[idx]):
                txt_path = f"{img_path[:-4]}txt"
                txt_arr = np.loadtxt(txt_path)
                txt_arr = txt_arr.T
                txt_list.append(txt_arr)

            audio_data = np.concatenate(txt_list, axis=1)
            data["audio"] = audio_data

        if self.include_orig_audio:
            # ORIGINAL AUDIO
            avg_audio_data = np.mean(audio_data, axis=0)
            # pad avg audio data with the average value up to size (96000,) at the front of the array
            if avg_audio_data.shape[0] < 96000:
                avg_audio_data = np.concatenate(
                    (
                        np.full(
                            (96000 - avg_audio_data.shape[0]), np.mean(avg_audio_data)
                        ),
                        avg_audio_data,
                    )
                )
            else:
                avg_audio_data = avg_audio_data[-96000:]

            data["orig_audio"] = avg_audio_data

        data_t = self.transforms(data)

        return data_t


def get_dataloaders(
    dataset, traj_ids_idxs, collate_fn, batch_size=128, validation_split=0.2
):
    split_traj_idx = int(np.floor((1 - validation_split) * len(traj_ids_idxs)))

    split = traj_ids_idxs[split_traj_idx][1]
    print(
        f"Splitting at {split_traj_idx}th traj / {len(traj_ids_idxs)} trajectories, {split}-th idx total\n"
    )

    # build data loaders based on split
    train_loader = utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=utils.data.SubsetRandomSampler(range(split)),
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_loader = utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=utils.data.SubsetRandomSampler(range(split, len(dataset))),
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_teleop_data_targets(teleop_paths, teleop_dir):
    """Get data and targets for teleop data."""
    img_paths = []
    curr_target_idxs = []
    traj_ids_idxs = []
    global_idx = 0

    num_frames_window = int(WINDOW_DUR * CAM_FPS)
    stride_len = num_frames_window // NUM_IMG_FRAMES

    # reording teleop paths for train/val split
    train_idxs = []
    val_idxs = []
    for i in range(0, 80, 20):
        train_idxs.extend(range(i, i + 15))
        val_idxs.extend(range(i + 15, i + 20))

    train_teleop_paths = [teleop_paths[i] for i in train_idxs]
    val_teleop_paths = [teleop_paths[i] for i in val_idxs]
    teleop_paths = train_teleop_paths + val_teleop_paths

    for path in teleop_paths:
        data_path = os.path.join(teleop_dir, path["traj_id"])
        traj_img_paths = [
            os.path.join(data_path, img_file) for img_file in path["cam0c"]
        ]
        traj_ids_idxs.append((path["traj_id"], global_idx))

        idxs = list(range(stride_len, len(traj_img_paths), stride_len))
        for i in range(len(idxs)):
            img_path_cat = traj_img_paths[max(0, idxs[i] - num_frames_window) : idxs[i]]
            img_paths.append(img_path_cat)
            curr_target_idxs.append((i, len(idxs) - 1))
            print(
                f"idx: {i}, window size: {idxs[i] - max(0, idxs[i] - num_frames_window)}"
            )

        global_idx += len(idxs)

        assert global_idx == len(img_paths), f"{global_idx} != {len(img_paths)}"

    assert len(img_paths) == len(curr_target_idxs)

    return img_paths, curr_target_idxs, traj_ids_idxs


def load_target_data(extract_dir):
    curr_target_idxs_path = os.path.join(extract_dir, "curr_target_idxs.pkl")
    img_paths_path = os.path.join(extract_dir, "image_paths.pkl")
    traj_ids_idxs_path = os.path.join(extract_dir, "traj_ids_idxs.pkl")
    with open(curr_target_idxs_path, "rb") as f:
        curr_target_idxs = pickle.load(f)
    with open(img_paths_path, "rb") as f:
        image_paths = pickle.load(f)
    with open(traj_ids_idxs_path, "rb") as f:
        traj_ids_idxs = pickle.load(f)

    return image_paths, curr_target_idxs, traj_ids_idxs


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prep_data",
        help="Directory to save the output",
    )
    parser.add_argument(
        "--teleop_pickle",
        type=str,
        default="/home/vdean/franka_learning_jared/jared_chopping_exps_v4.pkl",
        help="Pickle file with teleop data",
    )
    parser.add_argument(
        "--teleop_dir",
        type=str,
        default="/home/vdean/franka_demo/logs/jared_chopping_exps_v4/",
        help="Directory with teleop data",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    teleop_pickle = args.teleop_pickle
    teleop_dir = args.teleop_dir

    with open(teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    image_paths, curr_target_idxs, traj_ids_idxs = get_teleop_data_targets(
        teleop_paths, teleop_dir
    )
    print(f"Total number of samples: {len(image_paths)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths_fn = os.path.join(output_dir, "image_paths.pkl")
    print(f"Saving image_paths to {image_paths_fn}")
    with open(image_paths_fn, "wb") as f:
        pickle.dump(image_paths, f)

    curr_target_idxs_fn = os.path.join(output_dir, "curr_target_idxs.pkl")
    print(f"Saving curr_target_idxs to {curr_target_idxs_fn}")
    with open(curr_target_idxs_fn, "wb") as f:
        pickle.dump(curr_target_idxs, f)

    traj_ids_idxs_fn = os.path.join(output_dir, "traj_ids_idxs.pkl")
    print(f"Saving traj_ids_idxs to {traj_ids_idxs_fn}")
    with open(traj_ids_idxs_fn, "wb") as f:
        pickle.dump(traj_ids_idxs, f)


if __name__ == "__main__":
    main()
