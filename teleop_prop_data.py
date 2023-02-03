import argparse
import numpy as np
import os
import pickle

import PIL.Image as Image

from torch import utils

CAM_FPS = 30
NUM_IMG_FRAMES = 6
WINDOW_DUR = 3
AUDIO_FPS = 32000


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

    def __getitem__(self, idx, target_img_path=None):
        data = {}

        # INPUT VIDEO
        video_data = [
            Image.open(img_path)
            for img_path in self.img_paths[idx][
                :: WINDOW_DUR * CAM_FPS // NUM_IMG_FRAMES
            ]
        ]
        data["video"] = video_data

        if target_img_path is not None:
            target_img = Image.open(target_img_path)
            data["image"] = target_img
        else:
            assert (
                self.curr_target_idxs is not None
            ), "must provide curr_target_idxs for training"

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
            if avg_audio_data.shape[0] < AUDIO_FPS * WINDOW_DUR:
                avg_audio_data = np.concatenate(
                    (
                        np.full(
                            (AUDIO_FPS * WINDOW_DUR - avg_audio_data.shape[0]),
                            np.mean(avg_audio_data),
                        ),
                        avg_audio_data,
                    )
                )
            else:
                avg_audio_data = avg_audio_data[-AUDIO_FPS * WINDOW_DUR :]

            data["orig_audio"] = avg_audio_data

        data_t = self.transforms(data)

        return data_t


def get_dataloaders(
    dataset, traj_ids_idxs, collate_fn, batch_size=128, num_valid_trajs=20
):
    """Get train and validation dataloaders."""
    split = traj_ids_idxs[-num_valid_trajs][1]
    print(
        f"Num train trajectories: {len(traj_ids_idxs[:-num_valid_trajs])}, Num valid trajectories: {len(traj_ids_idxs[-num_valid_trajs:])}\nvalid begins at {split}-th idx / {len(dataset)} total\n"
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


def get_teleop_data_targets(teleop_paths, teleop_dir, mode="train"):
    """Get data and targets for teleop data."""
    img_paths = []  # sequential list of image paths
    curr_target_idxs = []  # list of tuples (relative_idx, relative_target_idx)
    traj_ids_idxs = []  # list of tuples (traj_id, traj_start_idx)
    actions = {}  # dict of lists of actions for each trajectory
    traj_ids = []  # list of tuples (traj_id, relative_idx))

    num_frames_window = int(WINDOW_DUR * CAM_FPS)
    stride_len = num_frames_window // NUM_IMG_FRAMES

    # reording teleop paths for train/val split
    # ensures that training set has 15 trajectories for each object
    # ensures validation set has unseen 5 trajectories for each object
    train_idxs = []
    val_idxs = []
    for i in range(0, 80, 20):
        train_idxs.extend(range(i, i + 15))
        val_idxs.extend(range(i + 15, i + 20))

    train_teleop_paths = [teleop_paths[i] for i in train_idxs]
    val_teleop_paths = [teleop_paths[i] for i in val_idxs]
    teleop_paths = train_teleop_paths + val_teleop_paths

    global_idx = 0
    for path in teleop_paths:
        data_path = os.path.join(teleop_dir, path["traj_id"])
        traj_img_paths = [
            os.path.join(data_path, img_file) for img_file in path["cam0c"]
        ]
        traj_ids_idxs.append((path["traj_id"], global_idx))
        actions[path["traj_id"]] = []

        if mode == "train":
            idxs = list(range(stride_len, len(traj_img_paths), stride_len))
        else:  # mode == "test"
            # knn requires all frames to be used
            # actions must be known at each step for open loop control
            idxs = list(range(stride_len, len(traj_img_paths)))

        for i in range(len(idxs)):
            img_path_cat = traj_img_paths[max(0, idxs[i] - num_frames_window) : idxs[i]]
            img_paths.append(img_path_cat)
            curr_target_idxs.append((i, len(idxs) - 1))
            actions[path["traj_id"]].append(path["actions"][idxs[i]])
            traj_ids.append((path["traj_id"], i))

        global_idx += len(idxs)

        assert global_idx == len(img_paths), f"{global_idx} != {len(img_paths)}"

    assert len(img_paths) == len(curr_target_idxs) == len(traj_ids)

    return img_paths, curr_target_idxs, traj_ids_idxs, actions, traj_ids


def load_target_data(extract_dir, mode="train"):
    curr_target_idxs_path = os.path.join(extract_dir, "curr_target_idxs.pkl")
    img_paths_path = os.path.join(extract_dir, "image_paths.pkl")
    traj_ids_idxs_path = os.path.join(extract_dir, "traj_ids_idxs.pkl")
    with open(curr_target_idxs_path, "rb") as f:
        curr_target_idxs = pickle.load(f)
    with open(img_paths_path, "rb") as f:
        image_paths = pickle.load(f)
    with open(traj_ids_idxs_path, "rb") as f:
        traj_ids_idxs = pickle.load(f)

    if mode == "test":
        actions_path = os.path.join(extract_dir, "actions.pkl")
        traj_ids_path = os.path.join(extract_dir, "traj_ids.pkl")

        with open(actions_path, "rb") as f:
            actions = pickle.load(f)
        with open(traj_ids_path, "rb") as f:
            traj_ids = pickle.load(f)

        return image_paths, curr_target_idxs, traj_ids_idxs, actions, traj_ids

    else:
        return image_paths, curr_target_idxs, traj_ids_idxs


def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./prep_data/3s_window_train/",
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
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="train or test",
        choices=["train", "test"],
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    teleop_pickle = args.teleop_pickle
    teleop_dir = args.teleop_dir

    with open(teleop_pickle, "rb") as f:
        teleop_paths = pickle.load(f)

    (
        image_paths,
        curr_target_idxs,
        traj_ids_idxs,
        actions,
        traj_ids,
    ) = get_teleop_data_targets(teleop_paths, teleop_dir, args.mode)
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

    actions_fn = os.path.join(output_dir, "actions.pkl")
    print(f"Saving actions to {actions_fn}")
    with open(actions_fn, "wb") as f:
        pickle.dump(actions, f)

    traj_ids_fn = os.path.join(output_dir, "traj_ids.pkl")
    print(f"Saving traj_ids to {traj_ids_fn}")
    with open(traj_ids_fn, "wb") as f:
        pickle.dump(traj_ids, f)


if __name__ == "__main__":
    main()
