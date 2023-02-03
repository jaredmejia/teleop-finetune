from sklearn.neighbors import KDTree
import argparse
import joblib
import os
import pickle
import sys
import torch
import tqdm
import yaml

from functools import partial

from finetune_utils import multi_collate
from teleop_prop_data import TeleopCompletionDataset, load_target_data

sys.path.insert(1, "/home/vdean/franka_learning_jared/")
from pretraining import load_encoder, load_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_VALID_TRAJS = 20


def get_embeddings(dl, model):
    """computes embeddings from dataloader and pretrained model"""
    emb_list = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dl)):
            emb = model(data)
            emb_list.append(emb.detach().cpu())
    embeddings = torch.cat(emb_list, dim=0)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_file", type=str)
    parser.add_argument(
        "--extract_dir", type=str, default="./prep_data/3s_window_test/"
    )
    parser.add_argument("--output_dir", type=str, default="./prep_data/3s_window_test/")
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    # load model configs
    model_config = yaml.safe_load(open(args.model_config_file, "r"))
    model_args = model_config["model"]["args"]
    model_name = model_config["model_name"]

    # load model
    model = load_encoder(model_name, model_config).to(DEVICE)
    transforms = load_transforms(model_name, model_config)

    # load data
    include_audio = False if model_args["modality"] == "video" else True
    collate_func = partial(
        multi_collate,
        device=DEVICE,
        include_audio=include_audio,
        include_orig_audio=False,
    )
    image_paths, curr_target_idxs, traj_ids_idxs = load_target_data(args.extract_dir)

    # filtering out validation trajectories for knn
    split_idx = traj_ids_idxs[-NUM_VALID_TRAJS][1]
    image_paths = image_paths[:split_idx]
    curr_target_idxs = curr_target_idxs[:split_idx]

    dataset = TeleopCompletionDataset(
        image_paths,
        curr_target_idxs,
        transforms,
        include_audio=include_audio,
        include_orig_audio=False,
    )
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_func,
    )

    # get embeddings
    embeddings = get_embeddings(dl, model)

    # standardize embeddings and save mean and std
    emb_mean = embeddings.mean(dim=0)
    emb_std = embeddings.std(dim=0)
    standardized_embeddings = (embeddings - emb_mean) / emb_std
    emb_stats = {"mean": emb_mean, "std": emb_std}
    emb_stats_fn = os.path.join(args.output_dir, f"emb_stats_{model_name}.pkl")
    print(f"Saving embedding stats to {emb_stats_fn}")
    with open(emb_stats_fn, "wb") as f:
        pickle.dump(emb_stats, f)

    # fit knn to standardized embeddings
    knn_model = KDTree(standardized_embeddings)

    # save knn model
    knn_fn = os.path.join(args.output_dir, f"knn_model_{model_name}.joblib")
    print(f"knn data shape: {knn_model.data.shape}")
    print(f"Saving knn model to {knn_fn}")
    joblib.dump(knn_model, knn_fn)


if __name__ == "__main__":
    main()
