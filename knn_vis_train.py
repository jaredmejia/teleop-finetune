import argparse
import os
import pickle
import sys

import joblib
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

import wandb
from finetune_utils import (
    multi_collate,
    write_audio_spec,
    write_audio_waveform,
    write_image,
    write_video,
)
from teleop_prop_data import TeleopCompletionDataset, load_target_data

sys.path.insert(1, "/home/vdean/franka_learning_jared/")
from pretraining import load_encoder, load_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_VALID_TRAJS = 20


def load_expert(extract_dir, model_name):
    knn_model_path = os.path.join(extract_dir, f"knn_model_{model_name}.joblib")
    emb_stats_path = os.path.join(extract_dir, f"emb_stats_{model_name}.pkl")
    knn_model = joblib.load(knn_model_path)
    with open(emb_stats_path, "rb") as f:
        emb_stats = pickle.load(f)
    return knn_model, emb_stats


def get_query_indices(traj_ids_idxs, max_idx):
    query_indices = []
    for i in range(-NUM_VALID_TRAJS, 0):
        # get random index in traj
        start_idx = traj_ids_idxs[i][1]
        if i == -1:
            end_idx = max_idx
        else:
            end_idx = traj_ids_idxs[i + 1][1]

        query_indices.append(np.random.randint(start_idx, end_idx))

    return query_indices


def get_nearest_neighbor(
    knn_model,
    emb_stats,
    encoder,
    dataset,
    q_idxs,
    k=1,
    device=None,
    include_audio=False,
    include_orig_audio=False,
):
    """returns nearest neighbor index and distance"""
    q_batch = [dataset[q_idx] for q_idx in q_idxs]
    collated_batch = multi_collate(
        q_batch,
        device=device,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    encoder.eval()
    with torch.no_grad():
        emb = encoder(collated_batch).cpu()
    emb = (emb - emb_stats["mean"]) / emb_stats["std"]
    dists, knn_idxs = knn_model.query(emb.numpy(), k=k)

    return dists, knn_idxs


def log_matching_data(
    dataset, dists, knn_idxs, q_idxs, include_audio=False, include_orig_audio=False
):
    """Logs the matching knn data to wandb"""
    for sample_idx, (q_idx, knn_idx, dist) in enumerate(zip(q_idxs, knn_idxs, dists)):
        log_dict = {"step": sample_idx}
        q_data = dataset[q_idx]
        knn_idx = knn_idx.item()
        knn_data = dataset[knn_idx]
        dist_str = f"{dist.item():.3f}"

        # log query data
        log_dict["query/video"] = write_video(
            q_data["video"], caption=f"Query Video, Sample {q_idx}", wandb_log=True
        )
        log_dict["query/image"] = write_image(
            q_data["image"], caption=f"Query Image, Sample {q_idx}", wandb_log=True
        )
        if include_audio:
            log_dict["query/audio_spec"] = write_audio_spec(
                q_data["audio"],
                caption=f"Query Audio Spec, Sample {q_idx}",
                wandb_log=True,
            )
        if include_orig_audio:
            log_dict["query/audio_waveform"] = write_audio_waveform(
                q_data["orig_audio"],
                caption=f"Query Audio Waveform, Sample {q_idx}",
                wandb_log=True,
            )

        # log knn data
        log_dict["knn/video"] = write_video(
            knn_data["video"],
            caption=f"KNN Video, Sample {knn_idx}, Dist: {dist_str}",
            name="./temp_imgs/sample_vid_knn.gif",
            wandb_log=True,
        )
        log_dict["knn/image"] = write_image(
            knn_data["image"],
            caption=f"KNN Image, Sample {knn_idx}, Dist: {dist_str}",
            wandb_log=True,
        )
        if include_audio:
            log_dict["knn/audio_spec"] = write_audio_spec(
                knn_data["audio"],
                caption=f"KNN Audio Spec, Sample {knn_idx}, Dist: {dist_str}",
                name="./temp_imgs/spec_2.png",
                wandb_log=True,
            )
        if include_orig_audio:
            log_dict["knn/audio_waveform"] = write_audio_waveform(
                knn_data["orig_audio"],
                caption=f"KNN Audio Waveform, Sample {knn_idx}, Dist: {dist_str}",
                name="./temp_imgs/waveform_2.png",
                wandb_log=True,
            )

        wandb.log(log_dict)


def main():
    parser = argparse.ArgumentParser(description="Visualize KNN results within dataset")
    parser.add_argument(
        "--extract_dir", type=str, default="./prep_data/3s_window_test/"
    )
    parser.add_argument(
        "--model_config_file", type=str, default="./configs/v-avid-p-c-b-a-2.yaml"
    )
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    # load model configs
    model_config = yaml.safe_load(open(args.model_config_file, "r"))
    model_args = model_config["model"]["args"]
    model_name = model_config["model_name"]
    include_audio = include_orig_audio = (
        True if model_args["modality"] == "audio-video" else False
    )

    # load model and transforms
    encoder = load_encoder(model_name, model_config).to(DEVICE)
    transforms = load_transforms(model_name, model_config)

    # load expert
    knn_model, emb_stats = load_expert(args.extract_dir, model_name)

    # load all data
    image_paths, curr_target_idxs, traj_ids_idxs = load_target_data(args.extract_dir)

    # train/val split
    split_idx = traj_ids_idxs[-NUM_VALID_TRAJS][1]

    # load dataset
    dataset = TeleopCompletionDataset(
        image_paths,
        curr_target_idxs,
        transforms,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    # get query indices
    np.random.seed(args.seed)
    query_indices = get_query_indices(traj_ids_idxs, len(image_paths))

    # get query nearest neighbors
    dists, knn_idxs = get_nearest_neighbor(
        knn_model,
        emb_stats,
        encoder,
        dataset,
        query_indices,
        k=1,
        device=DEVICE,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    # log matches to wandb
    wandb.init(
        project="finetune-teleop",
        entity="contact-mic",
        name=f"{model_name}-knn-visualize",
    )
    log_matching_data(
        dataset,
        dists,
        knn_idxs,
        query_indices,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )


if __name__ == "__main__":
    main()
