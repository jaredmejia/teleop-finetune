import argparse
import glob
import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from finetune_utils import (
    write_audio_spec,
    write_audio_waveform,
    write_image,
    write_video,
)
from teleop_prop_data import TeleopCompletionDataset, load_target_data

sys.path.insert(1, "/home/vdean/franka_learning_jared/")
from pretraining import load_transforms


def create_video(
    img_dir,
    out_vid_path,
    exp_dir_id=None,
    text=None,
    frame_rate=30,
    width=640,
    height=480,
):
    """Creates a video from a directory of images"""
    imgs = sorted(glob.glob(os.path.join(img_dir, "*color.jpeg")))
    video = cv2.VideoWriter(
        out_vid_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (width, height)
    )

    if text is None:
        text = f"Experiment: {os.path.basename(img_dir)}"

    print(f"Saving video to {out_vid_path}")

    for img in imgs:
        img = cv2.imread(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(
            img, exp_dir_id, (5, 445), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA
        )
        img = cv2.putText(img, text, (5, 465), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        video.write(img)

    video.release()
    cv2.destroyAllWindows()


def get_sec(time_str):
    """Get Seconds from time."""
    hour, minute, second, second_decimal = time_str.split(".")
    return (
        int(hour) * 3600 + int(minute) * 60 + int(second) + float("0." + second_decimal)
    )


def write_txt_waveform(txt_audio_file, out_waveform_path):
    """Writes a waveform plot from a txt audio file"""
    with open(txt_audio_file, "r") as filehandle:
        audio_data = []
        for line in filehandle:
            line = line[:-1]
            line = line.split(" ")
            if len(line) > 1:
                audio_data.append(line)
            else:
                break
    time_data = [get_sec(line[0]) for line in audio_data]
    time_data = np.asarray(time_data) - time_data[0]  # Start time axis at 0s
    audio_data = [line[1:] for line in audio_data]
    audio_data = np.array(audio_data).astype(float)

    assert audio_data.shape[1] == 4

    audio_data = audio_data.T
    audio_data = np.mean(audio_data, axis=0)

    print(f"Saving waveform to {out_waveform_path}")
    write_audio_waveform(
        torch.tensor(audio_data), x_vals=time_data, name=out_waveform_path
    )


def write_query_knn_data(
    query_knn_data_dir,
    knn_log_file,
    query_log_dir,
    dataset,
    transforms,
    include_audio=False,
    include_orig_audio=False,
    exp_num=None,
):
    """Writes the query and knn data to a directory"""

    # reading knn log file
    with open(f"{knn_log_file}", "r") as f:
        lines = f.readlines()

    print(f"Writing knn data from {knn_log_file}")

    # writing knn data
    knn_dists = []
    step_vals = []
    for line_idx in range(0, len(lines), 5):
        assert lines[line_idx].split()[0] == "step:"
        step = lines[line_idx].split()[-1]
        knn_idx = int(lines[line_idx + 1].split()[-1])
        knn_dis = float(lines[line_idx + 2].split()[-1])
        knn_dists.append(knn_dis)
        step_vals.append(step)

        # get corresponding nearest neighbor data
        knn_data = dataset[knn_idx]

        # write nearest neighbor data
        vid_name = os.path.join(query_knn_data_dir, f"{step.zfill(4)}_knn_video.gif")
        write_video(knn_data["video"], name=vid_name)
        img_name = os.path.join(query_knn_data_dir, f"{step.zfill(4)}_knn_img.jpeg")
        write_image(knn_data["image"], name=img_name)
        if include_audio:
            audio_name = os.path.join(
                query_knn_data_dir, f"{step.zfill(4)}_knn_spec.png"
            )
            write_audio_spec(knn_data["audio"], name=audio_name)
        if include_orig_audio:
            audio_name = os.path.join(
                query_knn_data_dir, f"{step.zfill(4)}_knn_waveform.png"
            )
            write_audio_waveform(knn_data["audio"], name=audio_name)

    # plot knn distance per step
    plt.figure()
    plt.plot(step_vals, knn_dists)
    plt.xlabel("step")
    plt.ylabel("knn distance")
    plt.savefig(os.path.join(query_knn_data_dir, "knn_distances.png"))
    plt.clf()
    plt.close()

    # save knn_dists + steps
    dist_step_data = {"knn_dists": knn_dists, "step_vals": step_vals}
    with open(os.path.join(query_knn_data_dir, "knn_dists.pkl"), "wb") as f:
        pickle.dump(dist_step_data, f)

    if os.path.exists(query_log_dir):
        # write matching query data
        print(f"Writing query data from {query_log_dir}")
        query_logs = sorted(glob.glob(os.path.join(query_log_dir, "*.pkl")))
        for query_log in query_logs:
            step = os.path.basename(query_log).split(".")[0].split("_")[-1]

            # load and transform query data
            with open(query_log, "rb") as f:
                query_data = pickle.load(f)
            query_data = transforms(query_data)

            # write query data
            vid_name = os.path.join(
                query_knn_data_dir, f"{step.zfill(4)}_query_video.gif"
            )
            write_video(query_data["video"], name=vid_name)
            img_name = os.path.join(
                query_knn_data_dir, f"{step.zfill(4)}_query_img.jpeg"
            )
            write_image(query_data["image"], name=img_name)
            if include_audio:
                spec_name = os.path.join(
                    query_knn_data_dir, f"{step.zfill(4)}_query_spec.png"
                )
                write_audio_spec(query_data["audio"], name=spec_name)
            if include_orig_audio:
                waveform_name = os.path.join(
                    query_knn_data_dir, f"{step.zfill(4)}_query_waveform.png"
                )
                write_audio_waveform(query_data["orig_audio"], name=waveform_name)

    exp_dist_data = {"EXP_NUM": exp_num, "step_vals": step_vals, "knn_dists": knn_dists}
    return exp_dist_data


def plot_dist_data(exp_dist_data_list, success_idxs, data_vis_dir, model_name):
    """Plots knn distance data for each experiment."""
    success_data = []
    success_steps = []
    failure_data = []
    failure_steps = []
    plt.figure()

    for exp_dist_data in exp_dist_data_list:
        idx = exp_dist_data["EXP_NUM"]
        if idx in success_idxs:
            success_data.extend(exp_dist_data["knn_dists"])
            success_steps.extend(exp_dist_data["step_vals"])
            plt.plot(
                np.array(exp_dist_data["step_vals"]).astype(float),
                exp_dist_data["knn_dists"],
                "-o",
                alpha=0.2,
                color="tab:blue",
            )
            plt.plot(
                np.array(exp_dist_data["step_vals"]).astype(float)[-1:],
                exp_dist_data["knn_dists"][-1:],
                "x",
                color="tab:blue",
            )
        else:
            failure_data.extend(exp_dist_data["knn_dists"])
            failure_steps.extend(exp_dist_data["step_vals"])
            plt.plot(
                np.array(exp_dist_data["step_vals"]).astype(float),
                exp_dist_data["knn_dists"],
                "-o",
                alpha=0.2,
                color="tab:orange",
            )
            plt.plot(
                np.array(exp_dist_data["step_vals"]).astype(float)[-1:],
                exp_dist_data["knn_dists"][-1:],
                "x",
                color="tab:orange",
            )

    success_steps = np.array(success_steps).astype(float)
    success_data = np.array(success_data).astype(float)
    failure_steps = np.array(failure_steps).astype(float)
    failure_data = np.array(failure_data).astype(float)

    # fitting curve to dist data
    success_poly_fit = np.poly1d(np.polyfit(success_steps, success_data, 3))
    failure_poly_fit = np.poly1d(np.polyfit(failure_steps, failure_data, 3))

    # plot fit of knn distance per step
    success_polyline = np.linspace(0, np.amax(success_steps), 100)
    failure_polyline = np.linspace(0, np.amax(failure_steps), 100)
    plt.plot(
        success_polyline,
        success_poly_fit(success_polyline),
        label="success",
        linewidth=3,
        color="tab:blue",
    )
    plt.plot(
        failure_polyline,
        failure_poly_fit(failure_polyline),
        label="failure",
        linewidth=3,
        color="tab:orange",
    )
    plt.xlabel("step")
    plt.ylabel("knn distance")
    plt.ylim(15, 35)
    plt.legend()
    plt.title(f"{model_name} knn distance per step")
    plt.savefig(os.path.join(data_vis_dir, f"{model_name}_knn_distances.png"))
    plt.clf()
    plt.close()


def main():
    parser = argparse.ArgumentParser("Visualize KNN results from eval trajectories")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/home/vdean/franka_learning_jared/outputs/av-avid-p-shf-c-b-a-2/exp_H-60_k-1",
    )
    parser.add_argument(
        "--target_data_dir", type=str, default="./prep_data/3s_window_test"
    )
    parser.add_argument(
        "--model_config_file", type=str, default="./configs/av-avid-p-shf-c-b-a-2.yaml"
    )
    args = parser.parse_args()

    # load model configs
    model_config = yaml.safe_load(open(args.model_config_file, "r"))
    model_args = model_config["model"]["args"]
    model_name = model_config["model_name"]
    include_audio = include_orig_audio = (
        True if model_args["modality"] == "audio-video" else False
    )

    # get files from experiment logs
    txt_audio_files = sorted(glob.glob(os.path.join(args.exp_dir, "*.txt")))
    img_dirs = [txt_audio_file[:-4] for txt_audio_file in txt_audio_files]
    knn_log_files = sorted(
        glob.glob(os.path.join(args.exp_dir, "knn_logs", "knn_log*.txt"))
    )
    exp_dir_id = os.path.join(
        os.path.basename(os.path.dirname(args.exp_dir)), os.path.basename(args.exp_dir)
    )
    if exp_dir_id[:-1] == "/":
        exp_dir_id = exp_dir_id[:-1]

    assert len(txt_audio_files) == len(img_dirs) == len(knn_log_files)

    # load transforms
    transforms = load_transforms(model_name, model_config)

    # load orig knn dataset
    image_paths, curr_target_idxs, traj_ids_idxs = load_target_data(
        args.target_data_dir
    )
    dataset = TeleopCompletionDataset(
        image_paths,
        curr_target_idxs,
        transforms,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    # output data vis dir
    data_vis_dir = os.path.join(args.exp_dir, "data_vis")
    if not os.path.exists(data_vis_dir):
        os.makedirs(data_vis_dir)
    print(f"Saving data visualizations to {data_vis_dir}")

    # save data for each traj iteratively
    exp_dist_data_list = []
    for EXP_NUM, (txt_audio_file, img_dir, knn_log_file) in enumerate(
        zip(txt_audio_files, img_dirs, knn_log_files)
    ):
        EXP_NUM = EXP_NUM + 1
        print(f"\nProcessing experiment {EXP_NUM}...")

        # experiment vis dir
        exp_vis_dir = os.path.join(data_vis_dir, os.path.basename(img_dir))
        if not os.path.exists(exp_vis_dir):
            os.makedirs(exp_vis_dir)

        # creating video of full eval traj
        out_vid_path = os.path.join(
            exp_vis_dir, f"eval_vid_{str(EXP_NUM).zfill(2)}.mp4"
        )
        # create_video(img_dir=img_dir, out_vid_path=out_vid_path, exp_dir_id=exp_dir_id)

        # creating waveform of audio
        out_waveform_path = os.path.join(exp_vis_dir, f"waveform_{EXP_NUM}.png")
        # write_txt_waveform(txt_audio_file, out_waveform_path=out_waveform_path)

        # query log dir
        query_log_dir = os.path.join(
            args.exp_dir, "query_logs", f"query_data_{EXP_NUM}"
        )

        # matching query knn data dir
        query_knn_data_dir = os.path.join(exp_vis_dir, f"query_knn_data_{EXP_NUM}")
        if not os.path.exists(query_knn_data_dir):
            os.makedirs(query_knn_data_dir)

        # write matching query knn data
        exp_dist_data = write_query_knn_data(
            query_knn_data_dir=query_knn_data_dir,
            knn_log_file=knn_log_file,
            query_log_dir=query_log_dir,
            dataset=dataset,
            transforms=transforms,
            include_audio=include_audio,
            include_orig_audio=include_orig_audio,
            exp_num=EXP_NUM,
        )
        exp_dist_data_list.append(exp_dist_data)

        print(f"Finished processing experiment {EXP_NUM}.\n")

    print(f"Plotting dist data...")
    if model_name == "av-avid-p-shf-c-b-a-2":
        success_idxs = [4, 5, 6, 11, 12, 14, 18]
    else:
        success_idxs = [6, 7, 8, 10, 11, 12, 13, 15, 17, 19]
    plot_dist_data(exp_dist_data_list, success_idxs, data_vis_dir, model_name)

    print(f"Finished.")


if __name__ == "__main__":
    main()
