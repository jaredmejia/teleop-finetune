from functools import partial

import matplotlib.pyplot as plt
import torch
import yaml
from teleop_prop_data import TeleopActionDataset, get_dataloaders, load_target_data

from finetune_models import MultiSensoryAttention
from finetune_utils import backbone_transforms, multi_collate, set_seeds


def vis_attn_grid(attn_weights, model_name):
    labels = [f"video_{i}" for i in range(6)]
    pos = 6
    if model_name != "action-v-msa":
        labels += [f"audio_{i}" for i in range(6)]
        pos += 6
    plt.figure(figsize=(8, 10))
    plt.matshow(attn_weights[0].detach().cpu().numpy(), cmap="viridis")
    plt.xticks(range(pos), labels=labels, rotation=90, fontsize=8)
    plt.yticks(range(pos), labels=labels, fontsize=8)
    plt.colorbar()
    plt.title(f"{model_name}: Attention weights")
    plt.savefig(f"./attn_weights_{model_name}.png")
    plt.close()


def vis_attn_time(model, dataset, traj_start_idx, traj_end_idx, model_name):
    """Visualize attention weights over time for a single trajectory"""
    model.eval()
    attn_weights_list = []
    with torch.no_grad():
        for i in range(traj_start_idx, traj_end_idx):
            sample = dataset[i]
            out, attn_weights = model(sample, return_attn=True)
            attn_weights_list.append(attn_weights)
    attn_weights = torch.cat(attn_weights_list, dim=0)

    # sum attn weights over both modalities
    attn_weights = attn_weights.sum(dim=1)
    import pdb

    pdb.set_trace()
    print(attn_weights.shape)


def main():

    extract_dir = "./prep_data/3s_window_train"
    # get dataset
    image_paths, curr_target_idxs, traj_ids_idxs, actions, traj_ids = load_target_data(
        extract_dir
    )
    include_image = False
    include_audio = include_orig_audio = True

    DEVICE = torch.device("cpu")

    collate_func = partial(
        multi_collate,
        device=DEVICE,
        include_image=include_image,
        include_audio=include_audio,
        include_orig_audio=include_orig_audio,
    )

    BATCH_SIZE = 1
    NUM_VALID_TRAJS = 20
    random_seed = 47

    # get model specific transformsa
    cfg_dict = {"model": {"args": {"modality": "audio-video"}}}
    transforms = backbone_transforms("msa", cfg_dict)
    transforms = partial(transforms, inference=True, device=DEVICE)

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
    # train_loader, val_loader = get_dataloaders(
    #     dataset,
    #     traj_ids_idxs=traj_ids_idxs,
    #     collate_fn=collate_func,
    #     batch_size=BATCH_SIZE,
    #     num_valid_trajs=NUM_VALID_TRAJS,
    # )
    # set_seeds(random_seed)

    # # sample_input = next(iter(val_loader))
    # # print(sample_input['video'].shape, sample_input['audio'].shape)

    start_idx = traj_ids_idxs[0][1]
    end_idx = traj_ids_idxs[1][1]

    for config_path in [
        "./configs/action-msa.yaml",
        "./configs/action-v-msa.yaml",
        "./configs/action-msa-pe_i.yaml",
        "./configs/action-msa-pe_l.yaml",
        "./configs/action-msa-bn-pe_l-ln.yaml",
    ]:
        config = yaml.safe_load(open(config_path, "rb"))
        model_args = config["model"]["args"]
        model = MultiSensoryAttention(**model_args)
        model.load_state_dict(
            torch.load(config["model"]["checkpoint"], map_location=DEVICE)
        )
        # model.eval()
        # with torch.no_grad():
        #     out, attn_weights = model(sample_input, return_attn=True)
        #     import pdb; pdb.set_trace()
        # vis_attn_grid(attn_weights, config['model_name'])
        vis_attn_time(model, dataset, start_idx, end_idx, config["model_name"])


if __name__ == "__main__":
    main()
