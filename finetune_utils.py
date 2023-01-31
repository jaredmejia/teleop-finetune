import numpy as np
import torch
import sys
import yaml

from torchvision import transforms as T

from finetune_preprocessing import SpecEncoder

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import load_transforms

# define __all__ for all functions and classes in this file
__all__ = [
    "set_seeds",
    "model_prep",
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backbone_transforms(avid_name, avid_cfg_path, audio_transform="AudioPrep"):
    avid_cfg = yaml.safe_load(open(avid_cfg_path))
    avid_transforms = load_transforms(avid_name, avid_cfg)
    r3m_transforms = load_transforms("r3m", None)

    if audio_transform == "SpecEncoder":
        spec_encoder = SpecEncoder()
        print("Using SpecEncoder for audio preprocessing")

    def backbone_transforms_f(data):
        if audio_transform == "AudioPrep":
            data_t = avid_transforms(data)  # video / audio
        elif audio_transform == "SpecEncoder":
            data_t = {}

            # video
            video_input = {"video": data["video"]}
            data_t["video"] = avid_transforms(data)["video"]

            # audio
            audio_input = data["audio"]
            data_t["audio"] = spec_encoder(audio_input)

        data_t["image"] = r3m_transforms(data)["image"]

        for data_type in data.keys():
            if data_type not in data_t.keys():
                data_t[data_type] = torch.tensor(data[data_type])

        return data_t

    return backbone_transforms_f


def multi_collate(batch, device=None, include_audio=True, include_orig_audio=False):
    video_batch, img_batch, audio_batch, target_batch, orig_audio_batch = (
        [],
        [],
        [],
        [],
        [],
    )
    for data in batch:
        video_batch.append(data["video"])
        img_batch.append(data["image"])
        target_batch.append(data["target"])

        if include_audio:
            audio_batch.append(data["audio"])

        if include_orig_audio:
            orig_audio_batch.append(data["orig_audio"])

    batched_data = {}
    video_batch = torch.stack(video_batch, dim=0).to(device)
    batched_data["video"] = video_batch

    img_batch = torch.stack(img_batch, dim=0).to(device)
    batched_data["image"] = img_batch

    target_batch = torch.stack(target_batch, dim=0).to(device)
    batched_data["target"] = target_batch

    if include_audio:
        audio_batch = torch.stack(audio_batch, dim=0).to(device)
        batched_data["audio"] = audio_batch

    if include_orig_audio:
        orig_audio_batch = torch.stack(orig_audio_batch, dim=0)
        batched_data["orig_audio"] = orig_audio_batch

    return batched_data
