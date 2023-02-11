import sys

import torch
import yaml
from einops import rearrange
from torch import nn
from torchvision import models

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import load_encoder


class AvidR3M(nn.Module):
    def __init__(
        self,
        avid_name,
        avid_cfg_path,
        frozen_backbone=True,
        avid_emb_dim=1024,
        r3m_emb_dim=512,
        hidden_dim=512,
        disable_backbone_dropout=True,
        modality="audio-video",
        batchnorm=False,
        output_dim=1,
    ):
        super().__init__()

        self.avid_backbone, self.r3m_backbone = self.model_prep(
            avid_name, avid_cfg_path
        )

        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(avid_emb_dim + r3m_emb_dim)

        self.feat_fusion = nn.Sequential(
            nn.Linear(avid_emb_dim + r3m_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.modality = modality
        self.disable_backbone_dropout = disable_backbone_dropout

        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def model_prep(self, avid_name, avid_cfg_path):
        avid_cfg = yaml.safe_load(open(avid_cfg_path))
        avid_backbone = load_encoder(avid_name, avid_cfg)
        r3m_backbone = load_encoder("r3m", None)
        return avid_backbone, r3m_backbone

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
        data["image"] = data["image"] * 255.0

        if self.frozen_backbone:
            with torch.no_grad():
                vid_aud_emb = self.avid_backbone(data)
                img_emb = self.r3m_backbone(data["image"])
        else:
            vid_aud_emb = self.avid_backbone(data)
            img_emb = self.r3m_backbone(data["image"])

        cat_emb = torch.cat((vid_aud_emb, img_emb), dim=1)

        if self.batchnorm:
            cat_emb = self.batchnorm(cat_emb)

        pred = self.feat_fusion(cat_emb)

        return pred


class AvidR3MAttention(nn.Module):
    def __init__(
        self,
        avid_name,
        avid_cfg_path,
        avid_emb_dim=1024,
        r3m_emb_dim=512,
        mod_emb_size=128,
        num_heads=8,
        seq_length=6,
        hidden_dim=512,
        modality="audio-video",
        output_dim=1,
    ):
        super().__init__()

        self.modality = modality

        self.avid_backbone, self.r3m_backbone = self.model_prep(
            avid_name, avid_cfg_path
        )
        self.avid_batchnorm = nn.BatchNorm1d(avid_emb_dim)

        self.downsample_video = nn.Linear(512, mod_emb_size)
        if self.modality == "audio-video":
            self.downsample_audio = nn.Linear(512, mod_emb_size)

        self.mha = nn.MultiHeadAttention(
            avid_emb_dim * seq_length, num_heads, batch_first=True
        )
        self.feat_fusion = nn.Sequential(
            nn.BatchNorm1d(avid_emb_dim * seq_length + r3m_emb_dim),
            nn.Linear(avid_emb_dim * seq_length + r3m_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # TODO: fill out forward pass
    def forward(self, data):
        pass


class SequentialEncoder(nn.Module):
    def __init__(
        self, modality, model_name="resnet18", seq_length=6, downsample_size=128
    ):
        super().__init__()
        self.modality = modality
        self.seq_length = seq_length
        self.model_name = model_name
        self.downsample_size = downsample_size
        self.encoder = load_encoder(
            model_name, {"pretrained": False, "modality": modality}
        )
        self.downsampler = nn.Linear(512, downsample_size)

    def forward(self, data):
        """
        Args:
            data (torch.Tensor): tensor of shape (batch_size, c, h, seq_length * w)
        Returns:
            (torch.Tensor): tensor of shape (batch_size, seq_length, downsample_size)
        """
        data = rearrange(data, "b s c h w -> (b s) c h w")
        emb = self.encoder(data)
        emb = self.downsampler(emb)
        emb = rearrange(emb, "(b s) d -> b s d", s=self.seq_length)
        return emb


class MultiSensoryAttention(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        embed_dim=128,
        seq_length=6,
        output_dim=7,
        audio_encoder=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.visual_encoder = SequentialEncoder(
            "video",
            model_name=backbone_name,
            seq_length=seq_length,
            downsample_size=embed_dim,
        )

        # TODO: add option for pretrained audio encoder (e.g. from AVID)
        self.audio_encoder = SequentialEncoder(
            "audio",
            model_name=backbone_name,
            seq_length=seq_length,
            downsample_size=embed_dim,
        )

        self.mha = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        self.mlp_inp_dim = 2 * embed_dim * seq_length
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_inp_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
        )

    def set_train(self):
        self.train()

    def forward(self, data, return_attn=False):
        """
        Args:
            data (dict): dictionary containing video and audio data
        Returns:
            (torch.Tensor): tensor of shape (batch_size, output_dim)
        """
        video = data["video"]  # (batch_size, .5 * seq_length, 3, h, w)
        audio = data["audio"]  # (batch_size, 1, h, .5 * seq_length * w)
        audio = rearrange(
            audio, "b c h (s w) -> b s c h w", s=self.seq_length
        )  # (batch_size, .5 * seq_length, 3, h, w)

        video_emb = self.visual_encoder(
            video
        )  # (batch_size, .5 * seq_length, embed_dim)
        audio_emb = self.audio_encoder(
            audio
        )  # (batch_size, .5 * seq_length, embed_dim)

        mha_inp = torch.cat(
            (video_emb, audio_emb), dim=1
        )  # (batch_size, seq_length, embed_dim)
        mha_out, attn_weights = self.mha(
            mha_inp, mha_inp, mha_inp
        )  # (batch_size, seq_length, embed_dim)

        # residual connection
        mlp_inp = mha_inp + mha_out  # (batch_size, seq_length, embed_dim)
        mlp_inp = rearrange(
            mlp_inp, "b s d -> b (s d)"
        )  # (batch_eize, seq_length * embed_dim)
        out = self.mlp(mlp_inp)  # (batch_size, output_dim)

        if return_attn:
            return out, attn_weights
        else:
            return out


def main():
    data = {
        "video": torch.randn(2, 6, 3, 224, 224),
        "audio": torch.randn(2, 6, 1, 224, 224),
    }

    model = MultiSensoryAttention()
    out = model(data)
    print(out.shape)


if __name__ == "__main__":
    main()
