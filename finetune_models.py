import torch
import sys
import yaml

from torch import nn

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
            nn.Linear(hidden_dim, 1),
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
