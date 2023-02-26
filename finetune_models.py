import sys

import torch
import yaml
from einops import rearrange
from torch import nn

sys.path.insert(1, "/home/vdean/franka_learning_jared")
from pretraining import load_encoder

R3M_EMB_DIM = 512
AVID_EMB_DIM_PER_MODALITY = 512


class AvidR3M(nn.Module):
    def __init__(
        self,
        avid_name,
        avid_cfg_path,
        frozen_backbone=True,
        hidden_dim=512,
        disable_backbone_dropout=True,
        mlp_dropout=False,
        modality="audio-video",
        batchnorm=False,
        output_dim=1,
        goal_conditioned=True,
    ):
        super().__init__()

        self.modality = modality
        if self.modality == "audio-video":
            avid_emb_dim = AVID_EMB_DIM_PER_MODALITY * 2
        else:
            avid_emb_dim = AVID_EMB_DIM_PER_MODALITY

        self.goal_conditioned = goal_conditioned
        if self.goal_conditioned:
            self.avid_backbone, self.r3m_backbone = self.model_prep(
                avid_name, avid_cfg_path
            )
            self.backbone_out_dim = avid_emb_dim + R3M_EMB_DIM
        else:
            self.avid_backbone = self.model_prep(avid_name, avid_cfg_path)
            self.backbone_out_dim = avid_emb_dim

        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(self.backbone_out_dim)

        self.feat_fusion = nn.Sequential(
            nn.Linear(self.backbone_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5) if mlp_dropout else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.disable_backbone_dropout = disable_backbone_dropout

        self.frozen_backbone = frozen_backbone
        if self.frozen_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

    def model_prep(self, avid_name, avid_cfg_path):
        """Load the pretrained AVID and R3M backbones.

        Args:
            avid_name (str): Name of the AVID model.
            avid_cfg_path (str): Path to the AVID config file.

        Returns:
            avid_backbone (nn.Module): AVID backbone.
            r3m_backbone (nn.Module): R3M backbone.
        """
        avid_cfg = yaml.safe_load(open(avid_cfg_path, "rb"))
        avid_backbone = load_encoder(avid_name, avid_cfg)

        if self.goal_conditioned:
            r3m_backbone = load_encoder("r3m", None)
            return avid_backbone, r3m_backbone
        else:
            return avid_backbone

    def freeze_backbone(self):
        """Freeze the backbone parameters and disable dropout and batchnorm
        for the backbone.
        """
        # freeze parameters of the backbone
        for param in self.avid_backbone.parameters():
            param.requires_grad = False

        # disable dropout and batchnorm for backbone
        self.avid_backbone.eval()

        if self.goal_conditioned:
            for param in self.r3m_backbone.parameters():
                param.requires_grad = False
            self.r3m_backbone.eval()

        self.frozen_backbone = True

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters and enable dropout and batchnorm
        for the backbone.
        """
        # unfreeze parameters of the backbone
        for param in self.avid_backbone.parameters():
            param.requires_grad = True

        # enable dropout and batchnorm for backbone
        self.avid_backbone.train()

        if self.goal_conditioned:
            for param in self.r3m_backbone.parameters():
                param.requires_grad = True
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
        """Forward pass through the model.

        Args:
            data (dict): Dictionary containing the input data with keys
                "video" and "image" if modality is "video", and additionally
                "audio" if modality is "audio-video".

        Returns:
            pred (torch.Tensor): Model prediction.
        """
        if self.frozen_backbone:
            with torch.no_grad():
                vid_aud_emb = self.avid_backbone(data)

                if self.goal_conditioned:
                    img_emb = self.r3m_backbone(data["image"])
        else:
            vid_aud_emb = self.avid_backbone(data)

            if self.goal_conditioned:
                img_emb = self.r3m_backbone(data["image"])

        if self.goal_conditioned:
            mlp_inp = torch.cat((vid_aud_emb, img_emb), dim=1)
        else:
            mlp_inp = vid_aud_emb

        if self.batchnorm:
            mlp_inp = self.batchnorm(mlp_inp)

        pred = self.feat_fusion(mlp_inp)

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
        seq_len=6,
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

        self.mha = nn.MultiheadAttention(
            avid_emb_dim * seq_len, num_heads, batch_first=True
        )
        self.feat_fusion = nn.Sequential(
            nn.BatchNorm1d(avid_emb_dim * seq_len + r3m_emb_dim),
            nn.Linear(avid_emb_dim * seq_len + r3m_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # TODO: fill out forward pass
    def forward(self, data):
        pass


class SequentialEncoder(nn.Module):
    def __init__(self, modality, model_name="resnet18", seq_len=6, downsample_size=128):
        """Encoder that takes in a sequence of images if modality is "video" or
        a sequence of audio frames if modality is "audio-video" and returns a
        sequence of embeddings.

        Args:
            modality (str): Modality of the input data. Can be "video" or
                "audio".
            model_name (str): Name of the encoder model. Defaults to "resnet18".
            seq_len (int): Length of the input sequence. Defaults to 6.
            downsample_size (int): Size of the downsampled embedding. Defaults
                to 128.
        """
        super().__init__()
        self.modality = modality
        self.seq_len = seq_len
        self.model_name = model_name
        self.downsample_size = downsample_size
        self.encoder = load_encoder(
            model_name, {"pretrained": False, "modality": modality}
        )
        self.downsampler = nn.Linear(512, downsample_size)

    def forward(self, data):
        """Encodes and downsamples the input data.

        Args:
            data (torch.Tensor): tensor of shape (batch_size, c, h, seq_len * w)
        Returns:
            (torch.Tensor): tensor of shape (batch_size, seq_len, downsample_size)
        """
        data = rearrange(data, "b s c h w -> (b s) c h w")
        emb = self.encoder(data)
        emb = self.downsampler(emb)
        emb = rearrange(emb, "(b s) d -> b s d", s=self.seq_len)
        return emb


class PositionalEncoding(nn.Module):
    def __init__(self, pos_enc_type, embed_dim=128, dropout=0.1, max_len=12):
        """Positional encoding for transformer models.

        Args:
            pos_enc_type (str): Type of positional encoding. Can be "index" or
                "learned".
                embed_dim (int): Embedding dimension.
                dropout (float): Dropout probability.
                max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pos_enc_type = pos_enc_type

        if pos_enc_type == "index":
            # set positional encoding as 1/n
            pos_emb = torch.arange(0, max_len) / max_len
            pos_emb = pos_emb.unsqueeze(1)
            pos_emb = (pos_emb * 2) - 1  # normalizing to [-1, 1]
            self.register_buffer("pos_emb", pos_emb, persistent=False)

        elif pos_enc_type == "learned":
            # learned positional embedding
            self.pos_encoder = nn.Linear(max_len, embed_dim)

            # create one-hot encoding for each position
            positions = torch.zeros((max_len, max_len), dtype=torch.float)
            positions = positions.fill_diagonal_(1)
            self.register_buffer("positions", positions, persistent=False)

        else:
            raise NotImplementedError(
                "Positional encoding type must be one of 'index' or 'learned'"
            )

    def forward(self, mha_inp):
        """Performs positional encoding on the input tensor.

        Args:
            mha_inp (torch.Tensor): tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            (torch.Tensor): tensor of shape (batch_size, seq_len, embed_dim)
        """
        if self.pos_enc_type == "index":
            return self.dropout(mha_inp + self.pos_emb)

        elif self.pos_enc_type == "learned":
            # apply learned positional encoding
            pos_emb = self.pos_encoder(self.positions)

            return self.dropout(mha_inp + pos_emb)

        else:
            raise NotImplementedError(
                "Positional encoding type must be one of 'index' or 'learned'"
            )


class MultiSensoryAttention(nn.Module):
    def __init__(
        self,
        backbone_name="resnet18",
        embed_dim=128,
        seq_len=6,
        output_dim=7,
        modality="audio-video",
        batchnorm=False,
        positional_encoding="index",
        layernorm=False,
        goal_conditioned=False,
        audio_encoder=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.modality = modality
        self.batchnorm = batchnorm
        self.layer_norm = layernorm

        self.visual_encoder = SequentialEncoder(
            "video",
            model_name=backbone_name,
            seq_len=seq_len,
            downsample_size=embed_dim,
        )

        # TOKENS FOR AUDIO
        if self.modality == "audio-video":
            # TODO: add option for pretrained audio encoder (e.g. from AVID)
            self.audio_encoder = SequentialEncoder(
                "audio",
                model_name=backbone_name,
                seq_len=seq_len,
                downsample_size=embed_dim,
            )
            self.max_len = seq_len * 2
        else:
            self.max_len = seq_len

        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(embed_dim)

        # POSITIONAL ENCODING
        if positional_encoding in ["index", "learned"]:
            self.pos_enc = PositionalEncoding(
                positional_encoding,
                embed_dim=embed_dim,
                dropout=0.1,
                max_len=self.max_len,
            )
        else:
            self.pos_enc = None

        self.mha = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.mha_out_dim = embed_dim * self.max_len

        if layernorm:
            self.layer_norm = nn.LayerNorm(embed_dim)

        # GOAL CONDITIONING
        self.goal_conditioned = goal_conditioned
        if goal_conditioned:
            self.r3m_backbone = load_encoder("r3m", None)
            self.mlp_inp_dim = self.mha_out_dim + R3M_EMB_DIM
        else:
            self.mlp_inp_dim = self.mha_out_dim

        # MLP
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
        if self.modality == "audio-video":
            return self.audio_video_forward(data, return_attn)
        else:
            return self.video_forward(data, return_attn)

    def video_forward(self, data, return_attn=False):
        """Performs forward pass for video-only modality.

        Args:
            data (dict): dictionary containing video data
        Returns:
            (torch.Tensor): tensor of shape (batch_size, output_dim)
        """
        assert self.modality == "video", "Modality must be video"

        video = data["video"]  # (batch_size, seq_len, 3, h, w)
        video_emb = self.visual_encoder(video)  # (batch_size, seq_len, embed_dim)

        if self.batchnorm:
            video_emb = rearrange(video_emb, "b s d -> b d s")
            video_emb = self.batchnorm(video_emb)
            video_emb = rearrange(video_emb, "b d s -> b s d")

        if self.pos_enc is not None:
            video_emb = self.pos_enc(video_emb)

        mha_out, attn_weights = self.mha(
            video_emb, video_emb, video_emb
        )  # (batch_size, seq_len, embed_dim)

        # residual connection
        mha_out = mha_out + video_emb

        if self.layer_norm:
            mha_out = self.layer_norm(mha_out)

        mha_out = rearrange(mha_out, "b s d -> b (s d)")

        # goal conditioning
        if self.goal_conditioned:
            img_emb = self.r3m_backbone(data["image"])
            mlp_inp = torch.cat([mha_out, img_emb], dim=1)
        else:
            mlp_inp = mha_out

        out = self.mlp(mlp_inp)  # (batch_size, output_dim)

        if return_attn:
            return out, attn_weights
        else:
            return out

    def audio_video_forward(self, data, return_attn=False):
        """Performs forward pass for audio-video modality.

        Args:
            data (dict): dictionary containing video and audio data
        Returns:
            (torch.Tensor): tensor of shape (batch_size, output_dim)
        """
        assert self.modality == "audio-video", "Modality must be audio-video"

        video = data["video"]  # (batch_size, seq_len, 3, h, w)
        audio = data["audio"]  # (batch_size, 1, h, seq_len * w)
        audio = rearrange(audio, "b c h (s w) -> b s c h w", s=self.seq_len)

        video_emb = self.visual_encoder(video)  # (batch_size, seq_len, embed_dim)
        audio_emb = self.audio_encoder(audio)  # (batch_size, seq_len, embed_dim)

        mha_inp = torch.cat(
            (video_emb, audio_emb), dim=1
        )  # (batch_size, 2 * seq_len, embed_dim)

        if self.batchnorm:
            mha_inp = rearrange(mha_inp, "b s d -> b d s")
            mha_inp = self.batchnorm(mha_inp)
            mha_inp = rearrange(mha_inp, "b d s -> b s d")

        if self.pos_enc is not None:
            mha_inp = self.pos_enc(mha_inp)

        mha_out, attn_weights = self.mha(mha_inp, mha_inp, mha_inp)

        # residual connection
        mha_out = mha_inp + mha_out

        if self.layer_norm:
            mha_out = self.layer_norm(mha_out)

        mha_out = rearrange(mha_out, "b s d -> b (s d)")

        # goal conditioning
        if self.goal_conditioned:
            img_emb = self.r3m_backbone(data["image"])
            mlp_inp = torch.cat([mha_out, img_emb], dim=1)
        else:
            mlp_inp = mha_out

        out = self.mlp(mlp_inp)  # (batch_size, output_dim)

        if return_attn:
            return out, attn_weights
        else:
            return out


def main():
    device = torch.device("cpu")

    data = {
        "video": torch.randn(2, 6, 3, 224, 224).to(device),
        "audio": torch.randn(2, 1, 60, 6 * 50).to(device),
    }

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
            torch.load(config["model"]["checkpoint"], map_location=device)
        )
        model.eval()
        with torch.no_grad():
            out, attn_weights = model(data, return_attn=True)

        print(out.shape, attn_weights.shape)


if __name__ == "__main__":
    main()
