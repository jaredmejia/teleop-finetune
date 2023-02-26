import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from einops import rearrange
from torchvision import transforms as T
from torchvision.transforms import functional as TF

CONTACT_AUDIO_FREQ = 32000
CONTACT_AUDIO_MAX = 2400
CONTACT_AUDIO_AVG = 2061
CONTACT_AUDIO_STD = 28
WINDOW_DUR = 1  # 3


class SpecEncoder:
    ### ADAPTED FROM SEE, HEAR, FEEL
    def __init__(
        self,
        num_stack=1,
        orig_sr=32000,
        out_sr=16000,
        norm_audio=False,
        norm_freq=False,
    ):
        self.num_stack = num_stack
        self.norm_audio = norm_audio
        self.norm_freq = norm_freq
        self.out_sr = out_sr
        self.n_mels = 64
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.out_sr,
            n_fft=int(self.out_sr * 0.025) + 1,
            hop_length=int(self.out_sr * 0.01),
            n_mels=self.n_mels,
        )
        self.audio_resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=self.out_sr, dtype=torch.float64
        )

    def __call__(self, audio_arr):
        """
        Args:
            audio_arr (np.ndarray): (n_channels, n_samples)
        Returns:
            log_spec (torch.Tensor): (1, 64, 100)
        """
        waveform = self.standardize_audio(audio_arr)
        eps = 1e-8
        spec = self.mel(waveform.float())
        log_spec = torch.log(spec + eps)

        assert log_spec.size(-2) == 64

        if self.norm_audio:
            log_spec /= log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100] for 1 sec

        elif self.norm_freq:
            log_spec[:] = log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100] for 1 sec

        log_spec = log_spec.unsqueeze(0)
        if self.num_stack > 1:
            log_spec = rearrange(log_spec, "c m (n l) -> n c m l", n=self.num_stack)

        return log_spec

    def standardize_audio(self, audio_arr):
        audio_arr = np.mean(audio_arr, axis=0)
        waveform = torch.tensor(audio_arr)
        waveform = (waveform - CONTACT_AUDIO_AVG) / (CONTACT_AUDIO_MAX)
        waveform = self.audio_resampler(waveform)

        # if smaller than 3 second window, pad with averge
        if waveform.shape[0] < WINDOW_DUR * self.out_sr:
            waveform = torch.cat(
                (
                    torch.zeros(
                        (WINDOW_DUR * self.out_sr - waveform.shape[0],),
                        dtype=waveform.dtype,
                    ),
                    waveform,
                )
            )
        else:
            waveform = waveform[-WINDOW_DUR * self.out_sr :]

        return waveform


class SequentialVisualTransform(object):
    def __init__(self, seq_len=6):
        self.seq_len = seq_len
        self.resize = T.Resize(size=(240, 320))
        self.to_tensor = T.ToTensor()

    def __call__(self, data, train=True):
        """Data is a list of 6 images"""
        # left padding with first image
        if len(data) < self.seq_len:
            data = [data[0]] * (self.seq_len - len(data)) + data

        # resize
        data = [self.resize(img) for img in data]

        if train:
            # same random crop for all images
            i, j, h, w = T.RandomCrop.get_params(data[0], (192, 256))
            crop = T.Lambda(lambda img: TF.crop(img, i, j, h, w))
            data = [crop(img) for img in data]
        else:
            # center crop
            data = [TF.center_crop(img, (192, 256)) for img in data]

        # to tensor
        data = [self.to_tensor(img) for img in data]

        return torch.stack(data, dim=0)


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def main():
    spec_encoder = SpecEncoder(num_stack=6, norm_audio=True, norm_freq=True)

    import glob

    txt_files = sorted(
        glob.glob(
            "/home/vdean/franka_demo/logs/jared_chopping_exps_v4/pos_1/22-12-08-19-27-14/*.txt"
        )
    )[-180:-90]

    txt_list = []
    for txt_path in txt_files:
        txt_arr = np.loadtxt(txt_path)
        txt_arr = txt_arr.T
        txt_list.append(txt_arr)
    audio_data = np.concatenate(txt_list, axis=1)
    print(f"audio_data shape : {audio_data.shape}")

    spec = spec_encoder(audio_data)
    print(spec.shape)


if __name__ == "__main__":
    main()
