import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

from einops import rearrange

CONTACT_AUDIO_FREQ = 32000
CONTACT_AUDIO_MAX = 2400
CONTACT_AUDIO_AVG = 2061
CONTACT_AUDIO_STD = 28
WINDOW_DUR = 3


class SpecEncoder:
    ### ADAPTED FROM SEE, HEAR, FEEL
    def __init__(self, orig_sr=32000, out_sr=16000, norm_audio=False, norm_freq=False):
        self.norm_audio = norm_audio
        self.norm_freq = norm_freq
        self.sr = out_sr
        self.n_mels = 64
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=int(self.sr * 0.025) + 1,
            hop_length=int(self.sr * 0.01),
            n_mels=self.n_mels,
        )
        self.audio_resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=self.sr, dtype=torch.float64
        )

    def __call__(self, audio_arr):
        waveform = self.standardize_audio(audio_arr)
        EPS = 1e-8
        spec = self.mel(waveform.float())
        log_spec = torch.log(spec + EPS)

        assert log_spec.size(-2) == 64

        if self.norm_audio:
            log_spec /= log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100] for 1 sec

        elif self.norm_freq:
            log_spec[:] = log_spec.sum(dim=-2, keepdim=True)  # [1, 64, 100] for 1 sec

        return log_spec.unsqueeze(0)

    def standardize_audio(self, audio_arr):
        audio_arr = np.mean(audio_arr, axis=0)
        waveform = torch.tensor(audio_arr)
        waveform = (waveform - CONTACT_AUDIO_AVG) / (CONTACT_AUDIO_MAX)
        waveform = self.audio_resampler(waveform)

        # if smaller than 3 second window, pad with averge
        if waveform.shape[0] < WINDOW_DUR * self.sr:
            waveform = torch.cat(
                (
                    torch.zeros(
                        (WINDOW_DUR * self.sr - waveform.shape[0],),
                        dtype=waveform.dtype,
                    ),
                    waveform,
                )
            )
        else:
            waveform = waveform[-WINDOW_DUR * self.sr :]

        return waveform


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
    waveform = waveform

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

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
    # spec_encoder = Spec_Encoder(num_stack=5, norm_audio=True, norm_freq=False)
    spec_encoder = SpecEncoder(num_stack=1, norm_audio=True, norm_freq=True)

    # waveform, sr = torchaudio.load("test.wav")
    waveform = torch.rand(1, CONTACT_AUDIO_FREQ)
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

    waveform = np.mean(audio_data, axis=0, keepdims=True)
    waveform = (waveform - CONTACT_AUDIO_AVG) / CONTACT_AUDIO_MAX

    plot_specgram(waveform, 32000, title="Spectrogram", xlim=None)
    plt.savefig("./shf_specgram.png")
    plt.clf()

    plt.plot(list(range(waveform.shape[1])), waveform[0])
    plt.savefig("./shf_audio.png")
    plt.clf()

    # print(f'waveform shape : {waveform.shape}')
    # waveform = torch.from_numpy(waveform)

    # spec = spec_encoder.forward(waveform.unsqueeze(0))
    spec = spec_encoder(audio_data)

    print(spec.shape)
    plot_spectrogram(
        spec.detach().numpy(),
        title="Spectrogram (db)",
        ylabel="freq_bin",
        aspect="auto",
        xmax=None,
    )
    plt.savefig("./shf_spec_2.png")
    plt.clf()
    # import matplotlib.pyplot as plt
    # import librosa
    import librosa.display

    # import numpy as np
    plt.figure()
    s_db = librosa.amplitude_to_db(np.abs(spec.numpy()), ref=np.max)
    # s_db = librosa.amplitude_to_db(np.abs(spec.numpy()[0, 0, :, :]))

    librosa.display.specshow(s_db, sr=16000, x_axis="time", y_axis="linear")
    plt.colorbar()
    plt.savefig("./shf_spec.png")
    plt.clf()


if __name__ == "__main__":
    main()
