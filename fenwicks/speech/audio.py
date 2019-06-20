import librosa
import numpy as np

__all__ = ['read_logmelspectrogram', 'gen_synth_silence']


def read_logmelspectrogram(filepath: str, sr: int = 16000, len_sec: int = 1, n_components: int = 40, n_fft: int = 400,
                           hop_length: int = 160) -> np.ndarray:
    """
    Build a logarithmic-Mel-spectrogram from a given wave file. The length of the speech is unified through clipping and
    zero-padding.

    :param filepath: Path to the input wave file.
    :param sr: Sampling rate. Default: 16000 samples per second.
    :param len_sec: Length of the speech, in seconds. Default: 1 second.
    :param n_components: Number of Fourier components. Default: 40.
    :param n_fft: Length of the sliding window on which we apply Fourier transformation.
    :param hop_length: Stride for the sliding window on which we apply Fourier transformation.
    :return: a 2D array for the log-Mel-spectrogram.
    """
    samples, _ = librosa.load(filepath, sr=sr)
    sample_len = sr * len_sec
    samples = samples[:sample_len]
    samples = np.pad(samples, (0, max(0, sample_len - len(samples))), 'constant')
    melspectrogram = librosa.feature.melspectrogram(y=samples, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                    n_mels=n_components)
    return librosa.power_to_db(melspectrogram)


def gen_synth_silence(sr: int = 16000, n_rand: int = 4600) -> np.ndarray:
    d = np.zeros(sr)
    loc = np.random.randint(0, sr, n_rand)
    d[loc[::2]] = -1
    d[loc[1::2]] = 1
    return d
