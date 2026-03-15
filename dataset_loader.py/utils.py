import torch
import librosa
import numpy as np

def load_audio(path):

    signal, sr = librosa.load(path, sr=16000)

    signal = signal[:16000]

    return signal


def audio_to_spec(signal):

    spec = librosa.stft(signal, n_fft=512)

    spec = np.abs(spec)

    spec = np.log1p(spec)

    return spec


def spec_to_audio(spec):

    spec = np.expm1(spec)

    audio = librosa.istft(spec)

    return audio