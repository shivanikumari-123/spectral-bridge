import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
import os
from model import AudioTransformer

print("Starting inference")

# check input audio
if not os.path.exists("input_audio.wav"):
    print("input_audio.wav not found")
    exit()

# load model
model = AudioTransformer()

if not os.path.exists("audio_transformer.pth"):
    print("Model file audio_transformer.pth not found")
    exit()

model.load_state_dict(torch.load("audio_transformer.pth", map_location="cpu"))
model.eval()

# load audio
signal, sr = librosa.load("input_audio.wav", sr=16000)

print("Audio loaded length:", len(signal))

# convert to spectrogram
spec = np.abs(librosa.stft(signal))

spec = spec[:128, :128]

# create mask
mask = np.ones_like(spec)
mask[:,40:60] = 0

context = spec * mask

context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

# predict
with torch.no_grad():
    pred = model(context)

pred = pred.squeeze().numpy()

# convert to audio
audio = librosa.istft(pred)

write("reconstructed.wav",16000,audio.astype(np.float32))

print("Reconstructed audio saved successfully")