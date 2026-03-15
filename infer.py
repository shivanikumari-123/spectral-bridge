import torch
import numpy as np
import librosa
from scipy.io.wavfile import write
import os
from model import AudioTransformer

print("Starting inference")

# check input file
if not os.path.exists("input_audio.wav"):
    print("ERROR: input_audio.wav not found")
    exit()

# check model file
if not os.path.exists("audio_transformer.pth"):
    print("ERROR: audio_transformer.pth not found")
    exit()

# load model
model = AudioTransformer()
model.load_state_dict(torch.load("audio_transformer.pth", map_location="cpu"))
model.eval()

print("Model loaded")

# load audio
signal, sr = librosa.load("input_audio.wav", sr=16000)
print("Audio loaded length:", len(signal))

# convert to spectrogram
spec = np.abs(librosa.stft(signal))

# limit size
spec = spec[:128, :128]

# create missing mask
mask = np.ones_like(spec)
mask[:, 40:60] = 0

context = spec * mask

context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

print("Running model prediction...")

with torch.no_grad():
    pred = model(context)

pred = pred.squeeze().numpy()

# convert back to audio
audio = librosa.istft(pred)

write("reconstructed.wav", 16000, audio.astype(np.float32))

print("Reconstructed audio saved as reconstructed.wav")