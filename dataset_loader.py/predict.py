import torch
from dataset_loader import SpectralDataset
from model import NeuralProcess


dataset = SpectralDataset("dataset/test.csv")

model = NeuralProcess()

model.load_state_dict(torch.load("spectral_model.pth"))

model.eval()

cx, cy, tx, _ = dataset[0]

with torch.no_grad():

    pred = model(cx, cy, tx)

print("Predicted target_y:")

print(pred[:10])