import os
import torch
from model import NeuralProcess
from kaggle_loader import load_kaggle_sample
import pandas as pd

print("Starting Kaggle prediction")

model = NeuralProcess()

model.load_state_dict(torch.load("audio_model.pth"))

model.eval()

test_folder = "dataset/test"

results = []

for file in os.listdir(test_folder):

    path = os.path.join(test_folder,file)

    context_x, context_y, target_x = load_kaggle_sample(path)

    with torch.no_grad():

        pred = model(context_x,context_y,target_x)

    pred = pred.squeeze().numpy()

    df = pd.DataFrame({
        "target_y":pred
    })

    results.append(df)

print("Prediction completed")