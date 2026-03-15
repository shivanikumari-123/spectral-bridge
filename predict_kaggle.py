import os
import torch
import pandas as pd
from model import NeuralProcess
from kaggle_loader import load_kaggle_sample

print("Starting Kaggle prediction")

model = NeuralProcess()
model.load_state_dict(torch.load("audio_model.pth"))

model.eval()

test_folder = "dataset/test"

all_predictions = []

for file in os.listdir(test_folder):

    path = os.path.join(test_folder, file)

    context_x, context_y, target_x = load_kaggle_sample(path)

    with torch.no_grad():
        pred = model(context_x, context_y, target_x)

    pred = pred.squeeze().numpy()

    df = pd.DataFrame(pred, columns=["target_y"])

    all_predictions.append(df)

final = pd.concat(all_predictions)

final.to_csv("submission.csv", index=False)

print("Submission file created")