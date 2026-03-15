import pandas as pd
import torch

def load_kaggle_sample(file):

    data = pd.read_csv(file)

    context_x = torch.tensor(data["context_x"].values).float().unsqueeze(1)
    context_y = torch.tensor(data["context_y"].values).float().unsqueeze(1)
    target_x = torch.tensor(data["target_x"].values).float().unsqueeze(1)

    return context_x, context_y, target_x