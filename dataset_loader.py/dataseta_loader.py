import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SpectralDataset(Dataset):

    def __init__(self, file):

        data = pd.read_csv(file)

        self.context_x = data.filter(regex="context_x").values
        self.context_y = data.filter(regex="context_y").values
        self.target_x = data.filter(regex="target_x").values
        self.target_y = data.filter(regex="target_y").values


    def __len__(self):

        return len(self.context_x)


    def __getitem__(self, idx):

        cx = torch.tensor(self.context_x[idx], dtype=torch.float32).unsqueeze(-1)
        cy = torch.tensor(self.context_y[idx], dtype=torch.float32).unsqueeze(-1)
        tx = torch.tensor(self.target_x[idx], dtype=torch.float32).unsqueeze(-1)
        ty = torch.tensor(self.target_y[idx], dtype=torch.float32).unsqueeze(-1)

        return cx, cy, tx, ty