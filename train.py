import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import NeuralProcess

print("Starting training pipeline...")

# =========================
# Dataset Loader (.npz files)
# =========================
class AudioDataset(Dataset):

    def __init__(self, folder):

        self.files = []

        for f in os.listdir(folder):
            if f.endswith(".npz"):
                self.files.append(os.path.join(folder, f))

        print("Dataset files found:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx])

        context_x = torch.tensor(data["context_x"]).float().unsqueeze(1)
        context_y = torch.tensor(data["context_y"]).float().unsqueeze(1)

        target_x = torch.tensor(data["target_x"]).float().unsqueeze(1)
        target_y = torch.tensor(data["target_y"]).float().unsqueeze(1)

        return context_x, context_y, target_x, target_y


# =========================
# Load Dataset
# =========================
dataset_path = "dataset"

dataset = AudioDataset(dataset_path)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

print("Dataset size:", len(dataset))


# =========================
# Model
# =========================
model = NeuralProcess()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.MSELoss()


# =========================
# Training Loop
# =========================
epochs = 5

for epoch in range(epochs):

    total_loss = 0

    for context_x, context_y, target_x, target_y in loader:

        context_x = context_x.squeeze(0)
        context_y = context_y.squeeze(0)
        target_x = target_x.squeeze(0)
        target_y = target_y.squeeze(0)

        pred = model(context_x, context_y, target_x)

        loss = loss_fn(pred, target_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch + 1, "Loss:", total_loss)


# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "audio_model.pth")

print("Training finished")
print("Model saved as audio_model.pth")