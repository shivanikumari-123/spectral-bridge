import torch
from torch.utils.data import DataLoader
from dataset_loader import SpectralDataset
from model import NeuralProcess


dataset = SpectralDataset("dataset/train.csv")

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = NeuralProcess()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn = torch.nn.MSELoss()

epochs = 20


for epoch in range(epochs):

    total_loss = 0

    for cx, cy, tx, ty in loader:

        cx = cx[0]
        cy = cy[0]
        tx = tx[0]
        ty = ty[0]

        optimizer.zero_grad()

        pred = model(cx, cy, tx)

        loss = loss_fn(pred, ty)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch+1, "Loss:", total_loss)


torch.save(model.state_dict(), "spectral_model.pth")

print("Training finished")