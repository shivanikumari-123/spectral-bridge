import torch
from model import NeuralProcess

print("Starting prediction...")

model = NeuralProcess()

# DO NOT load old weights
# model.load_state_dict(torch.load("audio_model.pth"))

model.eval()

context_x = torch.linspace(0,1,50).unsqueeze(1)
context_y = torch.sin(context_x*6)

target_x = torch.linspace(0,1,100).unsqueeze(1)

with torch.no_grad():
    pred = model(context_x, context_y, target_x)

print("Prediction finished")
print(pred[:10])
