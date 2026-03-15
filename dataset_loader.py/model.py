import torch
import torch.nn as nn

class NeuralProcess(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        self.decoder = nn.Sequential(
            nn.Linear(129,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, context_x, context_y, target_x):

        context = torch.cat([context_x,context_y],dim=1)

        encoded = self.encoder(context)

        global_rep = encoded.mean(dim=0,keepdim=True)

        global_rep = global_rep.repeat(len(target_x),1)

        decoder_input = torch.cat([target_x,global_rep],dim=1)

        pred = self.decoder(decoder_input)

        return pred