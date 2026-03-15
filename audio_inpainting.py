import torch
import torch.nn as nn


class AudioInpaintingModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Context Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        # Decoder / prediction network
        self.decoder = nn.Sequential(
            nn.Linear(129,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )


    def forward(self,x_context,y_context,x_target):

        context = torch.cat([x_context,y_context],dim=1).unsqueeze(0)

        encoded_context = self.encoder(context)

        query = torch.cat(
            [x_target, torch.zeros_like(x_target)],
            dim=1
        ).unsqueeze(0)

        query = self.encoder(query)

        attn_output,_ = self.attention(
            query,
            encoded_context,
            encoded_context
        )

        combined = torch.cat(
            [x_target.unsqueeze(0),attn_output],
            dim=2
        )

        pred = self.decoder(combined)

        return pred.squeeze(0)