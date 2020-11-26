"""Transformer for the conditional distribution of notes."""

import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn

from .common import PositionalEncoding
from ..dataset.chorales import RANGES


class MusicTransformer(pl.LightningModule):
    def __init__(self, embedding_dim=128, dropout=0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.dims = [high.midi - low.midi + 4 for (low, high) in RANGES]

        # Step 1: Embed the musical scores into a latent space of dimension `self.ninp`
        self.embed = [
            nn.Embedding(num_embeddings=dim, embedding_dim=32) for dim in self.dims
        ]
        self.ninp = 128

        # Step 2: Transformer model on the latent space
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.ninp, nhead=8, dropout=dropout),
            num_layers=5,
        )

        # Step 3: Final linear output decoder
        self.decoder = nn.Linear(self.ninp, sum(self.dims))

    def forward(self, x):
        # shape: (seq_length, batch_size, 4)
        x = torch.cat(
            [self.embed[i](x[..., i]) for i in range(4)],
            dim=-1,
        )
        # shape: (seq_length, batch_size, embedding)
        x *= np.sqrt(self.ninp)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x,
            nn.Transformer.generate_square_subsequent_mask(
                self.transformer_encoder, len(x)
            ),
        )
        x = self.decoder(x)
        x = torch.split(x, self.dims, dim=-1)
        x = [F.log_softmax(part, dim=-1) for part in x]
        # shape: list of 4 Tensors, each (seq_length, batch_size, dim)
        return x

    def shared_step(self, batch):
        x = rnn.pad_sequence(batch[0])
        y = rnn.pad_sequence(batch[1], batch_first=True)
        y_hat = self.forward(x)
        loss = 0
        for i in range(4):
            loss += F.nll_loss(y_hat[i].permute(1, 2, 0), y[..., i])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    @torch.no_grad()
    def sample(self, max_len=128) -> torch.Tensor:
        raise NotImplementedError("TODO(@ekzhang): no sampling yet")
