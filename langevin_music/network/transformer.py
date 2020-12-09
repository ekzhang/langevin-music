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
    def __init__(self, embedding_dim=128, dropout=0.1):
        super().__init__()
        self.model_type = "Transformer"
        self.dims = [high.midi - low.midi + 4 for (low, high) in RANGES]
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

        # Step 1: Embed the musical scores into a latent space of dimension `self.ninp`
        self.embed = nn.ModuleList(
            [nn.Embedding(num_embeddings=dim, embedding_dim=128) for dim in self.dims]
        )
        self.ninp = 512

        # Step 2: Transformer model on the latent space
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.ninp, nhead=8, dropout=dropout),
            num_layers=6,
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
            ).type_as(x),
        )
        x = self.decoder(x)
        x = torch.split(x, self.dims, dim=-1)
        x = [F.log_softmax(part, dim=-1) for part in x]
        # shape: list of 4 Tensors, each (seq_length, batch_size, dim)
        return x

    def shared_step(self, batch, acc):
        x = rnn.pad_sequence(batch[0])
        y = rnn.pad_sequence(batch[1], batch_first=True)
        y_hat = self.forward(x)
        loss = 0
        for i in range(4):
            loss += F.nll_loss(y_hat[i].permute(1, 2, 0), y[..., i])
            acc(y_hat[i].permute(1, 2, 0), y[..., i])
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.train_acc)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, self.valid_acc)
        self.log("val_loss", loss)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    @torch.no_grad()
    def sample(self, max_len=240) -> torch.Tensor:
        input = torch.zeros((1, 1, 4), dtype=torch.long)
        for i in range(max_len):
            output = self.forward(input)
            # output shape: [(seq_len, batch_size, self.dims[i]) for i in range(4)]
            choices = []
            for o in output:
                probs = np.exp(o[-1, 0].numpy())
                choice = np.random.choice(np.arange(len(probs)), p=probs)
                choices.append(choice)
            sample = torch.Tensor(choices).long()
            if all(x == 0 for x in sample):
                break
            input = torch.cat(
                [input, sample.unsqueeze(0).unsqueeze(0)]
            )  # set next input
        return input[1:].squeeze(1)
