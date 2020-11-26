"""Noise-conditional score networks for music generation."""

import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils import rnn
import numpy as np

from .common import PositionalEncoding
from ..dataset.chorales import RANGES


class NcsnV2Transformer(pl.LightningModule):
    """Reference: Song and Ermon (NeurIPS 2020).

    We use a transformer encoder for learning features from symbolic musical scores, rather
    than their semantic segmentation-inspired architecture.
    """

    def __init__(
        self, embedding_dim=128, dropout=0.1, noise_scales=np.geomspace(20.0, 0.05, 20)
    ):
        super().__init__()
        self.model_type = "NCSNv2 Transformer"
        self.noise_scales = noise_scales
        self.dims = [high.midi - low.midi + 4 for (low, high) in RANGES]
        self.ninp = embedding_dim
        self.embed = nn.Linear(sum(self.dims), self.ninp)
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        encoder_layer = nn.TransformerEncoderLayer(self.ninp, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=5)
        self.decoder = nn.Linear(in_features=self.ninp, out_features=sum(self.dims))

    def forward(self, x):
        """Forward pass of the NCSNv2 model.

        Expected input shape: (seq_length, batch_size, sum(self.dims))
        Output shape: (seq_length, batch_size, sum(self.dims))
        """
        x = self.embed(x)  # shape: (seq_length, batch_size, embedding)
        x *= math.sqrt(self.ninp)  # transformer scaling magic, multiply by sqrt(dim)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)  # shape: (seq_length, batch_size, sum(self.dims))
        return x

    def shared_step(self, batch):
        # x: Tensor[torch.long] of shape (seq_len, batch_size, 4)
        x = rnn.pad_sequence(batch)
        x = torch.cat(
            [F.one_hot(x[..., i], num_classes=dim) for i, dim in enumerate(self.dims)],
            dim=-1,
        ).float()
        # New shape: (seq_len, batch_size, sum(self.dims))
        assert len(x.shape) == 3 and x.shape[-1] == sum(self.dims)

        # TODO(ekzhang): Right now, we have a single sigma for the entire batch, but
        # it probably makes more sense to select different sigmas for each sample in
        # the batch instead.
        sigma = np.random.choice(self.noise_scales)
        noise = torch.randn_like(x) * sigma
        x_tilde = x + noise

        # Denoising NCSN loss
        s = self.forward(x_tilde)
        loss = 0.5 * F.mse_loss(s, -noise / sigma)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def sample(self, max_len=128, epsilon=2e-5, t=100, denoise=True) -> torch.Tensor:
        """Sample a chorale using annealed Langevin dynamics at multiple noise scales.

        See Algorithm 1 in the paper for details. The parameters `epsilon` and `t` are
        defined there. The sampling procedure initializes x_0 uniformly at random in
        the unit hypercube, then proceeds with L noise scales. At each noise scale, it
        does T steps of Langevin dynamics.

        If `denoise` is set to True, then one additional step of score-based gradient
        ascent is performed as a final cleaning operation (without Langevin noise).

        Returns a Tensor with dtype=torch.long of shape (seq_len, 4).
        """
        with torch.no_grad():
            x0 = torch.rand(max_len, 1, sum(self.dims))
            sigma_l = self.noise_scales[-1]
            for sigma in self.noise_scales:
                print(f"langevin dynamics: sigma = {sigma}")
                alpha = epsilon * sigma ** 2 / sigma_l ** 2
                for _ in range(t):
                    z = torch.randn_like(x0)
                    x0 += alpha * self.forward(x0) / sigma + np.sqrt(2.0 * alpha) * z
            if denoise:
                x0 += sigma_l * self.forward(x0)
        parts = torch.split(x0.squeeze(1), self.dims, dim=-1)
        return torch.stack([torch.argmax(part, dim=1) for part in parts], dim=1)
