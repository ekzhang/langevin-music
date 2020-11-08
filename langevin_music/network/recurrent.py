import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch.nn.utils import rnn
import numpy as np

from ..dataset.chorales import RANGES


class LSTMPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.dims = [high.midi - low.midi + 4 for (low, high) in RANGES]
        self.embed = [
            nn.Embedding(num_embeddings=dim, embedding_dim=32) for dim in self.dims
        ]
        self.num_layers = 3
        self.hidden_size = 256
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.dense = nn.Linear(
            in_features=self.hidden_size, out_features=sum(self.dims)
        )

    def forward(self, x, h):
        # (seq_length, batch_size, embedding)
        x = torch.cat(
            [self.embed[i](x[..., i]) for i in range(4)],
            dim=-1,
        )
        x, h = self.lstm(x, h)
        x = self.dense(x)
        x = torch.split(x, self.dims, dim=-1)
        x = [F.log_softmax(part, dim=-1) for part in x]
        return x, h

    def init_hidden(self, batch_size, **kw):
        return (
            torch.zeros((self.num_layers, batch_size, self.hidden_size), **kw),
            torch.zeros((self.num_layers, batch_size, self.hidden_size), **kw),
        )

    def shared_step(self, batch, hiddens):
        # print("batch[0] len:", len(batch[0]))
        # print("batch[0] element shape:", batch[0][0].shape)
        # print("batch[1] len:", len(batch[1]))
        # print("batch[1] element shape:", batch[1][0].shape)
        x = rnn.pad_sequence(batch[0])
        y = rnn.pad_sequence(batch[1], batch_first=True)
        y_hat, hiddens = self.forward(x, hiddens)
        loss = 0
        for i in range(4):
            loss += F.nll_loss(y_hat[i].permute(1, 2, 0), y[..., i])
        return loss

    def training_step(self, batch, batch_idx, hiddens):
        if hiddens is None:
            hiddens = self.init_hidden(len(batch[0]))
        loss = self.shared_step(batch, hiddens)
        self.log("train_loss", loss)
        hiddens[0].detach_()
        hiddens[1].detach_()
        return {"loss": loss, "hiddens": hiddens}

    def validation_step(self, batch, batch_idx):
        hiddens = self.init_hidden(len(batch[0]))
        loss = self.shared_step(batch, hiddens)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def sample(self, max_len=400) -> torch.Tensor:
        """Sample a chorale from the LSTM predictor.

        This method samples the chorale token-by-token, stopping early in the
        event that a (0, 0, 0, 0) entry is generated.

        Returns a Tensor with dtype=torch.long of shape (seq_len, 4).
        """
        with torch.no_grad():
            input = torch.randn(5, 4, 128)
            hidden1 = torch.randn(self.num_layers, 4, self.hidden_size)
            hidden2 = torch.randn(self.num_layers, 4, self.hidden_size)
            output_chorale = []
            for i in range(max_len):
                output, (hidden1, hidden2) = self.lstm(input, (hidden1, hidden2))
                topv, topi = output.topk(1)
                topi = topi[0]
                if np.all(topi.numpy() == 0):
                    break
                else:
                    output_chorale.append(topi)
            ts = torch.stack(output_chorale).squeeze()
            return ts
