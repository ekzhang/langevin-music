import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple

from ..dataset.chorales import Chorale, ChoraleDataset


class ChoraleSeqDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        chorales = ChoraleDataset()
        total = len(chorales)
        n_val = int(0.1 * total)
        self.bach_train, self.bach_val = random_split(chorales, [total - n_val, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bach_train,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bach_val,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(
        batch: List[Chorale],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # (batch_size, seq_length)
        tensors = [item.encode() for item in batch]
        # We pad the initial input with a zero tensor, as an initial starting token
        shifted_input = [
            torch.cat((torch.zeros_like(t[0:1], dtype=torch.long), t[:-1]))
            for t in tensors
        ]
        return shifted_input, tensors


class ChoraleNcsnModule(pl.LightningDataModule):
    """Module for extracting training samples for score matching."""

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        chorales = ChoraleDataset()
        total = len(chorales)
        n_val = int(0.1 * total)
        self.bach_train, self.bach_val = random_split(chorales, [total - n_val, n_val])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bach_train,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.bach_val,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(
        batch: List[Chorale],
    ) -> List[torch.Tensor]:
        # (batch_size, seq_length, 4)
        tensors = [item.encode() for item in batch]
        return tensors
