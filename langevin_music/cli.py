"""Defines the CLI interface and training loop."""

import click
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from .network.recurrent import LSTMPredictor
from .dataset.modules import ChoraleSeqDataModule
from .dataset.chorales import Chorale

@click.group()
def main():
    """Entry point for training neural networks and running inference."""
    pass


@main.command()
@click.option(
    "-b", "--batch_size", default=8, show_default=True, help="Batch size for training"
)
@click.option(
    "-s",
    "--seq_length",
    default=32,
    show_default=True,
    help="Max length of sequence to use in truncated BPTT",
)
@click.option(
    "-c",
    "--checkpoint",
    type=str,
    metavar="FILE",
    help="A checkpoint file to initialize training from",
)
@click.option(
    "-e",
    "--epochs",
    default=10,
    show_default=True,
    help="Number of epochs to train for",
)
def train(batch_size, seq_length, checkpoint, epochs):
    """Train a deep generative model for chorale composition."""
    if checkpoint:
        model = LSTMPredictor.load_from_checkpoint(checkpoint)
    else:
        model = LSTMPredictor()
    data = ChoraleSeqDataModule(batch_size)

    trainer = pl.Trainer(
        max_epochs=epochs,
        truncated_bptt_steps=seq_length,
        logger=pl_loggers.TensorBoardLogger("logs/"),
        log_every_n_steps=1,
    )
    trainer.split_idx = 0
    trainer.fit(model, datamodule=data)
    #torch.save(model, "entire_model.pt")


@main.command()
@click.option(
    "-c",
    "--checkpoint",
    type=str,
    metavar="FILE",
    help="A checkpoint file to sample from",
)
def sample(checkpoint):
    """Sample a chorale from a trained model, loaded from a checkpoint."""
    model = LSTMPredictor.load_from_checkpoint(checkpoint)
    chorale = Chorale.decode(model.sample())
    chorale.to_score().show()
    
