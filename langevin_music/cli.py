"""Defines the CLI interface and training loop."""

import click
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from .dataset.modules import ChoraleNcsnModule, ChoraleSeqDataModule
from .dataset.chorales import Chorale
from .network.ncsn import NcsnV2Transformer
from .network.recurrent import LSTMPredictor
from .network.transformer import MusicTransformer

ARCHITECTURES = {
    "ncsnv2-transformer": (NcsnV2Transformer, ChoraleNcsnModule),
    "lstm": (LSTMPredictor, ChoraleSeqDataModule),
    "transformer": (MusicTransformer, ChoraleSeqDataModule),
}


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
    "--bptt",
    default=0,
    show_default=True,
    help="Max length of sequence to use in truncated BPTT, if nonzero",
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
@click.option(
    "-a",
    "--arch",
    type=str,
    default="lstm",
    help="Model architecture to load",
)
@click.option("--gpus", default=0, help="Number of GPUs to use")
def train(batch_size, bptt, checkpoint, epochs, arch, gpus):
    """Train a deep generative model for chorale composition."""
    architecture, data_module = ARCHITECTURES[arch]

    if checkpoint:
        model = architecture.load_from_checkpoint(checkpoint)
    else:
        model = architecture()

    # Generic module
    data = data_module(batch_size)
    trainer = pl.Trainer(
        max_epochs=epochs,
        truncated_bptt_steps=(bptt if bptt > 0 else None),
        logger=pl_loggers.TensorBoardLogger("logs/"),
        log_every_n_steps=1,
        gpus=gpus,
    )
    trainer.split_idx = 0
    trainer.fit(model, datamodule=data)


@main.command()
@click.option(
    "-c",
    "--checkpoint",
    type=str,
    metavar="FILE",
    help="A checkpoint file to sample from",
)
@click.option(
    "-a",
    "--arch",
    type=str,
    default="lstm",
    help="Model architecture to load",
)
def sample(checkpoint, arch):
    """Sample a chorale from a trained model, loaded from a checkpoint."""
    architecture, _ = ARCHITECTURES[arch]
    model = architecture.load_from_checkpoint(checkpoint)
    sampled = model.sample()
    print(sampled)
    print(len(sampled))
    chorale = Chorale.decode(sampled)
    chorale.to_score().show()
