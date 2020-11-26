from .ncsn import NcsnV2Transformer
from .recurrent import LSTMPredictor

ARCH = {
    "ncsnv2-transformer": NcsnV2Transformer,
    "lstm": LSTMPredictor,
}


def get_arch(arch):
    if arch in ARCH:
        return ARCH[arch]
    raise ValueError(f"Unknown architecture {arch}, expected one of {ARCH.keys()}")
