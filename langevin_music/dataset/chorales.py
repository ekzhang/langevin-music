"""Chorale data loaders."""

from __future__ import annotations

import os
import pickle
import torch
from dataclasses import dataclass
from music21 import *
from torch.utils.data import Dataset
from typing import List, Literal, Tuple, Union

SOPRANO_RANGE = (pitch.Pitch("C4"), pitch.Pitch("G5"))
ALTO_RANGE = (pitch.Pitch("F3"), pitch.Pitch("D5"))
TENOR_RANGE = (pitch.Pitch("C3"), pitch.Pitch("A4"))
BASS_RANGE = (pitch.Pitch("C2"), pitch.Pitch("E4"))
RANGES = (SOPRANO_RANGE, ALTO_RANGE, TENOR_RANGE, BASS_RANGE)

bcl = corpus.chorales.ChoraleList()


@dataclass
class Chorale:
    """Class for keeping track of a preprocessed Chorale.

    This stores information in a similar fashion to the DeepBach paper, with a
    set of four sequence-encoded parts (in sixteenth notes) and a metadata list.

    Args:
        parts: A tuple of four parts, each of which is a list of tokens.
            These contain the MIDI values of the pitches, or "Hold" / "Rest".
        metadata: A list of metadata, currently unused.
    """

    Token = Union[int, Literal["Hold"], Literal["Rest"]]

    parts: Tuple[List[Token], List[Token], List[Token], List[Token]]
    metadata = None  # TODO

    def encode(self) -> torch.Tensor:
        """Encode object into a Tensor acceptable by PyTorch."""
        retval = []
        for i, part in enumerate(self.parts):
            mapped = []
            base_note = RANGES[i][0].midi
            for token in part:
                if token == "Hold":
                    mapped.append(1)
                elif token == "Rest":
                    mapped.append(2)
                else:
                    mapped.append(3 + token - base_note)
            retval.append(mapped)
        return torch.Tensor(retval).long().T

    @staticmethod
    def decode(tensor: torch.Tensor) -> Chorale:
        """Inverse function of encode(), converts a Tensor into an object."""
        parts = []
        for i, part in enumerate(tensor.T.tolist()):
            mapped = []
            base_note = RANGES[i][0].midi
            for entry in part:
                if entry == 0:
                    print("Warning: 0 in chorale tensor, treating as hold")
                    mapped.append("Hold")
                elif entry == 1:
                    mapped.append("Hold")
                elif entry == 2:
                    mapped.append("Rest")
                else:
                    mapped.append(base_note + entry - 3)
            parts.append(mapped)
        return Chorale(parts=tuple(parts))

    def to_score(self) -> stream.Score:
        """Attempts to convert this Chorale object into a readable score.

        This can be used to display a generated chorale, using music21's built
        in MuseScore integration. For example:

        >>> Chorale.decode(tensor).to_score().show()

        Implementation is basic but has some problems. Right now the key
        signature is inferred, but there are some issues:

        - How are we going to infer the time signature? What about anacrusis?
        - Why does music21 keep adding random, unecessary natural accents?
        """
        def to_part(tokens: List[Chorale.Token]) -> stream.Part:
            part = stream.Part()
            for t, token in enumerate(tokens):
                if token != "Hold":
                    length = 0.25
                    for s in range(t + 1, len(tokens)):
                        if tokens[s] == "Hold":
                            length += 0.25
                        else:
                            break
                    if token == "Rest":
                        current_note = note.Rest()
                    else:
                        current_note = note.Note(token)
                    current_note.quarterLength = length
                    part.append(current_note)
            part.makeAccidentals(
                overrideStatus=True,
                          cautionaryPitchClass=False,
                          inPlace=True)
            return part

        score = stream.Score([to_part(tokens) for tokens in self.parts])
        keysig = score.analyze("key")
        for part in score:
            part.insert(0, keysig)
            part.transpose(0, inPlace=True)
        return score


class ChoraleDataset(Dataset):
    """A dataset of 4-part Bach chorales, with preprocessing."""

    CACHE_FILE = "data/chorales.pkl"

    def __init__(self):
        super().__init__()
        if os.path.exists(self.CACHE_FILE):
            # Load cached data first, since music21 is slow
            print(f"Loading cached dataset from {self.CACHE_FILE}...")
            with open(self.CACHE_FILE, "rb") as f:
                self.scores = pickle.load(f)
        else:
            print("Chorale cache not found, building (may take a few minutes)...")
            self.scores = []
            for k in bcl.byBWV.keys():
                score: stream.Score = corpus.parse(f"bach/bwv{k}")
                print(f"> Processing BWV {k}")
                self.scores.extend(process(score))
            with open(self.CACHE_FILE, "wb") as f:
                pickle.dump(self.scores, f)
            print(f"Done! Chorales saved to {self.CACHE_FILE}.")

    def __getitem__(self, index: int) -> Chorale:
        return self.scores[index]

    def __len__(self) -> int:
        return len(self.scores)


def process(score: stream.Score) -> List[Chorale]:
    """Convert a music21 Bach chorale into training samples suitable for machine learning."""
    if len(score.parts) != 4:
        # Filter out non four-part chorales
        return []

    # First, generate all transpositions within valid vocal ranges
    min_trans, max_trans = -float("inf"), float("inf")
    for i in range(4):
        notes: List[note.Note] = list(score.parts[i].flat.getElementsByClass(note.Note))
        min_pitch = min(note.pitch for note in notes)
        max_pitch = max(note.pitch for note in notes)
        if min_pitch < RANGES[i][0]:
            print(f"Warning: min range {RANGES[i][0]}, saw {min_pitch} in voice {i}")
            min_pitch = RANGES[i][0]
            return []  # Some chorales are unusually out-of-range, so we'll skip
        if max_pitch > RANGES[i][1]:
            print(f"Warning: max range {RANGES[i][1]}, saw {max_pitch} in voice {i}")
            max_pitch = RANGES[i][1]
            return []  # Some chorales are unusually out-of-range, so we'll skip
        min_trans = max((min_trans, RANGES[i][0].midi - min_pitch.midi))
        max_trans = min((max_trans, RANGES[i][1].midi - max_pitch.midi))

    retval = []
    for semitones in range(min_trans, max_trans + 1):
        transposed = score.transpose(semitones)
        retval.append(process_score(transposed))
    return retval


def process_score(score: stream.Score) -> Chorale:
    """Convert a single, normalized score into string sequence form."""
    return Chorale(parts=tuple(process_part(score.parts[i]) for i in range(4)))


def process_part(part: stream.Part) -> List[Chorale.Token]:
    # 4 sixteenths/beat
    length = int((part.highestTime - 0.0) * 4)
    notes_and_rests = list(part.flat.getElementsByClass([note.Note, note.Rest]))

    # Now, parse the chorale and add to encoded
    j = 0
    t = []

    # This part is modified from DeepBach
    for j, n in enumerate(notes_and_rests):
        if (
            j == len(notes_and_rests) - 1
            or notes_and_rests[j + 1].offset > len(t) / 4.0
        ):
            current_note = n.pitch.midi if n.isNote else "Rest"
            t.append(current_note)
            t.extend(["Hold"] * (int(n.quarterLength * 4) - 1))

    assert len(t) == length
    return t


if __name__ == "__main__":
    # Testing code
    data = ChoraleDataset()
    Chorale.decode(data[0].encode()).to_score().show()
    print(data[0])
    print("Number of chorales:", len(data))
