"""Chorale data loaders."""

from music21 import *
import numpy as np
from itertools import islice

# it = corpus.chorales.Iterator()
it = islice(corpus.chorales.Iterator(), 10)
encoded = []
no = 0
for i in it:
    no += 1
    if no % 10 == 0:
        print(no)

    # First, transpose to C
    k = i.analyze("key")
    ival = interval.Interval(k.tonic, pitch.Pitch("C"))
    if ival.semitones > 5:
        continue
    sNew = i.transpose(ival)

    list_notes_and_rests = list(
        sNew.parts[0].flat.getElementsByOffset(
            offsetStart=0.0,
            offsetEnd=sNew.flat.highestTime,
            classList=[note.Note, note.Rest],
        )
    )
    length = int((sNew.flat.highestTime - 0.0) * 4)
    # 4 sixteenths/beat

    # Now, parse the chorale and add to encoded
    j = 0
    k = 0
    t = np.empty(length, dtype="S10")
    is_articulated = True  # this would be the first to add back in
    num_notes = len(list_notes_and_rests)
    # This part is taken from DeepBach
    while k < length:
        if j < num_notes - 1:
            if list_notes_and_rests[j + 1].offset > k / 4.0:
                n = list_notes_and_rests[j]
                t[k] = n.nameWithOctave if n.isNote else "Re"
                kp = k - 1
                while kp >= 0:
                    if t[kp] == "Sa":
                        kp = kp - 1
                    elif t[kp] == t[k]:
                        t[k] = "Sa"
                    else:
                        break
                k += 1
            else:
                j += 1
        else:
            n = list_notes_and_rests[j]
            t[k] = n.nameWithOctave if n.isNote else "Re"
            kp = k - 1
            while kp >= 0:
                if t[kp] == "Sa":
                    kp = kp - 1
                elif t[kp] == t[k]:
                    t[k] = "Sa"
                else:
                    break
            k += 1
    encoded.append(t)

print(encoded)
