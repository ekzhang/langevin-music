"""Starting file for running code.

Set up virtualenv with `pipenv install`, and run with `pipenv run python main.py`.
"""

from music21 import *
import numpy as np

it = corpus.chorales.Iterator()
#   = islice(music21.corpus.chorales.Iterator(), 3)
for i in it:
    #cps = corpus.parse(i)
    #print(i.analyze('key'))
    list_notes_and_rests = list(i.parts[0].flat.getElementsByOffset(
        offsetStart=0.0,
        offsetEnd=i.flat.highestTime,
                classList=[note.Note,
                           note.Rest]))
    #print(list_notes_and_rests)
    length = int((i.flat.highestTime - 0.0) * 4)
    # 4 sixteenths/beat
    j = 0
    i = 0
    t = np.empty(length, dtype="S10")
    is_articulated = True
    num_notes = len(list_notes_and_rests)
    # This part is taken from DeepBach
    while i < length:
        if j < num_notes - 1:
            if (list_notes_and_rests[j + 1].offset > i
                    / 4.0 ):
                n = list_notes_and_rests[j]
                t[i] = n.nameWithOctave if n.isNote else 'Re'
                i += 1
            else:
                j += 1
        else:
            n = list_notes_and_rests[j]
            t[i] = n.nameWithOctave if n.isNote else 'Re'
            i += 1
    print(t)

