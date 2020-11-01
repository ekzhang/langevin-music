"""Starting file for running code.

Set up virtualenv with `pipenv install`, and run with `pipenv run python main.py`.
"""

from music21 import *

# Load a chorale and open in MuseScore
corpus.parse("bach/bwv438").show()
