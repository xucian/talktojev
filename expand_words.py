"""Expand words.json (the frequency-ordered source list of build_vocab.py) with common English words.

Source of frequency: the `wordfreq` package (pip install wordfreq). A candidate is kept when it is
alphabetic, at least three letters (shorter only if already in the list), and either already in
the list or known to WordNet (which drops most proper nouns, abbreviations and web junk).
The old list is kept in full and in order; new words follow in frequency order.

    python expand_words.py [target_size]   # default 9500; writes words.json, keeps words.v3.json
"""
import json
import os
import shutil
import sys

from nltk.corpus import wordnet as wn
from wordfreq import top_n_list

HERE = os.path.dirname(os.path.abspath(__file__))
WORDS = os.path.join(HERE, "words.json")
BACKUP = os.path.join(HERE, "words.v3.json")
# A short list of slurs and the strongest profanity: the old list kept mild profanity on purpose.
BLOCK = {"nigger", "nigga", "faggot", "retard", "retarded", "cunt", "whore", "slut", "rape", "raped", "rapist"}


def main(target: int = 9500) -> None:
    old = json.load(open(WORDS))
    if not os.path.exists(BACKUP):
        shutil.copy(WORDS, BACKUP)
    have = set(old)
    out = list(old)
    for w in top_n_list("en", 40000):
        if len(out) >= target:
            break
        if w in have or not w.isalpha() or not w.isascii() or len(w) < 3 or w in BLOCK:
            continue
        if not wn.synsets(w):
            continue
        out.append(w)
        have.add(w)
    json.dump(out, open(WORDS, "w"), indent=0)
    print(f"words.json: {len(old)} -> {len(out)} (backup {os.path.basename(BACKUP)})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9500)
