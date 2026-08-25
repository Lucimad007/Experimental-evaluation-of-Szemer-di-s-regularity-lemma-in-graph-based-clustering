"""paper §3.2 step 1: partition initialization (alon et al. [30]).

divide v arbitrarily into an equitable p1 with 1+b classes v0, v1, …, vb,
|v1| = ⌊n/b⌋, therefore |v0| < b. set k1 = b.

two readings of "arbitrarily":
- random: shuffle vertices into contiguous blocks
- degree_based (default): order by degree then split (fiorucci et al. [28])
"""

import numpy as np


def random(self, b=2):
    """paper §3.2 step 1 / alg 1 line 1–2: equitable v = v0 ∪ v1…vb, k1 ← b."""
    # paper: k1 ← b
    self.k = b
    # 0 = v0; leftover n % b vertices stay here (|v0| < b)
    self.classes = np.zeros(self.N)
    # paper: |v1| = ⌊n/b⌋
    self.classes_cardinality = self.N // self.k
    for i in range(self.k):
        # fill class i+1 with a contiguous block, then shuffle labels
        self.classes[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)] = i + 1
    np.random.shuffle(self.classes)


def degree_based(self, b=2):
    """paper §3.2 step 1: same equitable split, vertices ordered by degree."""
    # paper: k1 ← b
    self.k = b
    # 0 = v0
    self.classes = np.zeros(self.N)
    # paper: |v1| = ⌊n/b⌋
    self.classes_cardinality = self.N // self.k
    for i in range(self.k):
        # assign the next ⌊n/b⌋ vertices in degree order to class i+1
        self.classes[self.degrees[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = i + 1
