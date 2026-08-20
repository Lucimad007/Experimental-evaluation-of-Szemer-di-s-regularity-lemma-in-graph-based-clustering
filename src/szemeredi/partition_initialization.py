"""Step 1 (partition initialization) of the Alon algorithm.

Two strategies are provided:
- ``random``: assign vertices to classes uniformly at random.
- ``degree_based``: order vertices by degree and split into contiguous blocks.

Both produce an equitable partition ``V = V0 ∪ V1 ∪ … ∪ Vb`` with
``|V1| = ⌊n/b⌋`` and ``|V0| < b``. The result is stored on ``self.classes`` as an
integer array where ``0`` denotes the exceptional class ``V0`` and ``1..b`` the
regular classes.

See ``spec/regularity_partitioning.md``.
"""

import numpy as np


def random(self, b=2):
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k
    for i in range(self.k):
        self.classes[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)] = i + 1
    np.random.shuffle(self.classes)


def degree_based(self, b=2):
    self.k = b
    self.classes = np.zeros(self.N)
    self.classes_cardinality = self.N // self.k
    for i in range(self.k):
        self.classes[self.degrees[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = i + 1
