"""§3.2 Step 1 (partition initialization) of Alon et al.

Divide V arbitrarily into an equitable partition P1 with 1+b classes
V0, V1, …, Vb, |V1| = ⌊n/b⌋, therefore |V0| < b. Set k1 = b.

Two implementations of "arbitrarily":
- ``random``: shuffle vertices into contiguous blocks.
- ``degree_based`` (default): order by degree, then split into blocks
  (Fiorucci et al., the code base modification 2 builds on).
"""

import numpy as np


def random(self, b=2):
    """§3.2 Step 1: equitable ``V = V0 ∪ V1 ∪ … ∪ Vb``, ``k1 ← b``."""
    self.k = b  # paper: k1 ← b
    self.classes = np.zeros(self.N)  # 0 = V0; leftover n % b vertices stay here (|V0| < b)
    self.classes_cardinality = self.N // self.k  # paper: |V1| = ⌊n/b⌋
    for i in range(self.k):
        self.classes[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)] = i + 1
    np.random.shuffle(self.classes)


def degree_based(self, b=2):
    """§3.2 Step 1: equitable ``V = V0 ∪ V1 ∪ … ∪ Vb``, ``k1 ← b``.

    "Dividing V arbitrarily" means any equitable split; degree order is the
    practical (Fiorucci) default. Pass ``random_initialization=True`` for a
    shuffled assignment.
    """
    self.k = b  # paper: k1 ← b
    self.classes = np.zeros(self.N)  # 0 = V0; leftover n % b vertices stay here (|V0| < b)
    self.classes_cardinality = self.N // self.k  # paper: |V1| = ⌊n/b⌋
    for i in range(self.k):
        self.classes[self.degrees[(i * self.classes_cardinality):((i + 1) * self.classes_cardinality)]] = i + 1
