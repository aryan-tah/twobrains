import numpy as np, scipy.sparse as sp
from twobrains.ablation import masked


def test_masked_removes_rows_and_columns():
    W = sp.csc_matrix(np.ones((4, 4), np.float32))
    keep = np.array([1, 0, 1, 1], bool)
    M = masked(W, keep).toarray()
    assert M[1].sum() == 0 and M[:, 1].sum() == 0
    assert M[0, 2] == 1
