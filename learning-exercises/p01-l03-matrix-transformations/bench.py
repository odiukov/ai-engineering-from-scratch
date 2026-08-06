"""Входные данные для замера скорости."""

import random

random.seed(0)

_M = [[1.0, 2.0], [3.0, 4.0]]
_v = [1.0, 1.0]

BENCH = {
    "rotation_matrix": (37,),
    "scaling_matrix": (2, 3),
    "apply": (_M, _v),
    "compose": (_M, _M),
    "determinant_2x2": (_M,),
    "eigenvalues_2x2": (_M,),
    "is_eigenvector": (_M, _v),
}
