"""Входные данные для замера скорости."""

import random

random.seed(0)          # обязательно: замер должен быть воспроизводим

_DIM = 32


def _vec():
    return [random.gauss(0, 1) for _ in range(_DIM)]


# 20 страниц по 64 патча — маленький ColPali-индекс
_page = [_vec() for _ in range(64)]
_pages = {f"page-{i:02d}": [_vec() for _ in range(64)] for i in range(20)}
_query = [_vec() for _ in range(12)]

BENCH = {
    "cosine": (_vec(), _vec()),
    "l2_normalize": (_vec(),),
    "maxsim": (_query, _page),
    "mean_sim": (_query, _page),
    "pool_page": (_page,),
    "bi_encoder_score": (_query, _page),
    "retrieve": (_query, _pages, 3),
    "storage_bytes": (10000,),
}
