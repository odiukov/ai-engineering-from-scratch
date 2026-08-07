"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["the", "cat", "sat", "on", "mat", "dog", "log", "frog", "ate", "rat"]
_TEXT = " ".join(random.choice(_WORDS) for _ in range(1500))

_TOKENS = list(_TEXT.encode("utf-8"))

# небольшая таблица слияний, собранная руками: bench не имеет права
# импортировать solution, поэтому вход задаётся константой
_MERGES = [
    ((116, 104), 256),  # "th"
    ((256, 101), 257),  # "the"
    ((32, 257), 258),   # " the"
    ((97, 116), 259),   # "at"
    ((115, 259), 260),  # "sat"
]

_VOCAB = {i: bytes([i]) for i in range(256)}
_VOCAB[256] = b"th"
_VOCAB[257] = b"the"
_VOCAB[258] = b" the"
_VOCAB[259] = b"at"
_VOCAB[260] = b"sat"

_IDS = [random.randrange(256) for _ in range(6000)]

BENCH = {
    "count_pairs": (_TOKENS,),
    "merge_pair": (_TOKENS, (116, 104), 256),
    "bpe_best_pair": (_TOKENS,),
    "wordpiece_best_pair": (_TOKENS,),
    "train_bpe": (_TEXT, 30),
    "encode": (_TEXT, _MERGES),
    "decode": (_IDS, _VOCAB),
    "tokenization_stats": (_TEXT, _MERGES),
}
