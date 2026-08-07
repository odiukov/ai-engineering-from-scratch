"""Входные данные для замера скорости."""

import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "model", "data"]
_TEXT = " ".join(random.choice(_WORDS) for _ in range(1200))

_BYTES = list(_TEXT.encode("utf-8"))

# таблица слияний задана вручную: bench не имеет права импортировать solution
_MERGES = [
    ((116, 104), 256),  # "th"
    ((256, 101), 257),  # "the"
    ((32, 257), 258),   # " the"
    ((111, 120), 259),  # "ox"
    ((102, 259), 260),  # "fox"
]

_SPECIALS = {"<|im_start|>": 300, "<|im_end|>": 301}
_CHAT_TEXT = "<|im_start|>user\n" + _TEXT[:400] + "<|im_end|>\n"

_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": _TEXT[:200]},
    {"role": "assistant", "content": _TEXT[200:400]},
] * 20

BENCH = {
    "normalize": (_TEXT,),
    "pre_tokenize": (_TEXT,),
    "apply_merges": (_BYTES, _MERGES),
    "train_bpe": (_TEXT, 20),
    "encode": (_TEXT, _MERGES),
    "split_on_specials": (_CHAT_TEXT, _SPECIALS),
    "encode_with_specials": (_CHAT_TEXT, _MERGES, _SPECIALS),
    "apply_chat_template": (_MESSAGES, True),
}
