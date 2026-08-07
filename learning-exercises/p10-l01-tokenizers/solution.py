"""
Токенизаторы: BPE, WordPiece, SentencePiece — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def count_pairs(tokens):
    """Сколько раз встречается каждая пара соседних токенов.

    count_pairs([1, 2, 1, 2])  ->  {(1, 2): 2, (2, 1): 1}
    count_pairs([7])           ->  {}

    Пары считаются с перекрытием, как в обычном BPE:
    count_pairs([5, 5, 5]) -> {(5, 5): 2}, а не 1.

    Это первый шаг каждой итерации обучения BPE: чтобы понять, что сливать,
    сначала надо посчитать, что вообще рядом стоит.
    """
    pairs = {}
    # один проход по списку, O(n) — на корпусе в мегабайты это единственный
    # вариант: пересчёт словарём медленнее в разы, но всё ещё линеен
    for i in range(len(tokens) - 1):
        key = (tokens[i], tokens[i + 1])
        pairs[key] = pairs.get(key, 0) + 1
    return pairs


def merge_pair(tokens, pair, new_id):
    """Заменяет все вхождения пары на один новый токен.

    merge_pair([1, 2, 3, 1, 2], (1, 2), 99)  ->  [99, 3, 99]
    merge_pair([5, 5, 5], (5, 5), 99)        ->  [99, 5]

    Ловушка в последнем примере: слияние идёт слева направо и БЕЗ
    перекрытия. Съев первые две пятёрки, мы обязаны шагнуть через обе,
    иначе один и тот же токен попадёт в два слияния сразу.
    """
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        # i < n - 1 обязательно проверить первым: иначе tokens[i + 1] вылетит
        # за конец списка на последнем элементе
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def bpe_best_pair(tokens):
    """Пара, которую BPE сольёт следующей: самая ЧАСТАЯ.

    bpe_best_pair([1, 2, 1, 2, 3, 4])  ->  (1, 2)
    bpe_best_pair([7])                 ->  None

    При равной частоте берём лексикографически меньшую пару — иначе
    обучение зависит от порядка обхода словаря и перестаёт быть
    воспроизводимым. У tiktoken и HuggingFace правило своё, но правило
    там тоже есть: тай-брейк обязан быть детерминированным.
    """
    pairs = count_pairs(tokens)
    if not pairs:
        return None
    # максимизируем частоту, а при равенстве — минимизируем саму пару;
    # минимум получаем сменой знака, чтобы обойтись одним max()
    return max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))


def train_bpe(text, num_merges):
    """Обучает byte-level BPE. Возвращает (merges, vocab).

    merges — список [(пара, новый_id), ...] СТРОГО в порядке обучения.
    vocab  — словарь {id: bytes}, где id 0..255 это одиночные байты.

    train_bpe("aaab", 1)  ->  ([((97, 97), 256)], vocab с vocab[256] == b"aa")
    train_bpe("abc", 0)   ->  ([], словарь из 256 байтов)

    Новые id выдаются подряд начиная с 256. Если пар не осталось (текст
    короче двух байтов), обучение останавливается раньше num_merges.

    Это ровно то, что делает tokenizers.ByteLevelBPETokenizer.train,
    только там 100 000 слияний на сотнях гигабайт и на Rust.
    """
    tokens = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for i in range(num_merges):
        pair = bpe_best_pair(tokens)
        if pair is None:
            break
        new_id = 256 + i
        tokens = merge_pair(tokens, pair, new_id)
        merges.append((pair, new_id))
        # склейка байтов родителей — именно она позволяет decode работать
        # без обратного разбора дерева слияний
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return merges, vocab


def encode(text, merges):
    """Кодирует текст в id, применяя слияния В ПОРЯДКЕ ОБУЧЕНИЯ.

    encode("aaab", [((97, 97), 256)])  ->  [256, 97, 98]
    encode("ab", [])                   ->  [97, 98]

    Порядок принципиален. Если слияние 1 собрало "th", а слияние 5 — "the"
    из "th" + "e", то применив 5 раньше 1, ты никогда не получишь "the":
    "th" ещё не существует.
    """
    tokens = list(text.encode("utf-8"))
    for pair, new_id in merges:
        tokens = merge_pair(tokens, pair, new_id)
    return tokens


def decode(ids, vocab):
    """Обратно из id в строку через таблицу байтов.

    decode([97, 98], {i: bytes([i]) for i in range(256)})  ->  "ab"

    Склеиваем байты всех токенов и только ПОТОМ декодируем в UTF-8: один
    токен может оборваться на середине многобайтового символа, и
    отдельный .decode() на нём упадёт. errors="replace" страхует от
    битых хвостов.
    """
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


def wordpiece_best_pair(tokens):
    """Пара, которую сольёт WordPiece: самая НЕОЖИДАННАЯ.

    Критерий: count(AB) / (count(A) * count(B)) — во сколько раз пара
    встречается чаще, чем если бы её половинки стояли рядом случайно.

    wordpiece_best_pair([1, 2, 1, 2, 3, 4])  ->  (3, 4)
    wordpiece_best_pair([7])                 ->  None

    В примере (1, 2) встречается дважды, а (3, 4) всего раз — BPE взял бы
    (1, 2). Но 3 и 4 больше нигде не появляются, их союз абсолютный,
    и WordPiece выбирает его. Так BERT получает морфемы вместо
    просто частых кусков.

    Тай-брейк тот же, что у BPE: при равном score — меньшая пара.
    """
    pairs = count_pairs(tokens)
    if not pairs:
        return None
    unigrams = {}
    for t in tokens:
        unigrams[t] = unigrams.get(t, 0) + 1

    def score(p):
        return pairs[p] / (unigrams[p[0]] * unigrams[p[1]])

    return max(pairs, key=lambda p: (score(p), -p[0], -p[1]))


def tokenization_stats(text, merges):
    """Метрики качества токенизатора на конкретном тексте.

    Возвращает словарь:
      "bytes"             — длина текста в байтах UTF-8
      "tokens"            — сколько токенов вышло
      "words"             — сколько слов (split по пробелам)
      "compression_ratio" — tokens / bytes, МЕНЬШЕ значит лучше
      "fertility"         — tokens / words, у GPT-4 на английском ~1.2

    tokenization_stats("ab", [])["compression_ratio"]  ->  1.0
    tokenization_stats("", [])["compression_ratio"]    ->  1.0

    На пустом тексте делить не на что — договорились возвращать 1.0 и 0.0,
    а не падать с ZeroDivisionError посреди прогона по корпусу.
    """
    ids = encode(text, merges)
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    return {
        "bytes": n_bytes,
        "tokens": len(ids),
        "words": n_words,
        "compression_ratio": len(ids) / n_bytes if n_bytes else 1.0,
        "fertility": len(ids) / n_words if n_words else 0.0,
    }
