"""
Данные для предобучения — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import hashlib
import re

# Верхняя граница значения хеша. Сигнатура пустого множества состоит из
# этих значений: минимума не по чему брать, а «бесконечность» удобна тем,
# что любой реальный хеш её вытеснит.
MAX_HASH = 2 ** 256 - 1


def clean_text(text):
    """Чистит сырой текст: теги, ссылки, управляющие символы, лишние пробелы.

    clean_text("<p>Hello  world</p>")        ->  "Hello world"
    clean_text("see http://x.com/a now")     ->  "see now"
    clean_text("a\\n\\n\\n\\nb")                 ->  "a\\n\\nb"

    Порядок важен: сначала убрать разметку, потом схлопывать пробелы —
    иначе на месте вырезанного тега останется двойной пробел.

    В уроке фильтр выбрасывает всё, кроме ASCII (`[^\\x20-\\x7E\\n]`). Мы так
    НЕ делаем: на многоязычном корпусе это стирает кириллицу и иероглифы
    целиком. Убираем только управляющие символы.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = text.replace("\t", " ")
    # isprintable() ложна для управляющих символов и табов, но истинна для
    # пробела и любых букв — ровно то, что нужно
    text = "".join(ch for ch in text if ch == "\n" or ch.isprintable())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def quality_filter(text, min_words=50, max_caps_ratio=0.3, max_special_ratio=0.1):
    """Пропускает документ дальше или отбраковывает. True — оставить.

    Три проверки, каждая ловит свой класс мусора:
      слишком коротко              — заглушка, страница-обрубок;
      много слов ЦЕЛИКОМ КАПСОМ    — SEO-спам;
      много не-буквенных символов  — списки ключевиков, битая разметка.

    quality_filter("hi", min_words=5)                    ->  False
    quality_filter("a b c d e", min_words=5)             ->  True
    quality_filter("BUY NOW CHEAP PILLS OK", min_words=5)  ->  False

    Доли считаются от размера самого документа, поэтому фильтр не зависит
    от длины: удвоенный документ проходит ровно так же, как исходный.
    """
    words = text.split()
    if len(words) < min_words:
        return False
    caps = sum(1 for w in words if w.isupper())
    if caps / len(words) > max_caps_ratio:
        return False
    special = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return special / max(len(text), 1) <= max_special_ratio


def shingles(text, k=5):
    """Множество k-словных шинглов документа (в нижнем регистре).

    shingles("the quick brown fox", 3)
        ->  {"the quick brown", "quick brown fox"}
    shingles("short", 3)  ->  set()

    Документ из n слов даёт n - k + 1 шинглов. Если слов меньше k, шинглов
    нет вообще — множество пустое, и такой документ дедупликация трогать
    не должна.

    Регистр съедаем: "The Cat" и "the cat" — один и тот же текст.
    """
    words = text.lower().split()
    if len(words) < k:
        return set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(set_a, set_b):
    """Мера Жаккара: доля общего в объединении.

    jaccard({1, 2}, {1, 2})     ->  1.0
    jaccard({1, 2}, {3, 4})     ->  0.0
    jaccard({1, 2}, {2, 3})     ->  0.3333...
    jaccard(set(), set())       ->  0.0

    Два пустых документа не «одинаковые», а «несравнимые» — договорились
    возвращать 0.0, а не 1.0 и не ZeroDivisionError.
    """
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def minhash_signature(shingle_set, num_hashes=64, seed=0):
    """MinHash-сигнатура: num_hashes чисел, сжимающих множество любого размера.

    Для каждой из num_hashes хеш-функций берём МИНИМУМ хеша по всем
    шинглам. Вероятность совпадения минимумов у двух множеств равна их
    мере Жаккара — на этом всё и держится.

    len(minhash_signature({"a", "b"}, num_hashes=8))  ->  8
    minhash_signature(set(), num_hashes=2)            ->  [MAX_HASH, MAX_HASH]

    seed обязателен: встроенный hash() в Python рандомизируется от запуска
    к запуску, и дедупликация переставала бы быть воспроизводимой. Берём
    hashlib — он даёт одинаковые числа всегда.

    Ловушка: две пустые сигнатуры совпадут во всех позициях, и оценка
    сходства выйдет 1.0, хотя jaccard честно скажет 0.0. Пустые документы
    надо отсекать до дедупликации.
    """
    signature = []
    for i in range(num_hashes):
        best = MAX_HASH
        for shingle in shingle_set:
            h = int(hashlib.sha256(f"{seed}:{i}:{shingle}".encode()).hexdigest(), 16)
            if h < best:
                best = h
        signature.append(best)
    return signature


def estimate_jaccard(sig_a, sig_b):
    """Оценка меры Жаккара по двум сигнатурам: доля совпавших позиций.

    estimate_jaccard([1, 2, 3], [1, 9, 3])  ->  0.6666...
    estimate_jaccard([], [])                ->  0.0

    Смысл всей затеи: сравнить два документа по 64 числам вместо тысяч
    шинглов. Точность растёт как 1/sqrt(num_hashes) — на 64 хешах ошибка
    порядка 0.12, на 256 — вдвое меньше.
    """
    if not sig_a or not sig_b:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def deduplicate(documents, threshold=0.8, k=5, num_hashes=64, bands=16, seed=0):
    """Убирает почти-дубликаты. Возвращает (оставшиеся, сколько удалено).

    Из каждой группы дубликатов остаётся ПЕРВЫЙ документ, порядок
    остальных сохраняется.

    deduplicate(["a b c d e f", "a b c d e f"], k=3)  ->  (["a b c d e f"], 1)

    Схема: шинглы -> MinHash -> LSH-корзины по полосам -> честный jaccard
    только для кандидатов из одной корзины. Полный перебор пар — O(n^2),
    на миллиардах документов это неисполнимо; LSH сводит его к почти
    линейному, ценой того, что редкую пару можно пропустить.

    Команда Llama так выбросила около 38% веб-данных. Дубликаты не просто
    жгут compute: модель начинает воспроизводить их дословно, и они же
    протекают из обучения в валидацию, завышая метрику.
    """
    sets = [shingles(doc, k) for doc in documents]
    signatures = [minhash_signature(s, num_hashes, seed) for s in sets]
    rows = num_hashes // bands

    buckets = {}
    for idx, sig in enumerate(signatures):
        if not sets[idx]:
            continue  # пустые множества дают ложное совпадение сигнатур
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            buckets.setdefault(key, []).append(idx)

    removed = set()
    for group in buckets.values():
        for pos, i in enumerate(group):
            if i in removed:
                continue
            for j in group[pos + 1:]:
                # честный jaccard обязателен: LSH даёт кандидатов, а не ответ
                if j not in removed and jaccard(sets[i], sets[j]) >= threshold:
                    removed.add(j)

    kept = [doc for idx, doc in enumerate(documents) if idx not in removed]
    return kept, len(removed)


def pack_sequences(token_ids, seq_length, pad_id=0):
    """Режет поток токенов на куски длины seq_length. Возвращает (куски, маски).

    Маска — список 0/1 той же длины: 1 у настоящего токена, 0 у добивки.

    pack_sequences([1, 2, 3, 4], 2)  ->  ([[1, 2], [3, 4]], [[1, 1], [1, 1]])
    pack_sequences([1, 2, 3], 2)     ->  ([[1, 2], [3, 0]], [[1, 1], [1, 0]])
    pack_sequences([], 4)            ->  ([], [])

    Документы идут встык, а не каждый со своей добивкой: последовательность
    из 200 токенов, дополненная до 2048, тратит 90% compute на нули.
    Добивается только последний, неполный кусок.
    """
    sequences = []
    masks = []
    for start in range(0, len(token_ids), seq_length):
        chunk = list(token_ids[start:start + seq_length])
        mask = [1] * len(chunk)
        pad = seq_length - len(chunk)
        if pad > 0:
            chunk += [pad_id] * pad
            mask += [0] * pad
        sequences.append(chunk)
        masks.append(mask)
    return sequences, masks
