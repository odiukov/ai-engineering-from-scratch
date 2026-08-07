"""
Собираем токенизатор с нуля — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re
import unicodedata

# Пре-токенизатор в духе GPT-2, но на стандартном re вместо стороннего regex.
# \p{L} в стандартной библиотеке нет, зато [^\W\d] — это ровно «буква или
# подчёркивание». Важно, что альтернативы покрывают ЛЮБОЙ символ: буквы,
# цифры, всё остальное непробельное, пробелы. Иначе finditer молча
# выбрасывает то, что не подошло ни под один вариант — так теряются
# иероглифы, если писать шаблон через [a-zA-Z].
PRETOKEN_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def normalize(text):
    """Нормализация Unicode по NFKC — приводит разные записи к одной.

    normalize("ﬁle")   ->  "file"    (лигатура ﬁ разъезжается в два символа)
    normalize("Ａ")     ->  "A"       (полноширинная латиница схлопывается)
    normalize("hello") ->  "hello"

    Без этого шага одно и то же слово получает разные id в зависимости от
    того, как его набрали. Это то, что делает tokenizers.normalizers.NFKC().
    """
    return unicodedata.normalize("NFKC", text)


def pre_tokenize(text):
    """Режет текст на куски ДО применения BPE.

    pre_tokenize("Hello, world!")  ->  ["Hello", ",", " world", "!"]
    pre_tokenize("Don't stop")     ->  ["Don", "'t", " stop"]
    pre_tokenize("x = 1")          ->  ["x", " =", " 1"]

    Ведущий пробел прилипает к слову: " world", а не " " и "world".
    Именно поэтому в словаре GPT-2 "the" и " the" — два разных токена.

    Главное свойство: склейка кусков обязана дать исходный текст обратно,
    символ в символ. Если шаблон покрывает не все символы, finditer молча
    их выкинет, и текст начнёт теряться на иероглифах и эмодзи.
    """
    return [m.group() for m in PRETOKEN_PATTERN.finditer(text)]


def apply_merges(byte_seq, merges):
    """Применяет список слияний к одной последовательности байтов.

    apply_merges([97, 97, 98], [((97, 97), 256)])  ->  [256, 98]
    apply_merges([97, 98], [])                     ->  [97, 98]

    merges — список [(пара, новый_id), ...] в порядке обучения; порядок
    менять нельзя, поздние слияния собираются из результатов ранних.
    Внутри одного слияния пара съедается без перекрытия: [5,5,5] по паре
    (5,5) даёт [x, 5], а не [x, x].
    """
    tokens = list(byte_seq)
    for pair, new_id in merges:
        out = []
        i = 0
        n = len(tokens)
        while i < n:
            if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


def train_bpe(text, num_merges):
    """Обучает byte-level BPE ПО КУСКАМ пре-токенизатора. Возвращает (merges, vocab).

    Пары считаются внутри каждого куска отдельно, поэтому слияние не может
    перепрыгнуть границу слова: токена "t th" не появится никогда.

    train_bpe("ab ab", 1)  ->  ([((97, 98), 256)], vocab с vocab[256] == b"ab")

    Порядок шагов: normalize -> pre_tokenize -> байты каждого куска ->
    цикл «посчитать пары, слить самую частую». При равной частоте берём
    лексикографически меньшую пару, чтобы обучение было воспроизводимым.

    Это то, что делает tokenizers.trainers.BpeTrainer, только на Rust и на
    гигабайтах.
    """
    chunks = [list(c.encode("utf-8")) for c in pre_tokenize(normalize(text))]
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for i in range(num_merges):
        pairs = {}
        for seq in chunks:
            for j in range(len(seq) - 1):
                key = (seq[j], seq[j + 1])
                pairs[key] = pairs.get(key, 0) + 1
        if not pairs:
            break
        best = max(pairs, key=lambda p: (pairs[p], -p[0], -p[1]))
        new_id = 256 + i
        # применяем ровно одно слияние ко всем кускам: apply_merges умеет
        # список, значит и список из одного элемента
        chunks = [apply_merges(seq, [(best, new_id)]) for seq in chunks]
        merges.append((best, new_id))
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
    return merges, vocab


def encode(text, merges):
    """Полный проход: normalize -> pre_tokenize -> BPE по каждому куску.

    encode("ab", [])                  ->  [97, 98]
    encode("ab", [((97, 98), 256)])   ->  [256]

    Каждый кусок кодируется НЕЗАВИСИМО, и только потом результаты
    склеиваются в один список id. Если склеить байты всех кусков и лишь
    затем применить слияния, граница слов перестанет держать, и вся
    работа пре-токенизатора пропадёт.
    """
    ids = []
    for chunk in pre_tokenize(normalize(text)):
        ids.extend(apply_merges(list(chunk.encode("utf-8")), merges))
    return ids


def split_on_specials(text, specials):
    """Режет текст на куски вокруг служебных токенов.

    Возвращает список пар (кусок, это_служебный_токен).

    split_on_specials("a<|end|>b", {"<|end|>": 300})
        ->  [("a", False), ("<|end|>", True), ("b", False)]
    split_on_specials("abc", {})  ->  [("abc", False)]
    split_on_specials("", {})     ->  []

    Пустые куски не возвращаются. Ловушка: если один служебный токен —
    префикс другого ("<|im_end|>" и "<|im_end|>x"), совпадение ищется от
    САМОГО ДЛИННОГО, иначе длинный никогда не сработает.
    """
    if not text:
        return []
    if not specials:
        return [(text, False)]
    # сортировка по убыванию длины — единственный способ заставить
    # чередование в регулярке предпочитать длинный вариант короткому
    pattern = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))
    parts = []
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            parts.append((text[last:m.start()], False))
        parts.append((m.group(), True))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], False))
    return parts


def encode_with_specials(text, merges, specials):
    """Кодирует текст, подставляя фиксированные id служебных токенов.

    encode_with_specials("a<|end|>", [], {"<|end|>": 300})  ->  [97, 300]

    Служебные токены НЕ участвуют в BPE: они находятся точным совпадением
    до слияний и заменяются своим id целиком. Иначе "<|end|>" развалится
    на куски и модель не увидит границу сообщения.
    """
    ids = []
    for part, is_special in split_on_specials(text, specials):
        if is_special:
            ids.append(specials[part])
        else:
            ids.extend(encode(part, merges))
    return ids


def apply_chat_template(messages, add_generation_prompt=False):
    """Собирает список сообщений в плоскую строку формата ChatML.

    messages — список словарей {"role": ..., "content": ...}.

    apply_chat_template([{"role": "user", "content": "Hi"}])
        ->  "<|im_start|>user\\nHi<|im_end|>\\n"

    apply_chat_template([{"role": "user", "content": "Hi"}], True)
        ->  "<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\n"

    Каждый блок: IM_START, роль, перевод строки, текст, IM_END, перевод
    строки. add_generation_prompt дописывает открытый блок ассистента —
    это и есть сигнал модели «теперь говори ты».

    Лишний пробел или потерянный \\n выводят вход за пределы обучающего
    распределения, и модель начинает нести чушь. Формат должен совпадать
    с обучением байт в байт.
    """
    out = [f"{IM_START}{msg['role']}\n{msg['content']}{IM_END}\n" for msg in messages]
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)
