"""
Системы вопрос-ответ — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re
from collections import Counter


def normalize_answer(text):
    """Нормализация ответа по правилам SQuAD перед сравнением.

    Нижний регистр, вся пунктуация УДАЛЯЕТСЯ (не заменяется пробелом),
    артикли a/an/the выбрасываются как отдельные слова, пробелы схлопываются.

    normalize_answer("The  Beatles!")    ->  "beatles"
    normalize_answer("June 29, 2007")    ->  "june 29 2007"
    normalize_answer("the theater")      ->  "theater"

    Ловушка: артикли режутся только как ЦЕЛЫЕ слова. Если вычёркивать
    подстроку "the", то "theater" превратится в "ater" и метрика начнёт
    врать в вашу пользу.

    Зачем в AI: Exact Match и F1 в SQuAD считаются поверх этой нормализации.
    Без неё "The Beatles" и "beatles" — разные ответы.
    """
    text = text.lower()
    # пунктуацию именно удаляем: "don't" -> "dont", как в официальном скрипте SQuAD
    text = re.sub(r"[^a-z0-9\s]", "", text)
    words = [w for w in text.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def exact_match(prediction, reference):
    """Exact Match: 1.0 если нормализованные строки совпали, иначе 0.0.

    exact_match("The Beatles", "beatles")        ->  1.0
    exact_match("June 29th, 2007", "June 29, 2007")  ->  0.0

    Второй пример — не баг, а честное поведение метрики: порядковое
    окончание "th" переживает нормализацию и ломает точное совпадение.
    Именно поэтому одного EM мало и рядом всегда считают F1.
    """
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction, reference):
    """Token-level F1: частичный зачёт за пересечение токенов.

    token_f1("June 29th 2007", "June 29 2007")  ->  0.666...
    token_f1("the cat", "a cat")                ->  1.0
    token_f1("dogs", "cats")                    ->  0.0

    Пересечение считается как МУЛЬТИМНОЖЕСТВО: если слово встретилось в
    предсказании трижды, а в эталоне один раз, зачёт идёт один.

    Крайний случай: если после нормализации одна из строк пуста, F1 равен
    1.0 только когда пусты обе, иначе 0.0 (делить будет не на что).
    """
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        # обе пустые — считаем совпадением, одна пустая — промах
        return 1.0 if pred == ref else 0.0
    common = Counter(pred) & Counter(ref)  # пересечение мультимножеств
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def best_span(start_scores, end_scores, max_answer_len=10):
    """Голова extractive QA: выбрать (start, end) с максимумом суммы оценок.

    Ограничения: end >= start и длина ответа end - start + 1 <= max_answer_len.
    При равных суммах побеждает самый ранний start, затем самый ранний end.

    best_span([0.1, 5.0, 0.2], [0.3, 0.4, 9.0])        ->  (1, 2)
    best_span([9.0, 0.1], [0.1, 0.2], max_answer_len=1) ->  (0, 0)

    Ловушка: взять argmax отдельно по start_scores и отдельно по end_scores
    нельзя — легко получить end раньше start, то есть спан "наизнанку".
    Перебирай пары.

    Если списки пустые или разной длины — ValueError.
    """
    if len(start_scores) != len(end_scores):
        raise ValueError("start_scores and end_scores must have the same length")
    if not start_scores:
        raise ValueError("scores must not be empty")
    if max_answer_len < 1:
        raise ValueError("max_answer_len must be at least 1")
    best = None
    for i in range(len(start_scores)):
        # j не уходит дальше i + max_answer_len - 1, поэтому длина под контролем
        for j in range(i, min(len(end_scores), i + max_answer_len)):
            score = start_scores[i] + end_scores[j]
            # строгое > оставляет самый ранний спан при равенстве
            if best is None or score > best[0]:
                best = (score, i, j)
    return (best[1], best[2])


def answer_span(tokens, start_scores, end_scores, max_answer_len=10):
    """Достать сам текст ответа из пассажа по лучшему спану.

    answer_span(["it", "was", "June", "29"], [0, 0, 5, 0], [0, 0, 0, 5])
        ->  "June 29"

    Конец спана ВКЛЮЧИТЕЛЬНЫЙ: срез идёт до end + 1. Забыть про +1 —
    классический способ потерять последнее слово ответа.
    """
    if len(tokens) != len(start_scores):
        raise ValueError("tokens and scores must have the same length")
    start, end = best_span(start_scores, end_scores, max_answer_len)
    return " ".join(tokens[start : end + 1])


def retrieve_top_k(question, corpus, top_k=2):
    """Ретривер: вернуть top_k пассажей как список (score, index) по убыванию.

    Оценка — доля слов вопроса, встретившихся в пассаже:
    score = |Q ∩ D| / |Q| на множествах нормализованных токенов.
    Это лексический заменитель dense-ретривера: значение так же лежит в
    [0, 1], как косинус у Sentence-BERT, и так же годится под порог.

    retrieve_top_k("when was the iPhone released",
                   ["The iPhone was released in 2007.", "Dogs bark."], 1)
        ->  [(0.75, 0)]

    При равных оценках раньше идёт пассаж с меньшим индексом.
    """
    q = set(normalize_answer(question).split())
    scored = []
    for i, passage in enumerate(corpus):
        d = set(normalize_answer(passage).split())
        # пустой вопрос не должен ронять функцию делением на ноль
        score = len(q & d) / len(q) if q else 0.0
        scored.append((score, i))
    # ключ (-score, i): убывание по оценке, возрастание по индексу при ничьей
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[:top_k]


def recall_at_k(rankings, gold_indices, top_k=5):
    """Retrieval recall@k: доля вопросов, у которых нужный пассаж попал в топ-k.

    rankings — список выдач ретривера (по одной на вопрос), каждая в формате
    retrieve_top_k. gold_indices — индекс правильного пассажа для каждого вопроса.

    recall_at_k([[(0.9, 3), (0.2, 1)]], [3], 1)  ->  1.0
    recall_at_k([[(0.9, 3), (0.2, 1)]], [1], 1)  ->  0.0
    recall_at_k([[(0.9, 3), (0.2, 1)]], [1], 2)  ->  1.0

    Это главное число всего RAG: ридер физически не может ответить по
    пассажу, которого ретривер не принёс. Меряй recall ДО качества ответов.
    """
    if len(rankings) != len(gold_indices):
        raise ValueError("rankings and gold_indices must have the same length")
    if not rankings:
        raise ValueError("rankings must not be empty")
    hits = 0
    for ranking, gold in zip(rankings, gold_indices):
        if any(idx == gold for _, idx in ranking[:top_k]):
            hits += 1
    return hits / len(rankings)


def answer_with_refusal(question, corpus, threshold=0.5):
    """Отказ по порогу: если лучший пассаж слабее threshold — не отвечать.

    Вернуть текст лучшего пассажа либо строку "I don't know." ровно в таком
    написании.

    answer_with_refusal("who released the iPhone",
                        ["Apple released the iPhone."], 0.5)  ->  "Apple released the iPhone."
    answer_with_refusal("what is the capital of Peru",
                        ["Apple released the iPhone."], 0.5)  ->  "I don't know."

    Пустой корпус — тоже отказ, а не исключение.

    Это refusal calibration из урока: система обязана уметь молчать, когда
    ретривер ничего похожего не нашёл. Уверенный неправильный ответ дороже
    честного "не знаю".
    """
    top = retrieve_top_k(question, corpus, top_k=1)
    if not top or top[0][0] < threshold:
        return "I don't know."
    return corpus[top[0][1]]
