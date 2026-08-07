"""
Распознавание именованных сущностей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def spans_to_bio(tokens, spans):
    """Разметка BIO по списку сущностей. spans — тройки (start, end, label).

    Граница end — НЕ включительно, как в срезах Python. Первый токен
    сущности получает "B-label", остальные — "I-label", всё прочее — "O".

    spans_to_bio(["New", "York", "is", "big"], [(0, 2, "GPE")])
        ->  ['B-GPE', 'I-GPE', 'O', 'O']

    Ловушка: BIO не умеет вложенные и пересекающиеся сущности. "Bank of
    America Tower" — одновременно ORG и FACILITY, и одной строкой меток это
    не записать. Пересечение — это ValueError, а не «победит последний».
    """
    labels = ["O"] * len(tokens)
    taken = [False] * len(tokens)
    for start, end, label in spans:
        for i in range(start, end):
            if taken[i]:
                raise ValueError(f"overlapping spans at token {i}")
            taken[i] = True
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(labels):
    """Обратное преобразование: из BIO-меток обратно в тройки (start, end, label).

    bio_to_spans(["B-ORG", "O", "B-ORG", "O", "B-PRODUCT"])
        ->  [(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
    bio_to_spans(["B-GPE", "I-GPE", "O"])  ->  [(0, 2, 'GPE')]

    Ловушка: "I-X" продолжает сущность, только если предыдущая сущность
    была того же типа X. "I-LOC" сразу после "B-ORG" — сломанная разметка,
    продолжением её считать нельзя.

    Ловушка: сущность, доходящая до конца предложения, закрывается уже
    после цикла. Забыть про это — потерять последнюю сущность в каждом
    втором тексте.

    В уроке функция принимает ещё и tokens, но не использует их: длину даёт
    сам список меток.
    """
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
            current = None
    if current:  # сущность у самого края предложения
        spans.append(current)
    return spans


def is_valid_bio(labels):
    """Проверка корректности BIO-последовательности.

    Правила: "O" законно всегда, "B-X" законно всегда, "I-X" законно
    только сразу после "B-X" или "I-X" ТОГО ЖЕ типа. Всё остальное — брак.

    is_valid_bio(["B-ORG", "I-ORG", "O"])  ->  True
    is_valid_bio(["O", "I-ORG"])           ->  False
    is_valid_bio(["B-ORG", "I-GPE"])       ->  False

    Ловушка: "I-" в самой первой позиции невалидно всегда — продолжать
    нечего.

    Зачем: именно эти переходы CRF выучивает как маловероятные. Прежде чем
    ставить CRF, полезно уметь сказать, что вообще считается валидным.
    """
    prev_type = None  # тип открытой сущности, None — сущности нет
    for label in labels:
        if label == "O":
            prev_type = None
        elif label.startswith("B-"):
            prev_type = label[2:]
        elif label.startswith("I-"):
            if prev_type is None or prev_type != label[2:]:
                return False
        else:
            return False  # ни O, ни B-, ни I- — вообще не BIO
    return True


def word_shape(word):
    """Форма слова: заглавная -> X, строчная -> x, цифра -> d, прочее как есть.

    word_shape("iPhone")    ->  'xXxxxx'
    word_shape("USA-2024")  ->  'XXX-dddd'

    Ловушка: дефис, точка и апостроф остаются собой. Именно они отличают
    "U.S.A." от "USA", а это разные признаки для модели.

    Зачем: у классического NER заглавные буквы — самый сильный сигнал
    имени собственного, а форма слова упаковывает этот сигнал в строку.
    """
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)


def token_features(token, prev_token, next_token):
    """Словарь признаков одного токена для классического (не нейронного) NER.

    Ключи ровно такие: lower, is_upper, is_title, has_digit, suffix_3,
    shape, prev_lower, next_lower.

    token_features("Apple", None, "sued")["prev_lower"]  ->  '<BOS>'
    token_features("Apple", None, "sued")["shape"]       ->  'Xxxxx'
    token_features("2024", "in", None)["has_digit"]      ->  True

    Ловушка: краёв предложения не существует как токенов. Отсутствующий
    сосед — это "<BOS>" и "<EOS>", а не пустая строка: пустая строка
    сольётся с пустым токеном из битой токенизации.

    Ловушка: suffix_3 у слова короче трёх символов — это всё слово, срез
    token[-3:] так и работает, падать тут нечему.
    """
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def rule_based_ner(tokens, gazetteers):
    """Разметка по словарям сущностей. gazetteers — {метка: множество слов}.

    Токен, найденный в словаре, получает "B-метка", остальные — "O".

    rule_based_ner(["Apple", "sued"], {"ORG": {"Apple"}})  ->  ['B-ORG', 'O']

    Ловушка: слово может лежать сразу в двух словарях. Побеждает первая
    подходящая метка в порядке ключей gazetteers — порядок словаря и есть
    приоритет, и от него результат зависит.

    Ловушка: функция никогда не выдаёт "I-". Многословная сущность
    "New York" из словаря отдельных слов получит два "B-GPE" подряд, то
    есть развалится на две сущности. Это встроенное ограничение подхода.

    Так работал NER до статистических моделей: точность по известным
    именам высокая, покрытие новых нулевое, "Apple" фрукт от "Apple"
    компании не отличить в принципе.
    """
    labels = []
    for token in tokens:
        for label, words in gazetteers.items():
            if token in words:
                labels.append(f"B-{label}")
                break
        else:  # ни один словарь не сработал
            labels.append("O")
    return labels


def entity_f1(true_labels, pred_labels):
    """Метрики уровня СУЩНОСТЕЙ: {'precision', 'recall', 'f1'}.

    Совпадением считается только полное совпадение тройки (start, end,
    label). Угадать тип, но ошибиться в границе — это ноль, а не половина.

    entity_f1(["B-ORG", "I-ORG"], ["B-ORG", "I-ORG"])  ->  f1 = 1.0
    entity_f1(["B-ORG", "I-ORG"], ["B-ORG", "O"])      ->  f1 = 0.0

    Ловушка: token-level F1 на втором примере дал бы 0.5 и соврал. Именно
    поэтому NER меряют библиотекой seqeval, а не accuracy по токенам.

    Ловушка: пустое предсказание — это precision 0.0, а не деление на ноль.
    """
    true_spans = set(bio_to_spans(true_labels))
    pred_spans = set(bio_to_spans(pred_labels))
    hits = len(true_spans & pred_spans)

    precision = hits / len(pred_spans) if pred_spans else 0.0
    recall = hits / len(true_spans) if true_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def constrained_decode(token_scores):
    """Лучшая ВАЛИДНАЯ BIO-последовательность по оценкам каждого токена.

    token_scores — список словарей {метка: оценка}, по одному на токен.
    Вернуть последовательность меток с максимальной суммой оценок среди
    тех, что проходят is_valid_bio.

    constrained_decode([{"O": 1.0, "I-ORG": 5.0}])  ->  ['O']
        ("I-ORG" первым токеном невозможно, сколько бы ни весило)

    Ловушка: жадный argmax по каждому токену отдельно легко выдаёт
    невалидную цепочку вроде ["O", "I-ORG"]. Нужен проход динамическим
    программированием (Витерби): для каждой метки помнить лучшую сумму,
    с которой в неё можно прийти.

    Ловушка: при равных суммах порядок обязан быть детерминированным,
    иначе одинаковый вход даст разные ответы. Сравниваем (сумма, метка).

    Ловушка: валидного пути может не быть вовсе (все кандидаты — "I-").
    Это ValueError, а не молчаливый возврат мусора.

    Это ровно то, что CRF делает мягко: у него запрещённые переходы не
    невозможны, а просто очень маловероятны.
    """

    def can_follow(prev_label, label):
        if not label.startswith("I-"):
            return True
        if prev_label is None:
            return False
        return prev_label[2:] == label[2:] and prev_label[0] in "BI"

    # best[label] = (сумма, путь) — лучший способ дойти до этой метки
    best = {}
    for i, scores in enumerate(token_scores):
        step = {}
        for label, score in scores.items():
            if i == 0:
                if can_follow(None, label):
                    step[label] = (score, [label])
                continue
            # ключ (-сумма, метка) выбирает максимум, при равенстве —
            # лексикографически меньшую предыдущую метку
            options = [
                (total, prev) for prev, (total, _) in best.items() if can_follow(prev, label)
            ]
            if not options:
                continue
            total, prev = min(options, key=lambda o: (-o[0], o[1]))
            step[label] = (total + score, best[prev][1] + [label])
        if not step:
            raise ValueError(f"no valid BIO path through token {i}")
        best = step

    if not best:
        return []
    winner = min(best, key=lambda label: (-best[label][0], label))
    return best[winner][1]
