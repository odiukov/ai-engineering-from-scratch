"""
Аудио-языковые модели — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def linear(x, W, b):
    """Линейный слой: вектор x → W @ x + b. Меняет размерность.

    linear([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])  ->  [1.0, 2.0]
    linear([1.0, 2.0], [[1.0, 1.0]], [5.0])                    ->  [8.0]

    W — список строк; длина строки равна len(x), число строк задаёт
    размерность выхода. Именно поэтому проектор и умеет превращать 1280
    признаков аудио-энкодера в 4096 измерений эмбеддингов LLM.

    Длина b должна совпадать с числом строк W, длина каждой строки — с
    длиной x. Иначе ValueError.
    """
    if len(W) != len(b):
        raise ValueError("bias length must match the number of rows in W")
    if any(len(row) != len(x) for row in W):
        raise ValueError("every row of W must have the same length as x")
    return [sum(w * v for w, v in zip(row, x)) + bias for row, bias in zip(W, b)]


def gelu(xs):
    """GELU поэлементно (tanh-приближение, то самое, что в nn.GELU).

    gelu([0.0])   ->  [0.0]
    gelu([10.0])  ->  примерно [10.0]   (большие числа проходят почти как есть)
    gelu([-10.0]) ->  примерно [0.0]    (большие отрицательные почти гасятся)

    Формула: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))).

    Отличие от ReLU принципиально: у GELU есть отрицательный «провал» —
    gelu(-1) примерно -0.159, а не ноль. Слой остаётся гладким, градиент
    в отрицательной зоне не умирает.
    """
    c = math.sqrt(2.0 / math.pi)
    return [0.5 * x * (1.0 + math.tanh(c * (x + 0.044715 * x ** 3))) for x in xs]


def project(frames, layers):
    """Проектор LALM: кадры аудио-энкодера → векторы в пространстве LLM.

    project([[1.0, 2.0]], [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0])])  ->  [[1.0, 2.0]]

    layers — список пар (W, b). Между слоями стоит GELU, ПОСЛЕ последнего
    слоя — нет: последний выход должен попасть в LLM как есть.

    Ключевое свойство: размерность выхода задаётся последним W и никак не
    зависит от размерности аудио-признаков. В этом весь смысл проектора —
    он мост, а не вычислитель.

    Число кадров не меняется: сколько кадров дал энкодер, столько
    аудио-«токенов» увидит LLM. Пустой layers — ValueError.

    Это и есть Stage 1 обучения: энкодер и LLM заморожены, учится только это.
    """
    if not layers:
        raise ValueError("projector needs at least one layer")
    out = []
    for frame in frames:
        v = frame
        for i, (W, b) in enumerate(layers):
            v = linear(v, W, b)
            if i < len(layers) - 1:  # нелинейность между слоями, но не в конце
                v = gelu(v)
        out.append(v)
    return out


def build_lm_sequence(parts, embed_text, dim):
    """Переплести текст и аудио в одну последовательность векторов для LLM.

    parts — список пар вида ("text", "hello") или ("audio", [[...], [...]]).
    embed_text — заглушка вместо токенайзера: строка → список векторов.

    build_lm_sequence([("text", "hi"), ("audio", [[1.0]])], emb, 1)
        ->  векторы токенов "hi", затем [1.0]

    Порядок сохраняется буквально: аудио, вставленное между двумя кусками
    текста, окажется ровно посередине. Для декодера разницы нет — он видит
    один плоский список векторов ОДНОЙ размерности dim, и по форме
    аудио-кадр неотличим от текстового токена.

    Вектор не той размерности — ValueError. Неизвестный вид куска (не
    "text" и не "audio") — ValueError: молча пропускать модальность нельзя.
    """
    out = []
    for kind, payload in parts:
        if kind == "text":
            vectors = embed_text(payload)
        elif kind == "audio":
            vectors = payload
        else:
            raise ValueError(f"unknown part kind: {kind}")
        for v in vectors:
            if len(v) != dim:
                raise ValueError("every vector must live in the LLM embedding space")
            out.append(list(v))
    return out


def trainable_parameter_count(modules, trainable):
    """Сколько параметров реально учится на этой стадии.

    modules — словарь «имя модуля → число параметров».
    trainable — набор имён, которые НЕ заморожены.

    trainable_parameter_count({"llm": 7e9, "proj": 5e6}, ["proj"])  ->  5000000.0

    Имя из trainable, которого нет в modules — ValueError: почти всегда это
    опечатка, из-за которой градиенты молча никуда не идут.

    Зачем: Stage 1 замораживает энкодер и LLM и учит один проектор — это
    доли процента от общего числа параметров, поэтому стадия и дешёвая.
    """
    total = 0.0
    for name in trainable:
        if name not in modules:
            raise ValueError(f"unknown module: {name}")
        total += modules[name]
    return total


def accuracy_by_category(items):
    """Точность по каждой категории плюс общая под ключом "overall".

    items — список словарей с полями "category", "predicted", "correct".

    accuracy_by_category([
        {"category": "speech", "predicted": "a", "correct": "a"},
        {"category": "multi",  "predicted": "b", "correct": "c"},
    ])  ->  {"speech": 1.0, "multi": 0.0, "overall": 0.5}

    "overall" считается по ВСЕМ вопросам сразу, а не как среднее категорий:
    у категорий разное число вопросов, и среднее средних врало бы.

    Пустой список — ValueError. Категория с именем "overall" — тоже
    ValueError, иначе она затрёт итог.

    Зачем: агрегат прячет провал. У всех моделей 2026 года общий MMAU-Pro
    около 52-60%, а подкатегория multi-audio — 20-26%, то есть уровень
    случайного тыка.
    """
    if not items:
        raise ValueError("items must not be empty")
    hits, seen = {}, {}
    for item in items:
        cat = item["category"]
        if cat == "overall":
            raise ValueError('"overall" is a reserved category name')
        seen[cat] = seen.get(cat, 0) + 1
        hits[cat] = hits.get(cat, 0) + (item["predicted"] == item["correct"])
    result = {cat: hits[cat] / seen[cat] for cat in seen}
    result["overall"] = sum(hits.values()) / sum(seen.values())
    return result


def is_above_chance(accuracy, n_choices, margin=0.05):
    """Отличается ли результат от случайного тыка на margin.

    is_above_chance(0.22, 4)  ->  False   (multi-audio: угадывание даёт 0.25)
    is_above_chance(0.60, 4)  ->  True

    Уровень случайности для вопроса с n вариантами равен 1 / n. Модель
    считается «понимающей», только если она выше этого уровня БОЛЬШЕ чем на
    margin — ровно на пороге ответ False.

    n_choices меньше 2, accuracy вне [0, 1] или отрицательный margin —
    ValueError.
    """
    if n_choices < 2:
        raise ValueError("n_choices must be at least 2")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")
    if margin < 0:
        raise ValueError("margin must not be negative")
    return accuracy > 1.0 / n_choices + margin


def gate_on_silence(answer, samples, threshold):
    """VAD-заглушка: на тишине гасим ответ модели, чтобы не выдать галлюцинацию.

    gate_on_silence("dog barks", [0.0, 0.0], 0.01)   ->  ""
    gate_on_silence("dog barks", [0.5, -0.5], 0.01)  ->  "dog barks"

    Громкость меряется как RMS: sqrt(среднее квадратов). Ответ проходит,
    только если RMS СТРОГО больше threshold; ровно на пороге — глушим.

    Ловушка: обычное среднее вместо RMS даёт ноль на любом знакопеременном
    сигнале, и громкая запись [-1, 1, -1, 1] будет принята за тишину.
    Квадрат убирает знак — именно поэтому он там и стоит.

    Пустой samples или отрицательный threshold — ValueError.

    Зачем: LALM на Whisper-энкодере наследует его любимый баг — на тишине
    уверенно выдумывает событие.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    if threshold < 0:
        raise ValueError("threshold must not be negative")
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    return answer if rms > threshold else ""
