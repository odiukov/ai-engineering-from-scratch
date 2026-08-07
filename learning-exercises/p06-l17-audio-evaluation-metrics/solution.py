"""
Метрики оценки аудио — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def normalize_text(text):
    """Нормализация перед подсчётом WER: нижний регистр, без пунктуации, один пробел.

    normalize_text("Please turn on the lights.")  ->  "please turn on the lights"
    normalize_text("  Привет,   МИР! ")           ->  "привет мир"
    normalize_text("!!!")                         ->  ""

    Это ровно то, что делает jiwer.Compose([ToLowerCase(), RemovePunctuation(),
    Strip()]) — только руками. Правило номер один из урока: нормализуй до
    подсчёта и сообщай, каким правилом нормализовал. Без этого «Hello.» и
    «hello» дадут 100% ошибок на ровном месте.

    Пунктуация выбрасывается, а не заменяется пробелом: так же, как в jiwer.
    """
    # isalnum вместо string.punctuation: последнее знает только ASCII,
    # а метрики считают и по русскому, и по китайскому
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())
    return " ".join(cleaned.split())


def edit_ops(reference, hypothesis):
    """Минимальный набор правок гипотезы до эталона: (substitutions, deletions, insertions).

    edit_ops(["a", "b", "c"], ["a", "b", "c"])  ->  (0, 0, 0)
    edit_ops(["a", "b", "c"], ["a", "x", "c"])  ->  (1, 0, 0)
    edit_ops(["a", "b", "c"], ["a", "c"])       ->  (0, 1, 0)
    edit_ops(["a", "c"], ["a", "b", "c"])       ->  (0, 0, 1)

    Расстояние Левенштейна с разбором по типам правок: S — не то слово,
    D — слово потеряли, I — слово придумали. Из этих трёх чисел собираются
    и WER, и CER.

    Ловушка направления: удаление — это лишнее слово в ЭТАЛОНЕ, вставка —
    лишнее в ГИПОТЕЗЕ. Перепутаешь — метрика останется правильной, а разбор
    ошибок ASR превратится в гадание.

    Сложность O(n*m) по времени, O(m) по памяти: держим только предыдущую
    строку таблицы.
    """
    m = len(hypothesis)
    # каждая клетка — (стоимость, subs, dels, ins); нулевая строка это
    # «эталон пуст, вся гипотеза — вставки»
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, len(reference) + 1):
        cur = [(i, 0, i, 0)]  # гипотеза пуста: весь эталон удалён
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cur.append(prev[j - 1])  # совпадение бесплатно
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            # min по кортежу сравнивает сначала стоимость, а при равенстве
            # разрешает ничью детерминированно — иначе разбор поплывёт
            cur.append(
                min(
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                    (dele[0] + 1, dele[1], dele[2] + 1, dele[3]),
                    (ins[0] + 1, ins[1], ins[2], ins[3] + 1),
                )
            )
        prev = cur
    _, subs, dels, ins = prev[m]
    return subs, dels, ins


def wer(reference, hypothesis):
    """Word Error Rate: (S + D + I) / N по словам после нормализации.

    wer("Please turn on the lights.", "please turn on the light")  ->  0.2
    wer("привет мир", "привет мир")                                ->  0.0

    N — число слов в ЭТАЛОНЕ, а не в гипотезе. Отсюда главное свойство:
    WER не ограничен единицей. Модель, которая на три слова эталона выдала
    двадцать, честно получит WER больше 1.0.

    Меньше 5% — это уровень человека на читаной речи.

    Пустой после нормализации эталон — ValueError: делить не на что.
    """
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        raise ValueError("эталон пуст после нормализации")
    subs, dels, ins = edit_ops(ref, hyp)
    return (subs + dels + ins) / len(ref)


def cer(reference, hypothesis):
    """Character Error Rate: та же формула, но по символам без пробелов.

    cer("Привет, мир!", "привет мир")  ->  0.0   (спасла нормализация)
    cer("кот", "кит")                  ->  ...   примерно 0.333

    Зачем отдельная метрика: в китайском и кантонском границы слов условны,
    и WER там меряет качество токенизатора, а не распознавания. Пробелы
    выбрасываются по той же причине.

    Пустой после нормализации эталон — ValueError.
    """
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        raise ValueError("эталон пуст после нормализации")
    subs, dels, ins = edit_ops(ref, hyp)
    return (subs + dels + ins) / len(ref)


def cosine_similarity(a, b):
    """SECS: косинус между эмбеддингами диктора — насколько клон похож на оригинал.

    cosine_similarity([1, 0], [1, 0])   ->  1.0
    cosine_similarity([1, 0], [0, 1])   ->  0.0
    cosine_similarity([1, 0], [-1, 0])  ->  -1.0

    Косинус, а не расстояние: длина ECAPA-эмбеддинга зависит от громкости и
    длительности записи, а направление — от голоса. Больше 0.75 считается
    узнаваемым клоном.

    Разная длина или нулевой вектор — ValueError: у нулевого направления нет.
    """
    if len(a) != len(b):
        raise ValueError("эмбеддинги разной длины")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        raise ValueError("нулевой эмбеддинг")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def percentile(values, p):
    """p-й процентиль с линейной интерполяцией: P50, P95, P99 задержки.

    percentile([1, 2, 3, 4], 50)   ->  2.5
    percentile([1, 2, 3, 4], 0)    ->  1.0
    percentile([1, 2, 3, 4], 100)  ->  4.0

    Правило номер два из урока: сообщай распределение, а не среднее. Один
    зависший запрос на сотню подтянет среднюю задержку и останется невидим
    в P50 — а пользователь его почувствует, и увидишь ты его только в P99.

    Ловушка: сортировать надо КОПИЮ. sort() на входном списке молча
    переставит замеры вызывающей стороны.

    Пустой список или p вне [0, 100] — ValueError.
    """
    if not values:
        raise ValueError("пустая выборка")
    if not 0 <= p <= 100:
        raise ValueError("p должен лежать в [0, 100]")
    ordered = sorted(values)  # sorted, а не .sort(): вход не наш
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def der(false_alarm_s, miss_s, confusion_s, total_speech_s):
    """Diarization Error Rate: (FA + Miss + Confusion) / общее время речи.

    der(1.0, 2.0, 3.0, 100.0)  ->  0.06
    der(0.0, 0.0, 0.0, 100.0)  ->  0.0

    Три ошибки разной природы: FA — приписали речь тишине, Miss — прозевали
    речь, Confusion — речь нашли, но отдали не тому диктору.

    Знаменатель — время речи, а не длина записи, поэтому DER спокойно
    превышает 1.0: система, размечающая как речь всю тишину, наберёт
    сколько угодно false alarm. Проценты выше ста тут не баг.

    Отрицательные слагаемые или неположительное total_speech_s — ValueError.
    """
    if min(false_alarm_s, miss_s, confusion_s) < 0:
        raise ValueError("время ошибки не может быть отрицательным")
    if total_speech_s <= 0:
        raise ValueError("общее время речи должно быть положительным")
    return (false_alarm_s + miss_s + confusion_s) / total_speech_s


def frechet_distance_1d(real, generated):
    """FAD на одном измерении: расстояние Фреше между двумя нормальными выборками.

    frechet_distance_1d([1, 2, 3], [1, 2, 3])  ->  0.0
    frechet_distance_1d([0, 0, 2, 2], [3, 3, 5, 5])  ->  9.0

    Формула: (m1 - m2)^2 + v1 + v2 - 2*sqrt(v1*v2), где m — среднее, v —
    дисперсия. Настоящий FAD считает то же самое по эмбеддингам VGGish, где
    вместо дисперсии матрица ковариаций, а вместо корня — матричный корень.

    Смысл: сравниваются РАСПРЕДЕЛЕНИЯ, а не пары записей. Модель, у которой
    среднее сошлось, а разброс вдвое уже, всё равно получит штраф — она
    звучит однообразно. Меньше — лучше.

    Ловушка: FAD зависит от референсной выборки, поэтому сравнивать модели
    между собой можно только на одной и той же.

    Выборка короче двух элементов — ValueError.
    """
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("для оценки распределения нужно хотя бы два значения")
    m1 = sum(real) / len(real)
    m2 = sum(generated) / len(generated)
    v1 = sum((x - m1) ** 2 for x in real) / len(real)
    v2 = sum((x - m2) ** 2 for x in generated) / len(generated)
    return (m1 - m2) ** 2 + v1 + v2 - 2 * math.sqrt(v1 * v2)
