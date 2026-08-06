"""
ЭТАЛОН — урок 01. Открывай ПОСЛЕ того, как сам добился зелёных тестов.

Не единственно верный вариант, а один из хороших. Смысл — сравнить со своим:
где ты написал длиннее, где короче, где иначе. Расхождение не значит ошибку,
но повод разобраться почему.

Проверено: все 32 теста проходят.
Запуск проверки эталона (временно подменяет твой файл):
    cp exercise.py /tmp/mine.py && cp solution.py exercise.py \
      && python3 -m pytest -q; cp /tmp/mine.py exercise.py
"""

import math


def magnitude(v):
    # x*x читается быстрее и работает быстрее, чем x**2
    return sum(x * x for x in v) ** 0.5


def dot(a, b):
    # zip ставит компоненты в пары; один проход, без индексов
    return sum(x * y for x, y in zip(a, b))


def cosine_similarity(a, b):
    # переиспользуем уже написанное вместо копирования формул
    return dot(a, b) / (magnitude(a) * magnitude(b))


def angle_between(a, b):
    # подрезка обязательна: из-за округления косинус выходит 1.0000000002,
    # а acos от значения вне [-1, 1] бросает ValueError
    c = max(-1.0, min(1.0, cosine_similarity(a, b)))
    return math.degrees(math.acos(c))


def project(a, onto):
    # dot(onto, onto) — это |onto|^2, но без лишнего корня: корень бы всё
    # равно возвели обратно в квадрат
    k = dot(a, onto) / dot(onto, onto)
    return [k * x for x in onto]


def matvec(M, v):
    # внешний цикл по строкам, внутренний sum — это dot(row, v).
    # можно было написать [dot(row, v) for row in M] — короче и честнее,
    # раз dot уже есть
    return [sum(row[j] * v[j] for j in range(len(v))) for row in M]


def is_invertible_2x2(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    # сравнение с порогом, а не с нулём: у float определитель почти никогда
    # не бывает ровно 0.0
    return abs(det) > 1e-9


def most_similar_pair(vectors):
    # перебор всех пар: j начинается с i+1, поэтому каждая пара берётся один
    # раз и индексы уже отсортированы. O(n^2) — для десятков векторов норм,
    # для миллионов нужен ANN-индекс (FAISS и подобные), это тема Фазы 11.
    best, pair = None, None
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            s = cosine_similarity(vectors[i], vectors[j])
            if best is None or s > best:
                best, pair = s, (i, j)
    return pair
