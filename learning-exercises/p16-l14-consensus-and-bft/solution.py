"""
Консенсус и византийская отказоустойчивость — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import re

TRUE_ANSWER = "4.2"
BYZANTINE_LIE = "42"


def max_faulty(n):
    """Сколько византийцев выдерживает сеть из n узлов: f = (n - 1) // 3.

    max_faulty(4)   ->  1
    max_faulty(7)   ->  2
    max_faulty(3)   ->  0
    max_faulty(100) ->  33

    PBFT (Castro & Liskov, 1999) требует n >= 3f + 1. Разворачиваем это
    относительно f: наибольшее целое f, при котором n >= 3f + 1, равно
    (n - 1) // 3. Отсюда знаменитое «одна треть минус эпсилон».

    n < 1 — не сеть, это ValueError.
    """
    if n < 1:
        raise ValueError("узлов должно быть хотя бы один")
    return (n - 1) // 3


def quorum_size(n):
    """Сколько одинаковых голосов нужно, чтобы решение считалось принятым.

    quorum_size(4)  ->  3
    quorum_size(5)  ->  4
    quorum_size(7)  ->  5

    Не «2f + 1» механически: эта формула верна только когда n ровно 3f + 1.
    Общее условие — любые два кворума обязаны пересекаться хотя бы по одному
    ЧЕСТНОМУ узлу, иначе две группы примут разные решения:

        2 * |Q| - n > f   =>   |Q| > (n + f) / 2

    Отсюда |Q| = (n + f) // 2 + 1. Проверь на n = 5: f = 1, и 2f + 1 = 3
    уже мало (два кворума по 3 пересекаются по одному узлу, а он может
    оказаться предателем), нужно 4.
    """
    return (n + max_faulty(n)) // 2 + 1


def canonicalize(answer):
    """Семантическая канонизация ответа: разные слова, один смысл — один ключ.

    canonicalize("the study reports 4.2% improvement")  ->  "4.2"
    canonicalize("4.2% gain")                           ->  "4.2"
    canonicalize("42.0")                                ->  "42"
    canonicalize("  Yes,  DEFINITELY ")                 ->  "yes, definitely"

    Голосовать по сырым строкам нельзя: «4.2% improvement» и «the study
    reports 4.2%» — один и тот же ответ, а строковое равенство разведёт их по
    разным кластерам и развалит любое большинство.

    Правило: если в ответе есть число — оно и есть ответ (округляем до двух
    знаков, чтобы 42 и 42.0 попали в один кластер). Если числа нет —
    нижний регистр и схлопнутые пробелы. В продакшене вместо этого дешёвая
    эмбеддинг-модель, но принцип тот же: сначала канонизация, потом счёт.
    """
    m = re.search(r"-?\d+(?:\.\d+)?", answer)
    if m:
        # :g убирает хвостовой ноль, иначе "42.0" и "42" разъедутся
        return f"{round(float(m.group()), 2):g}"
    return " ".join(answer.lower().split())


def cluster_votes(votes):
    """Сгруппировать голоса по канонической форме ответа.

    votes — список пар (answer, confidence).

    cluster_votes([("4.2%", 0.9), ("4.2 percent", 0.8), ("42", 0.7)])
      ->  {"4.2": [("4.2%", 0.9), ("4.2 percent", 0.8)], "42": [("42", 0.7)]}

    Порядок ключей — порядок первого появления кластера: dict в Python его
    сохраняет, и это делает разбор голосования воспроизводимым.
    """
    clusters = {}
    for answer, confidence in votes:
        clusters.setdefault(canonicalize(answer), []).append((answer, confidence))
    return clusters


def plurality(votes):
    """Классическое большинство: побеждает самый многочисленный кластер.

    plurality([("4.2", 0.9), ("42", 0.1), ("42", 0.1)])  ->  "42"
    plurality([])                                        ->  None

    Уверенность игнорируется целиком — в этом и слабость. Три агента на одной
    базовой модели, ошибающиеся одинаково (монокультура), перевешивают двух
    независимых и уверенных.

    Ничья разрешается лексикографически меньшим ключом: агрегатор обязан быть
    детерминированным, иначе один и тот же вход даст разные протоколы.
    """
    clusters = cluster_votes(votes)
    if not clusters:
        return None
    return min(clusters, key=lambda k: (-len(clusters[k]), k))


def weighted_consensus(votes, threshold=0.5):
    """CP-WBFT: вес кластера — сумма уверенностей, а не число голосов.

    weighted_consensus([("4.2", 0.9), ("4.2", 0.9), ("42", 0.5),
                        ("42", 0.5), ("42", 0.5)])            ->  "4.2"
    weighted_consensus([("4.2", 0.5), ("42", 0.5)])           ->  None

    Победитель принимается, только если его вес СТРОГО больше
    threshold * (сумма всех уверенностей). Не набрал — None, и это не сбой, а
    штатная эскалация: слабое большинство лучше не принимать вовсе.

    Осторожно с нулевой суммой уверенностей: делить не на что, решения нет.
    Разумный диапазон threshold — 0.5..0.67 для 5-7 голосующих.
    """
    clusters = cluster_votes(votes)
    total = sum(c for _, c in votes)
    if not clusters or total <= 0:
        return None
    weights = {k: sum(c for _, c in group) for k, group in clusters.items()}
    best = min(weights, key=lambda k: (-weights[k], k))
    return best if weights[best] > threshold * total else None


def geometric_median(values, iterations=64, tol=1e-9):
    """Геометрическая медиана набора чисел (алгоритм Вейсфельда).

    geometric_median([1.0, 2.0, 3.0, 4.0, 100.0])  ->  примерно 3.0
    среднее того же набора                          ->  22.0

    Точка, минимизирующая сумму расстояний до выборки. В отличие от среднего,
    один выброс её почти не двигает — на этом и построен DecentLLMs: ответ
    тянется к плотному скоплению честных предложений, а не к их среднему.

    Итерация: x <- sum(p_i / d_i) / sum(1 / d_i), где d_i = |x - p_i|.

    Ловушка: как только оценка совпадает с одной из точек, d_i = 0 и вся
    формула падает с ZeroDivisionError. Подстрахуй знаменатель снизу.
    """
    if not values:
        raise ValueError("выборка пуста")
    x = sum(values) / len(values)  # старт со среднего: сойдётся быстрее
    for _ in range(iterations):
        num = 0.0
        den = 0.0
        for p in values:
            d = abs(x - p)
            if d < tol:
                d = tol  # точка под оценкой: не даём знаменателю обнулиться
            num += p / d
            den += 1.0 / d
        nxt = num / den
        if abs(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def simulate_bft(n, f, rng, trials=200, honest_noise=0.0):
    """Прогнать раунд BFT в сети из n узлов с f скоординированными предателями.

    Вернуть доли исходов: {"correct": ..., "wrong": ..., "no_decision": ...}.

    simulate_bft(4, 1, random.Random(0))  ->  {"correct": 1.0, ...}
    simulate_bft(4, 2, random.Random(0))  ->  {"no_decision": 1.0, ...}

    Худший случай для протокола: все f предателей выдают ОДНУ И ТУ ЖЕ ложь
    BYZANTINE_LIE, то есть действуют согласованно. Честные отвечают
    TRUE_ANSWER, но с вероятностью honest_noise выдают собственную выдумку —
    так ведёт себя стохастический LLM даже без злого умысла.

    Решение принято, если какой-то кластер набрал quorum_size(n) голосов.

    Смотри на «wrong» отдельно от «no_decision»: при f + 1 предателях
    протокол не начинает врать, он перестаёт принимать решения. Safety
    держится, ломается liveness — и это ровно то, что обещает PBFT.
    """
    if not 0 <= f <= n:
        raise ValueError("предателей не может быть больше, чем узлов")
    need = quorum_size(n)
    counts = {"correct": 0, "wrong": 0, "no_decision": 0}
    for _ in range(trials):
        votes = [(BYZANTINE_LIE, 0.95) for _ in range(f)]
        for _ in range(n - f):
            if rng.random() < honest_noise:
                votes.append((f"noise-{rng.randrange(10 ** 6)}", 0.4))
            else:
                votes.append((TRUE_ANSWER, 0.8))
        clusters = cluster_votes(votes)
        winner = max(
            (k for k in clusters if len(clusters[k]) >= need),
            key=lambda k: (len(clusters[k]), k),
            default=None,
        )
        if winner is None:
            counts["no_decision"] += 1
        elif winner == canonicalize(TRUE_ANSWER):
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return {k: v / trials for k, v in counts.items()}
