"""
Society of Mind и дебаты агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Позиция агента — одно число (его текущий ответ на задачу). Уверенность —
# вес этого числа в общей агрегации. Радиус доверия — насколько далёкую
# чужую позицию агент вообще готов слушать.
DEFAULT_TOL = 0.1
DEFAULT_RADIUS = float("inf")


def weighted_mean(values, weights):
    """Среднее значений с весами: sum(w*v) / sum(w).

    weighted_mean([10.0, 20.0], [1.0, 1.0])  ->  15.0
    weighted_mean([10.0, 20.0], [3.0, 1.0])  ->  12.5

    Ловушка: sum(weights) == 0 — это не «ноль в ответе», это отсутствие
    ответа. Брось ValueError, иначе получишь ZeroDivisionError в глубине
    цикла дебатов и будешь искать причину не там.

    Это агрегация позиций из Du et al. 2023: агент с большей уверенностью
    тянет консенсус сильнее.
    """
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        raise ValueError("weighted_mean of an empty list is undefined")
    total = sum(weights)
    if total == 0:
        raise ValueError("sum of weights must not be zero")
    return sum(w * v for w, v in zip(weights, values)) / total


def spread(answers):
    """Разброс мнений: max - min. Ноль означает полный консенсус.

    spread([1.0, 5.0, 3.0])  ->  4.0
    spread([7.0])            ->  0.0

    Это критерий сходимости дебатов. Пока spread велик, спорить есть о чём.
    """
    if not answers:
        raise ValueError("spread of an empty list is undefined")
    return max(answers) - min(answers)


def agreement_score(answers, tol=DEFAULT_TOL):
    """Доля агентов, чей ответ не дальше tol от среднего.

    agreement_score([5.0, 5.0, 5.0])        ->  1.0
    agreement_score([0.0, 5.0, 10.0], 1.0)  ->  0.3333333333333333

    Осторожно: 1.0 здесь значит «все согласны», а не «все правы». Урок
    отдельно подчёркивает: consensus не equals truth.
    """
    if not answers:
        raise ValueError("agreement_score of an empty list is undefined")
    mean = sum(answers) / len(answers)
    agree = sum(1 for a in answers if abs(a - mean) <= tol)
    return agree / len(answers)


def debate_round(answers, confidences, radius=DEFAULT_RADIUS, stubbornness=0.0):
    """Один раунд дебатов: каждый агент двигается к среднему тех, кого слышит.

    Агент i слышит агента j, только если |answers[j] - answers[i]| <= radius
    (себя он слышит всегда). Новый ответ:
        stubbornness * своё + (1 - stubbornness) * взвешенное среднее слышимых.

    debate_round([0.0, 10.0, 20.0], [1.0, 1.0, 1.0])      ->  [10.0, 10.0, 10.0]
    debate_round([0.0, 1.0, 10.0], [1.0, 1.0, 1.0], 2.0)  ->  [0.5, 0.5, 10.0]

    ГЛАВНАЯ ЛОВУШКА урока: раунд обязан быть ОДНОВРЕМЕННЫМ. Все читают
    позиции раунда r-1. Если писать новые ответы в тот же список, второй
    агент увидит уже обновлённого первого, и это уже не дебаты, а испорченный
    телефон — числа поедут, а тест это поймает.

    radius — это bounded confidence: агент не слушает того, кто слишком
    далеко. Именно из-за него поляризованная группа не сходится никогда.
    """
    if len(answers) != len(confidences):
        raise ValueError("answers and confidences must have the same length")
    if not 0.0 <= stubbornness <= 1.0:
        raise ValueError("stubbornness must be in [0, 1]")
    updated = []
    for own in answers:
        # индексы слышимых: сравниваем со СТАРЫМ списком, не с updated
        heard = [j for j, a in enumerate(answers) if abs(a - own) <= radius]
        mean = weighted_mean([answers[j] for j in heard], [confidences[j] for j in heard])
        updated.append(stubbornness * own + (1.0 - stubbornness) * mean)
    return updated


def run_debate(answers, confidences, rounds, radius=DEFAULT_RADIUS, stubbornness=0.0):
    """История дебатов: список позиций по раундам, включая нулевой.

    Длина результата — rounds + 1: нулевой раунд это исходные позиции.

    run_debate([0.0, 10.0], [1.0, 1.0], 1)  ->  [[0.0, 10.0], [5.0, 5.0]]
    run_debate([0.0, 10.0], [1.0, 1.0], 0)  ->  [[0.0, 10.0]]

    Ловушка: не порти входной список. История начинается с его КОПИИ.

    Du et al. советуют потолок в 3 раунда: дальше платишь N*R вызовов LLM
    за всё меньший сдвиг.
    """
    if rounds < 0:
        raise ValueError("rounds must not be negative")
    history = [list(answers)]
    current = list(answers)
    for _ in range(rounds):
        current = debate_round(current, confidences, radius, stubbornness)
        history.append(current)
    return history


def rounds_to_consensus(
    answers,
    confidences,
    tol=DEFAULT_TOL,
    radius=DEFAULT_RADIUS,
    stubbornness=0.0,
    max_rounds=50,
):
    """Сколько раундов нужно, чтобы spread упал до tol. None — не сошлись.

    rounds_to_consensus([5.0, 5.0], [1.0, 1.0])                    ->  0
    rounds_to_consensus([40.0, 42.0], [1.0, 1.0], 0.5, 1e9, 0.5)   ->  2

    Ноль означает «спорить было не о чем с самого начала».
    None означает «за max_rounds не сошлись» — это НЕ ошибка, а результат:
    при поляризации (два кластера дальше radius друг от друга) консенсуса
    не будет никогда, и честный ответ здесь — None, а не бесконечный цикл.
    """
    if max_rounds < 0:
        raise ValueError("max_rounds must not be negative")
    current = list(answers)
    if spread(current) <= tol:
        return 0
    for r in range(1, max_rounds + 1):
        current = debate_round(current, confidences, radius, stubbornness)
        if spread(current) <= tol:
            return r
    return None


def opinion_clusters(answers, tol=DEFAULT_TOL):
    """Разбиение агентов на кластеры мнений: индексы, сгруппированные по близости.

    Соседние (после сортировки по значению) агенты попадают в один кластер,
    если расстояние между ними не больше tol.

    opinion_clusters([1.0, 1.05, 9.0])  ->  [[0, 1], [2]]
    opinion_clusters([5.0, 5.0, 5.0])   ->  [[0, 1, 2]]

    Один кластер — консенсус. Два — поляризация: дебаты «закончились», но
    группа распалась на лагеря. Ровно этого и не видно по одному только
    agreement_score, если считать его от общего среднего.
    """
    if not answers:
        return []
    order = sorted(range(len(answers)), key=lambda i: answers[i])
    clusters = [[order[0]]]
    for prev, idx in zip(order, order[1:]):
        # порог считаем от ПРЕДЫДУЩЕГО соседа, а не от начала кластера:
        # цепочка близких точек — это один кластер, даже если её концы далеки
        if answers[idx] - answers[prev] <= tol:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    return [sorted(c) for c in clusters]


def sycophancy_collapse(answers, confidences):
    """Провал дебатов: все копируют ответ самого уверенного агента.

    sycophancy_collapse([1.0, 2.0, 3.0], [0.2, 0.9, 0.5])  ->  [2.0, 2.0, 2.0]
    sycophancy_collapse([1.0, 2.0], [0.5, 0.5])            ->  [1.0, 1.0]

    При равной уверенности побеждает первый — так же, как в реальном чате
    побеждает тот, кто высказался громче и раньше.

    Это sycophancy cascade из урока: spread падает в ноль за один раунд,
    agreement_score равен 1.0, и всё это ничего не говорит о правильности.
    """
    if len(answers) != len(confidences):
        raise ValueError("answers and confidences must have the same length")
    if not answers:
        raise ValueError("sycophancy_collapse of an empty list is undefined")
    loudest = max(range(len(answers)), key=lambda i: confidences[i])
    return [answers[loudest]] * len(answers)
