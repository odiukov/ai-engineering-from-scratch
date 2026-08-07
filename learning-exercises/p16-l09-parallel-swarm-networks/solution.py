"""
Параллельные и роевые архитектуры — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Три топологии связей из урока. mesh — все со всеми (полносвязный граф),
# star — все через одного (супервизор), ring — кольцо.
TOPOLOGIES = ("mesh", "star", "ring")


def build_topology(n, kind):
    """Рёбра графа связей между n агентами. Ребро — пара (i, j), i < j.

    Результат отсортирован, дублей нет. Агенты нумеруются 0..n-1,
    в звезде центр — агент 0.

    build_topology(3, "mesh")  ->  [(0, 1), (0, 2), (1, 2)]
    build_topology(3, "star")  ->  [(0, 1), (0, 2)]
    build_topology(3, "ring")  ->  [(0, 1), (0, 2), (1, 2)]

    Ловушка в кольце: ребро (n-1, 0) надо нормализовать в (0, n-1), иначе
    оно не совпадёт по форме с остальными и посчитается дважды. И при n=2
    кольцо вырождается в одно ребро, а не в два.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if kind not in TOPOLOGIES:
        raise ValueError(f"unknown topology {kind!r}")
    if kind == "mesh":
        edges = {(i, j) for i in range(n) for j in range(i + 1, n)}
    elif kind == "star":
        edges = {(0, j) for j in range(1, n)}
    else:
        # set плюс нормализация пары: при n=2 оба ребра кольца — это (0, 1)
        edges = {tuple(sorted((i, (i + 1) % n))) for i in range(n)} if n > 1 else set()
    return sorted(edges)


def channel_count(n, kind):
    """Сколько каналов связи нужно топологии. Это цена координации.

    channel_count(10, "mesh")  ->  45    (это 10*9/2 — рост как n^2)
    channel_count(10, "star")  ->  9     (это n-1 — рост линейный)

    Ровно этот разрыв и есть аргумент урока: супервизор платит O(n), рой
    без центра платит O(n^2). На 100 агентах это 99 каналов против 4950.
    """
    return len(build_topology(n, kind))


def is_connected(n, edges):
    """Связен ли граф: дойти можно от любого агента до любого.

    is_connected(3, [(0, 1), (0, 2)])  ->  True
    is_connected(3, [(0, 1)])          ->  False

    Несвязный граф в мультиагентной системе — это два роя, которые не знают
    друг о друге. Обычный обход в ширину от нулевого агента.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    neighbours = {i: [] for i in range(n)}
    for i, j in edges:
        neighbours[i].append(j)
        neighbours[j].append(i)
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for nxt in neighbours[node]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return len(seen) == n


def simulate_fixed(durations, assignment, n_workers):
    """Жёсткое назначение: задача i заранее закреплена за воркером assignment[i].

    Возвращает пару (makespan, counts): время до конца ПОСЛЕДНЕГО воркера и
    сколько задач сделал каждый.

    simulate_fixed([1.0, 1.0], [0, 1], 2)  ->  (1.0, {0: 1, 1: 1})
    simulate_fixed([1.0, 1.0], [0, 0], 2)  ->  (2.0, {0: 2, 1: 0})

    Последовательный прогон — это тот же вызов с одним воркером и нулями
    в assignment: отдельная функция не нужна.

    Ловушка: makespan это МАКСИМУМ по воркерам, а не сумма. Сумма — это
    работа, а не время; вся суть параллелизма в разнице между ними.
    """
    if len(durations) != len(assignment):
        raise ValueError("durations and assignment must have the same length")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    busy = {w: 0.0 for w in range(n_workers)}
    counts = {w: 0 for w in range(n_workers)}
    for duration, worker in zip(durations, assignment):
        if worker not in busy:
            raise ValueError(f"worker id {worker} out of range")
        busy[worker] += duration
        counts[worker] += 1
    return (max(busy.values()), counts)


def simulate_swarm(durations, n_workers):
    """Рой: свободный воркер забирает следующую задачу из общей очереди.

    Возвращает пару (makespan, counts) в том же формате, что simulate_fixed.
    Задачи разбираются в порядке списка, при ничьей берёт воркер с меньшим id.

    simulate_swarm([1.0, 1.0], 2)             ->  (1.0, {0: 1, 1: 1})
    simulate_swarm([4.0, 1.0, 1.0, 1.0], 2)   ->  (4.0, {0: 1, 1: 3})

    Второй пример — вся идея урока в трёх числах: воркер 1 сделал три задачи,
    пока воркер 0 возился с одной длинной. Никакого планировщика для этого
    не понадобилось.

    Ловушка: «свободный» это МИНИМАЛЬНОЕ время освобождения, а не круговой
    обход. Round-robin по воркерам даст ровные counts и завышенный makespan.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    free_at = [0.0] * n_workers
    counts = {w: 0 for w in range(n_workers)}
    for duration in durations:
        # index(min(...)) даёт первый минимум — это и есть «меньший id при ничьей»
        worker = free_at.index(min(free_at))
        free_at[worker] += duration
        counts[worker] += 1
    return (max(free_at), counts)


def speedup(baseline, candidate):
    """Во сколько раз candidate быстрее baseline.

    speedup(2.0, 0.5)  ->  4.0
    speedup(2.0, 2.0)  ->  1.0

    Ловушка: candidate == 0 это не «бесконечное ускорение», а испорченный
    замер. ValueError, иначе inf поедет дальше по отчёту.
    """
    if candidate <= 0:
        raise ValueError("candidate time must be positive")
    if baseline < 0:
        raise ValueError("baseline time must not be negative")
    return baseline / candidate


def hot_spot_ratio(counts):
    """Перекос нагрузки: во сколько раз самый загруженный воркер обогнал самого свободного.

    hot_spot_ratio({0: 2, 1: 2})  ->  1.0
    hot_spot_ratio({0: 5, 1: 1})  ->  5.0
    hot_spot_ratio({0: 5, 1: 0})  ->  inf

    Простаивающий воркер даёт бесконечность — это не баг, а самый громкий
    сигнал о hot-spotting из всех возможных.
    """
    if not counts:
        raise ValueError("counts must not be empty")
    values = list(counts.values())
    low = min(values)
    if low == 0:
        return float("inf")
    return max(values) / low


def aging_order(tasks, now, aging):
    """Порядок разбора очереди с учётом старения: сначала наибольший приоритет.

    Задача — dict с ключами "id", "priority", "arrival". Эффективный
    приоритет: priority + aging * (now - arrival). Ничья — кто раньше пришёл,
    затем по id.

    tasks = [{"id": "a", "priority": 1, "arrival": 0},
             {"id": "b", "priority": 5, "arrival": 9}]
    aging_order(tasks, 10, 0.0)  ->  ['b', 'a']
    aging_order(tasks, 10, 0.5)  ->  ['a', 'b']

    Без старения (aging=0) низкоприоритетная задача не дождётся своей
    очереди никогда — это starvation из урока. Старение — самое дешёвое
    лекарство: чем дольше ждёшь, тем выше лезешь.
    """
    if aging < 0:
        raise ValueError("aging must not be negative")
    ranked = sorted(
        tasks,
        key=lambda t: (
            -(t["priority"] + aging * (now - t["arrival"])),
            t["arrival"],
            t["id"],
        ),
    )
    return [t["id"] for t in ranked]
