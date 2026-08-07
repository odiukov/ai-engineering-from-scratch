"""
DualPipe: расписание пайплайна без пузырей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Модель, в которой мы работаем. Настоящий DualPipe гоняет чанки на 2048
H800, где один чанк — это attention, all-to-all dispatch, MLP экспертов и
all-to-all combine. Здесь GPU нет, поэтому пайплайн — это списки:

  * ОПЕРАЦИЯ  — кортеж (kind, micro_batch, direction):
      kind      "F" (forward) или "B" (backward),
      direction +1 — микробатч течёт от ранга 0 к рангу P-1,
                -1 — от ранга P-1 к рангу 0 (это и есть «bidirectional»);
  * ПОРЯДОК   — order[rank] = список операций этого ранга в том порядке,
                в котором он их выполнит. Само расписание (кто когда) из
                порядка выводится, а не задаётся руками;
  * СОБЫТИЕ   — кортеж (rank, step, kind, micro_batch, direction): что и на
                каком шаге реально запустилось;
  * ВРЕМЯ     — целый шаг. Любая операция стоит ровно один шаг. Это грубо
                (у настоящего backward цена вдвое выше forward), зато видно
                главное: где ранг простаивает.

«Пузырь» (bubble) — шаг, на котором ранг не делает ничего, потому что ждёт
вход от соседа или градиент сверху. Считаем пузыри как пустые слоты.

Чего здесь нет: реальных весов, реальных all-to-all и совмещения compute с
comm внутри одного чанка. Совмещение — это второй трюк DualPipe, он
работает ВНУТРИ слота; мы моделируем только первый трюк — двунаправленную
подачу микробатчей, из-за которой пустых слотов становится меньше.
"""


def gpipe_order(p, micro_batches):
    """Порядок операций каждого ранга по GPipe: сначала все F, потом все B.

    gpipe_order(2, 2)  ->  [[('F', 0, 1), ('F', 1, 1), ('B', 1, 1), ('B', 0, 1)],
                            [('F', 0, 1), ('F', 1, 1), ('B', 1, 1), ('B', 0, 1)]]

    Каждый ранг делает 2 * micro_batches операций. Backward идёт в обратном
    порядке микробатчей: последний вошёл — первый выходит.

    Направление у всех операций одно и то же: +1. GPipe односторонний.

    Это baseline: память под активации максимальная (в полёте сразу все
    micro_batches), пузырь такой же, как у 1F1B.
    """
    ops = [("F", i, 1) for i in range(micro_batches)]
    ops += [("B", i, 1) for i in range(micro_batches - 1, -1, -1)]
    # один и тот же порядок на всех рангах; копия на ранг, чтобы вызывающий
    # мог править списки, не задевая соседей
    return [list(ops) for _ in range(p)]


def one_f_one_b_order(p, micro_batches):
    """Порядок операций каждого ранга по 1F1B.

    one_f_one_b_order(2, 2)  ->  [[('F', 0, 1), ('F', 1, 1), ('B', 0, 1), ('B', 1, 1)],
                                  [('F', 0, 1), ('B', 0, 1), ('F', 1, 1), ('B', 1, 1)]]

    Рецепт: ранг r делает warmup из min(p - 1 - r, micro_batches) forward-ов,
    затем чередует F и B, затем добивает оставшиеся B.

    Смысл warmup: чем ближе ранг к началу пайплайна, тем больше микробатчей
    он должен запустить, прежде чем первый градиент вернётся к нему. Ранг
    p-1 (последний) чередует F и B с самого начала.

    Операций столько же, сколько у GPipe (2 * micro_batches), и пузырь тот
    же. Выигрыш 1F1B — в памяти: в полёте не micro_batches активаций, а p.
    """
    orders = []
    for r in range(p):
        warm = min(p - 1 - r, micro_batches)
        ops = [("F", i, 1) for i in range(warm)]
        pending = 0  # какой микробатч следующим уходит в backward
        for i in range(warm, micro_batches):
            ops.append(("F", i, 1))
            ops.append(("B", pending, 1))
            pending += 1
        # cooldown: forward-ы кончились, остались только backward-ы
        while pending < micro_batches:
            ops.append(("B", pending, 1))
            pending += 1
        orders.append(ops)
    return orders


def dualpipe_order(p, micro_batches):
    """Порядок операций по DualPipe: два встречных потока микробатчей.

    dualpipe_order(2, 2)  ->  [[('F', 0, 1), ('F', 1, -1), ('B', 0, 1), ('B', 1, -1)],
                               [('F', 0, 1), ('F', 1, -1), ('B', 0, 1), ('B', 1, -1)]]

    Микробатчи делятся пополам. Первая половина течёт как обычно (direction
    +1, вход на ранге 0). Вторая половина течёт навстречу (direction -1,
    вход на ранге p-1), и для неё ранг r играет роль ранга p-1-r.

    Дальше два потока чередуются слот в слот: A[0], B[0], A[1], B[1], ...
    Именно поэтому пузыри схлопываются — там, где прямой поток ждёт
    градиент, встречный поток даёт ранту работу.

    micro_batches обязан быть чётным: пополам не делится — ValueError.

    Цена: ранг r обслуживает и слой r, и слой p-1-r, то есть держит ДВЕ
    копии параметров. Это и есть «dual» в названии. DualPipeV (Sea AI Lab)
    убирает вторую копию ценой чуть большего пузыря.
    """
    if micro_batches % 2 != 0:
        raise ValueError("micro_batches должен быть чётным: потоки делятся пополам")
    half = micro_batches // 2
    forward_stream = one_f_one_b_order(p, half)
    orders = []
    for r in range(p):
        straight = forward_stream[r]
        # зеркало: встречный поток видит ранг r как ранг p-1-r, а номера
        # микробатчей сдвинуты на half, чтобы не пересекаться с прямыми
        reverse = [(kind, mb + half, -1) for kind, mb, _ in forward_stream[p - 1 - r]]
        merged = []
        for i in range(max(len(straight), len(reverse))):
            if i < len(straight):
                merged.append(straight[i])
            if i < len(reverse):
                merged.append(reverse[i])
        orders.append(merged)
    return orders


def simulate_pipeline(order):
    """Раскладывает очереди операций по шагам времени. Список событий.

    simulate_pipeline([[('F', 0, 1)], [('F', 0, 1)]])
        ->  [(0, 0, 'F', 0, 1), (1, 1, 'F', 0, 1)]

    order[rank] — это ОЧЕРЕДЬ ПРИОРИТЕТА, а не жёсткий порядок. На каждом
    шаге свободный ранг берёт первую операцию своей очереди, у которой уже
    готовы зависимости, и пропускает те, что ещё ждут. Реальный ранг ведёт
    себя так же: у него в работе оба потока микробатчей, и он запускает
    тот, чьи данные приехали.

    Зависимости:
      * F(rank, mb, d) ждёт F(rank - d, mb, d) — вход приходит от соседа
        со стороны, откуда течёт микробатч;
      * B(rank, mb, d) ждёт B(rank + d, mb, d) — градиент возвращается с
        другой стороны — и свой же F(rank, mb, d).

    Если соседа нет (край пайплайна), зависимости просто нет: это точка
    входа потока.

    Возвращает события, отсортированные по (шаг, ранг).

    Ловушка: брать строго головную операцию очереди нельзя — на этом
    двунаправленное расписание вырождается и работает ХУЖЕ 1F1B: прямой
    поток стоит и ждёт встречный вместо того, чтобы считать. Очередь, в
    которой зависимости зациклены, исполнить нельзя — ValueError.
    """
    p = len(order)
    start = {}                              # (rank, op) -> шаг запуска
    free = [0] * p                          # с какого шага ранг свободен
    waiting = [list(ops) for ops in order]  # ещё не запущенное, в приоритете
    total = sum(len(ops) for ops in order)
    events = []
    step = 0
    while len(events) < total:
        idle_everywhere = True
        for r in range(p):
            if free[r] > step:
                idle_everywhere = False
                continue
            for op in waiting[r]:
                kind, mb, d = op
                if kind == "F":
                    deps = [(r - d, op)] if 0 <= r - d < p else []
                else:
                    deps = [(r, ("F", mb, d))]
                    if 0 <= r + d < p:
                        deps.append((r + d, op))
                # операция стоит один шаг, поэтому зависимость отпускает нас
                # на следующем шаге после своего запуска
                if any(dep not in start or start[dep] + 1 > step for dep in deps):
                    continue
                start[(r, op)] = step
                events.append((r, step, kind, mb, d))
                free[r] = step + 1
                waiting[r].remove(op)
                idle_everywhere = False
                break
        if idle_everywhere and len(events) < total:
            raise ValueError("расписание неисполнимо: круговая зависимость")
        step += 1
    events.sort(key=lambda e: (e[1], e[0]))
    return events


def makespan(events):
    """Сколько шагов занял весь прогон: последний шаг плюс один.

    makespan([(0, 0, 'F', 0, 1), (1, 1, 'F', 0, 1)])  ->  2

    Шаги нумеруются с нуля, поэтому «последний шаг 1» означает две единицы
    времени. Пустой список — ноль шагов.
    """
    if not events:
        return 0
    return max(step for _, step, _, _, _ in events) + 1


def bubble_slots(events):
    """Сколько пустых шагов у каждого ранга. Список длины P.

    bubble_slots([(0, 0, 'F', 0, 1), (1, 1, 'F', 0, 1)])  ->  [1, 1]

    Ранг простаивает не только когда ждёт соседа, но и когда уже всё сделал,
    а остальные ещё нет: GPU всё это время куплен и стоит. Поэтому пузырь
    считается до общего makespan, а не до последней операции ранга.

    Использует makespan.
    """
    total_steps = makespan(events)
    p = max((rank for rank, _, _, _, _ in events), default=-1) + 1
    busy = [0] * p
    for rank, _, _, _, _ in events:
        busy[rank] += 1
    return [total_steps - b for b in busy]


def bubble_fraction(events):
    """Доля впустую купленного GPU-времени: пузыри / (P * makespan).

    bubble_fraction([(0, 0, 'F', 0, 1), (1, 1, 'F', 0, 1)])  ->  0.5

    Ноль — идеально плотное расписание. У 1F1B доля равна
    (P - 1) / (micro_batches + P - 1): растит micro_batches — падает, но
    медленно. Ради этой цифры DualPipe и придумали.

    Использует makespan и bubble_slots.
    """
    total_steps = makespan(events)
    slots = bubble_slots(events)
    if not slots or total_steps == 0:
        return 0.0
    return sum(slots) / (len(slots) * total_steps)


def peak_activation_memory(order):
    """Пик активаций в полёте на каждом ранге. Список длины P.

    peak_activation_memory(gpipe_order(2, 4))         ->  [4, 4]
    peak_activation_memory(one_f_one_b_order(2, 4))   ->  [2, 1]

    Микробатч «в полёте» с момента своего F до своего B: всё это время его
    активации лежат в памяти и удалить их нельзя. Считаем счётчиком: F даёт
    +1, B даёт -1, нужен максимум.

    Ровно за этим 1F1B и нужен. Пузырь у него такой же, как у GPipe, а вот
    пик активаций не micro_batches, а всего p — именно это позволяет
    поднять micro_batches, не упираясь в память.
    """
    peaks = []
    for ops in order:
        live = 0
        top = 0
        for kind, _, _ in ops:
            live += 1 if kind == "F" else -1
            top = max(top, live)
        peaks.append(top)
    return peaks
