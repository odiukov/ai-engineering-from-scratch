"""
Mixture of Experts: разреженный FFN — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def router_scores(hidden, W_router):
    """Оценки роутера: насколько каждый эксперт подходит этому токену.

    W_router — матрица (E, d_model), по строке на эксперта. Результат —
    список из E чисел.

    router_scores([1.0, 2.0], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        ->  [1.0, 2.0, 3.0]

    Роутер — это один линейный слой, никакой отдельной сети. Скор эксперта
    e — просто скалярное произведение скрытого состояния на его строку.
    Считается это за копейки, а решает, какие миллиарды параметров вообще
    проснутся на этом токене.

    Ширина строк W_router обязана совпадать с длиной hidden — иначе ValueError.
    """
    if any(len(row) != len(hidden) for row in W_router):
        raise ValueError("every router row must match the hidden width")
    return [sum(h * w for h, w in zip(hidden, row)) for row in W_router]


def select_experts(scores, bias, top_k):
    """Выбрать top_k экспертов по сумме scores + bias. Вернуть их индексы.

    select_experts([1.0, 5.0, 3.0], [0.0, 0.0, 0.0], 2)  ->  [1, 2]
    select_experts([1.0, 5.0, 3.0], [9.0, 0.0, 0.0], 2)  ->  [0, 1]

    Индексы идут по убыванию суммы scores + bias; при равенстве побеждает
    меньший индекс, иначе результат зависел бы от реализации сортировки.

    Здесь и живёт трюк DeepSeek-V3 (auxiliary-loss-free balancing): bias
    участвует ТОЛЬКО в выборе. На вес эксперта в смеси он не влияет — это
    работа gate_weights, которая биас в глаза не видела. Так балансировка
    нагрузки не искажает предсказания модели и не требует второго
    градиентного сигнала.

    top_k вне [1, len(scores)] — ValueError, как и bias другой длины.
    """
    if len(bias) != len(scores):
        raise ValueError("bias must have one entry per expert")
    if not 1 <= top_k <= len(scores):
        raise ValueError("top_k must be between 1 and the number of experts")
    biased = [s + b for s, b in zip(scores, bias)]
    order = sorted(range(len(biased)), key=lambda i: (-biased[i], i))
    return order[:top_k]


def gate_weights(scores, indices):
    """Веса смеси для выбранных экспертов: softmax по их СЫРЫМ скорам.

    gate_weights([1.0, 5.0, 3.0], [1, 2])  ->  [0.880..., 0.119...]
    gate_weights([2.0, 2.0], [0, 1])       ->  [0.5, 0.5]

    Softmax берётся только по выбранным экспертам, поэтому веса суммируются
    в единицу и выход слоя — выпуклая комбинация их выходов.

    Обрати внимание: функция принимает scores, а не scores + bias. Эксперт,
    попавший в top-k только благодаря биасу, получит маленький вес — ровно
    такой, какой заслужил его сырой скор. Балансировка меняет, кто считает,
    а не то, что получится.

    Пустой список индексов — ValueError: гейтить нечего.
    """
    if not indices:
        raise ValueError("nothing selected to gate")
    chosen = [scores[i] for i in indices]
    m = max(chosen)  # вычитание максимума: exp большого скора переполняется
    exps = [math.exp(c - m) for c in chosen]
    total = sum(exps)
    return [e / total for e in exps]


def apply_expert(x, W):
    """Прогнать токен через одного эксперта: линейный слой и SiLU.

    W — матрица (d_model, d_hidden). Результат — вектор длиной d_hidden.

    apply_expert([1.0], [[0.0, 2.0]])  ->  [0.0, 1.761...]

    SiLU (она же swish): silu(v) = v / (1 + e^(-v)). Именно она стоит в
    SwiGLU-экспертах современных MoE. Настоящий эксперт — это три матрицы,
    здесь одна: считаем не архитектуру эксперта, а маршрутизацию.

    Ловушка: при большом отрицательном v выражение e^(-v) улетает в
    OverflowError. Разбери случай v < 0 отдельно через v * e^v / (1 + e^v) —
    математически то же самое, численно безопасно.
    """
    if len(W) != len(x):
        raise ValueError("expert matrix must match the token width")
    d_hidden = len(W[0])
    out = [0.0] * d_hidden
    for value, row in zip(x, W):
        if value == 0.0:
            continue  # ноль ничего не добавляет
        for j in range(d_hidden):
            out[j] += value * row[j]
    result = []
    for v in out:
        if v >= 0:
            result.append(v / (1.0 + math.exp(-v)))
        else:
            e = math.exp(v)
            result.append(v * e / (1.0 + e))
    return result


def moe_forward(x, experts, W_router, top_k, bias):
    """Полный проход MoE-слоя для одного токена: (выход, индексы экспертов).

    experts — список матриц, по одной на эксперта.

    Порядок действий: скоры роутера, выбор top_k с учётом биаса, веса по
    сырым скорам, взвешенная сумма выходов ТОЛЬКО выбранных экспертов.

    moe_forward([1.0], [[[2.0]], [[3.0]]], [[1.0]], 1, [0.0, 0.0])
        ->  ([2.622...], [1])   (сработал только эксперт 1)

    Здесь и видно разреженность: сколько бы экспертов ни лежало в памяти,
    матричных умножений ровно top_k. Отсюда 671B параметров при 37B
    активных у DeepSeek-V3: память растёт с числом экспертов, а счёт — нет.
    """
    scores = router_scores(x, W_router)
    indices = select_experts(scores, bias, top_k)
    gates = gate_weights(scores, indices)
    d_hidden = len(experts[0][0])
    out = [0.0] * d_hidden
    for idx, gate in zip(indices, gates):
        # неактивных экспертов не считаем вообще — в этом весь смысл MoE
        h = apply_expert(x, experts[idx])
        for j in range(d_hidden):
            out[j] += gate * h[j]
    return out, indices


def expert_usage(tokens, W_router, top_k, bias):
    """Сколько токенов досталось каждому эксперту. Список длиной E.

    expert_usage([[1.0], [-1.0]], [[1.0], [-1.0]], 1, [0.0, 0.0])
        ->  [1, 1]

    Сумма счётчиков всегда равна len(tokens) * top_k: каждый токен
    выбирает ровно top_k экспертов.

    Это главный диагностический прибор MoE. Если роутер отправляет 90%
    токенов в третьего эксперта, остальные простаивают, а их параметры
    впустую занимают VRAM. Отсюда и растёт вся тема балансировки.

    Экспертов прогонять не нужно — достаточно роутера.
    """
    usage = [0] * len(W_router)
    for x in tokens:
        for idx in select_experts(router_scores(x, W_router), bias, top_k):
            usage[idx] += 1
    return usage


def update_bias(bias, usage, target, gamma):
    """Шаг auxiliary-loss-free балансировки: подтолкнуть биас на +-gamma.

    update_bias([0.0, 0.0], [10, 2], 6, 0.1)  ->  [-0.1, 0.1]
    update_bias([0.5], [6], 6, 0.1)           ->  [0.5]   (ровно в цель)

    Перегруженному эксперту биас снижаем, недогруженному повышаем, точно
    попавшему в цель не трогаем. target обычно равен
    len(tokens) * top_k / E — столько бы досталось каждому при идеально
    равномерной раздаче.

    Ключевое: обновление живёт ВНЕ функции потерь. Никакого штрафа в
    основном градиенте, никакого нового гиперпараметра в лоссе — только
    gamma. Это и есть находка DeepSeek-V3 2024 года: раньше на балансировку
    добавляли auxiliary loss, и он тянул модель в сторону от предсказаний.

    Старый список не портим: возвращаем новый.
    """
    new = []
    for b, used in zip(bias, usage):
        if used > target:
            new.append(b - gamma)
        elif used < target:
            new.append(b + gamma)
        else:
            new.append(b)
    return new


def moe_params(n_routed, expert_params, top_k, n_shared=0):
    """Параметры MoE-слоя: (всего, активных на токен).

    moe_params(8, 1000, 2)         ->  (8000, 2000)
    moe_params(256, 1000, 8, 1)    ->  (257000, 9000)

    Всего = (n_routed + n_shared) * expert_params — столько лежит в VRAM
    всегда, независимо от маршрутизации.
    Активных = (top_k + n_shared) * expert_params — столько считается на
    один токен.

    Shared expert проходит КАЖДЫЙ токен: он копит общее знание, а
    маршрутизируемые эксперты специализируются. Поэтому он всегда в активных.

    Отношение активных к общему — та самая sparsity: у DeepSeek-V3 это
    37B/671B, около 5.5%. Именно это и разрывает связку «больше знаний =
    больше вычислений на токен», на которой сидят плотные модели.
    """
    if top_k > n_routed:
        raise ValueError("cannot activate more experts than exist")
    total = (n_routed + n_shared) * expert_params
    active = (top_k + n_shared) * expert_params
    return total, active
