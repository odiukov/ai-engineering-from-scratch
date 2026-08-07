"""
Модели мира и видео-диффузия — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def token_grid(shape, patch):
    """Размер сетки токенов после 3D-патчификации видео.

    shape — (T, H, W) в пикселях/кадрах, patch — (patch_t, patch_h, patch_w).
    Возвращается (T // patch_t, H // patch_h, W // patch_w).

    token_grid((8, 16, 16), (2, 2, 2))    ->  (4, 8, 8)
    token_grid((120, 360, 640), (2, 8, 8)) ->  (60, 45, 80)

    Если хоть одна сторона не делится нацело — ValueError. Молчаливое
    округление вниз отрежет последние кадры видео, и модель будет учиться на
    данных, которые не совпадают с тем, что ей показали.

    Аналог: nn.Conv3d(C, dim, kernel_size=patch, stride=patch) — свёртка с
    шагом в размер ядра и есть пространственно-временная патчификация.
    """
    if len(shape) != 3 or len(patch) != 3:
        raise ValueError("shape and patch must both be (T, H, W) triples")
    if any(p < 1 for p in patch):
        raise ValueError("patch sizes must be positive")
    if any(s % p for s, p in zip(shape, patch)):
        raise ValueError("every side must be divisible by its patch size")
    return tuple(s // p for s, p in zip(shape, patch))


def token_count(shape, patch):
    """Сколько токенов даст видео после патчификации: T' * H' * W'.

    token_count((8, 16, 16), (2, 2, 2))     ->  256
    token_count((120, 360, 640), (2, 8, 8)) ->  216000

    Второе число — 5 секунд 360p при 24 кадрах в секунду. 216 тысяч токенов:
    именно поэтому полный joint attention по видео не считают вообще никогда.

    Зачем: это первое число, которое считают, прежде чем оценивать память.
    """
    grid = token_grid(shape, patch)
    return grid[0] * grid[1] * grid[2]


def flat_index(grid, t, h, w):
    """Номер токена с координатами (t, h, w) в плоской последовательности.

    Порядок обхода — тот же, что даёт reshape после Conv3d: сначала время,
    внутри кадра — строки, внутри строки — столбцы.

    flat_index((4, 8, 8), 0, 0, 0)  ->  0
    flat_index((4, 8, 8), 0, 1, 0)  ->  8
    flat_index((4, 8, 8), 1, 0, 0)  ->  64

    Выход за границы сетки — ValueError, а не отрицательный индекс. В Python
    tokens[-1] это валидное обращение к последнему элементу, и ошибка
    координат выльется не в исключение, а в тихо перепутанные токены.

    Зачем: divided attention группирует токены именно по этим координатам.
    """
    grid_t, grid_h, grid_w = grid
    if not (0 <= t < grid_t and 0 <= h < grid_h and 0 <= w < grid_w):
        raise ValueError("coordinates are outside the token grid")
    return (t * grid_h + h) * grid_w + w


def divided_attention_groups(grid, axis):
    """Разбить токены на группы для divided attention.

    axis="time"  — группа на каждую пространственную позицию (h, w), внутри
                   неё токены всех кадров: T штук, групп H * W.
    axis="space" — группа на каждый кадр, внутри неё все H * W токенов кадра.

    divided_attention_groups((2, 1, 2), "time")   ->  [[0, 2], [1, 3]]
    divided_attention_groups((2, 1, 2), "space")  ->  [[0, 1], [2, 3]]

    Любое другое значение axis — ValueError.

    Ловушка: временная группа берёт ОДНУ И ТУ ЖЕ точку кадра на разных
    кадрах, а не соседние токены подряд. Соседние подряд — это уже
    пространственная группа. Перепутаешь — attention будет смотреть не туда,
    а тесты формы этого не заметят.

    Зачем: два прохода по T^2 и (H*W)^2 вместо одного по (T*H*W)^2. Это
    и есть TimeSformer, и на нём стоит каждый видео-DiT 2026 года.
    """
    grid_t, grid_h, grid_w = grid
    if axis == "time":
        return [
            [flat_index(grid, t, h, w) for t in range(grid_t)]
            for h in range(grid_h)
            for w in range(grid_w)
        ]
    if axis == "space":
        return [
            [flat_index(grid, t, h, w) for h in range(grid_h) for w in range(grid_w)]
            for t in range(grid_t)
        ]
    raise ValueError("axis must be 'time' or 'space'")


def attention_pairs(grid, mode):
    """Сколько пар «запрос-ключ» посчитает attention на такой сетке.

    mode="joint"   — все токены смотрят на все: (T * H * W) ** 2.
    mode="divided" — время плюс пространство: H*W * T**2 + T * (H*W)**2.

    attention_pairs((4, 2, 2), "joint")    ->  256
    attention_pairs((4, 2, 2), "divided")  ->  128

    Другой mode — ValueError.

    Ловушка: divided НЕ всегда дешевле. При T = 1 он добавляет бессмысленный
    проход по времени и проигрывает joint. Выигрыш появляется, когда велики
    ОБЕ оси, и на реальном видео он в тысячи раз.

    Зачем: это ровно то число, от которого зависит, влезет ли модель в память.
    """
    grid_t, grid_h, grid_w = grid
    spatial = grid_h * grid_w
    if mode == "joint":
        return (grid_t * spatial) ** 2
    if mode == "divided":
        # первое слагаемое — по одной attention на каждую точку кадра вдоль
        # времени, второе — по одной на каждый кадр внутри пространства
        return spatial * grid_t ** 2 + grid_t * spatial ** 2
    raise ValueError("mode must be 'joint' or 'divided'")


def axis_position_encoding(grid, t_dim, h_dim, w_dim):
    """3D-позиционное кодирование: свои синусоиды по t, h и w, склеенные подряд.

    Для каждого токена в плоском порядке строится вектор длины
    t_dim + h_dim + w_dim. По каждой оси половина каналов — синусы, половина —
    косинусы с частотами exp(-log(10000) * k / half), где half = dim // 2.

    axis_position_encoding((1, 1, 1), 2, 2, 2)  ->  [[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]]

    Позиция 0 всегда кодируется нулями в синусах и единицами в косинусах.

    Нечётная или нулевая длина по любой оси — ValueError: синусы и косинусы
    обязаны разбиться на пары поровну.

    Ловушка: частоты считаются от half = dim // 2, а НЕ от dim. Если делить
    на dim, спектр сожмётся вдвое, соседние позиции станут неразличимы, и
    модель перестанет отличать кадр t от кадра t+1.

    Зачем: у видео три независимые координаты, и модель обязана знать все
    три. Реальный RoPE не складывает, а поворачивает пары каналов, но
    информация в нём та же.
    """
    dims = (t_dim, h_dim, w_dim)
    if any(d < 2 or d % 2 for d in dims):
        raise ValueError("every axis dimension must be even and at least 2")

    def encode(position, dim):
        half = dim // 2
        freqs = [math.exp(-math.log(10000.0) * k / half) for k in range(half)]
        # сначала все синусы, потом все косинусы — тот же порядок, что в
        # torch.cat([sin, cos], dim=-1) из урока
        return [math.sin(position * f) for f in freqs] + [
            math.cos(position * f) for f in freqs
        ]

    grid_t, grid_h, grid_w = grid
    # кодировки по каждой оси считаются один раз на позицию, а не на токен:
    # позиций T + H + W, а токенов T * H * W
    t_table = [encode(t, t_dim) for t in range(grid_t)]
    h_table = [encode(h, h_dim) for h in range(grid_h)]
    w_table = [encode(w, w_dim) for w in range(grid_w)]

    return [
        t_table[t] + h_table[h] + w_table[w]
        for t in range(grid_t)
        for h in range(grid_h)
        for w in range(grid_w)
    ]


def inverse_dynamics(state, next_state):
    """Обратная динамика: какое действие переводит state в next_state.

    Состояние — список чисел (координаты, углы, что угодно). Действие —
    покоординатная разность.

    inverse_dynamics([0.0, 0.0], [1.0, -2.0])  ->  [1.0, -2.0]
    inverse_dynamics([3.0], [3.0])             ->  [0.0]

    Разная длина — ValueError.

    Ловушка: направление. Это next_state - state, «куда надо попасть минус
    где мы сейчас». Обратный знак даст робота, который едет от цели.

    Зачем: последнее звено роботической петли. VLM ставит цель, видео-модель
    воображает, как выглядит выполнение, а inverse dynamics превращает
    воображённые кадры в моторные команды.
    """
    if len(state) != len(next_state):
        raise ValueError("state and next_state must have the same length")
    return [b - a for a, b in zip(state, next_state)]


def imagine_rollout(state, actions, dynamics):
    """Воображаемый прогон: применить действия по очереди, вернуть все состояния.

    dynamics(state, action) выдаёт следующее состояние. Возвращается список
    длиной len(actions) — состояния ПОСЛЕ каждого действия, без стартового.

    imagine_rollout([0.0], [[1.0], [1.0]], lambda s, a: [s[0] + a[0]])
        ->  [[1.0], [2.0]]
    imagine_rollout([0.0], [], lambda s, a: s)  ->  []

    Стартовое состояние не мутируется: модель мира часто гоняют по нескольким
    веткам действий из одной точки, и общий изменяемый стейт склеит ветки.

    Ключевое свойство: inverse_dynamics, применённая к соседним состояниям
    прогона, обязана вернуть ровно те действия, которые в него подали.

    Зачем: это и есть «модель мира как симулятор» — прокрутить план в голове,
    прежде чем шевелить моторами.
    """
    current = list(state)
    states = []
    for action in actions:
        current = list(dynamics(current, action))
        states.append(current)
    return states
