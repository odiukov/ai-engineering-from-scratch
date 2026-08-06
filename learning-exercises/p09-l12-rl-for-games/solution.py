"""
RL для игр: AlphaZero, MuZero и GRPO — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Доска крестиков-нолики — строка из 9 символов, слева-направо и сверху-вниз:
"." пустая клетка, "x" и "o" — ходы. Например "xox.x...o" это

    x o x
    . x .
    . . o
"""

import math


def softmax(logits):
    """Превращает логиты в распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([0.0, 1000.0])    ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум логитов
    перед экспонентой — результат тот же, переполнения нет.

    Для GRPO это выходной слой LLM: распределение над следующим токеном.
    """
    # сдвиг на максимум: exp(z - m) <= 1, переполниться нечему
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def group_advantages(rewards):
    """Group-relative advantage GRPO: (r - mean) / (std + 1e-8).

    group_advantages([1.0, 0.0])        ->  [1.0, -1.0]
    group_advantages([1.0, 1.0, 1.0])   ->  [0.0, 0.0, 0.0]
    group_advantages([0.5])             ->  [0.0]

    Никакого критика: baseline — это среднее по группе из G сэмплов на один
    и тот же промпт, нормировка — стандартное отклонение той же группы.

    Ловушка: у группы, где ВСЕ ответы верные (или все неверные), std равен
    нулю. Прибавь 1e-8 к знаменателю — и такая группа честно даст нулевой
    сигнал вместо ZeroDivisionError. Это не баг, а свойство GRPO: если
    задача уже решена или ещё безнадёжна, учиться на ней нечему.
    """
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    # +1e-8 спасает группу с одинаковыми наградами: сигнал ноль, а не падение
    sd = math.sqrt(var) + 1e-8
    return [(r - mean) / sd for r in rewards]


def kl_penalty_gradient(probs, ref_probs):
    """Градиент KL(pi || pi_ref) по ЛОГИТАМ. Список той же длины, что probs.

    Формула: grad[i] = probs[i] * (log(probs[i]/ref[i]) - KL(pi || pi_ref)).

    kl_penalty_gradient([0.5, 0.5], [0.5, 0.5])  ->  [0.0, 0.0]

    Вычитание самой KL — не украшение: без него сумма компонент не равна
    нулю, а градиент по логитам softmax обязан суммироваться в ноль (сдвиг
    всех логитов на константу ничего не меняет).

    Ловушка: log(0). Подпирай ref_probs снизу маленькой константой.

    Это то, что RLHF и GRPO держат как поводок до pi_ref. Без него политика
    уедет туда, где reward model или verifier ошибаются.
    """
    kl = sum(
        p * (math.log(max(p, 1e-12)) - math.log(max(q, 1e-12)))
        for p, q in zip(probs, ref_probs)
    )
    return [
        p * (math.log(max(p, 1e-12)) - math.log(max(q, 1e-12)) - kl)
        for p, q in zip(probs, ref_probs)
    ]


def grpo_step(logits_row, samples, rewards, ref_probs=None, lr=0.1, beta=0.0):
    """Один шаг GRPO по группе сэмплов. Вернуть новый список логитов.

    samples — список выбранных индексов действий (по одному на сэмпл группы),
    rewards — награда verifier для каждого из них.

    Максимизируется   (1/G) * sum_k A_k * log pi(a_k)  -  beta * KL(pi || ref),
    поэтому шаг идёт ПО градиенту:
        new[i] = logits_row[i] + lr * grad[i]

    grpo_step([0.0, 0.0], [0, 1], [1.0, 0.0], lr=1.0)  ->  [0.5, -0.5]

    Знак и деление на G обязательны. Без 1/G размер шага растёт вместе с
    размером группы, и G=64 разнесёт политику при тех же настройках, что
    работали на G=8.

    Это `trl.GRPOTrainer` в миниатюре: ни критика, ни reward model, ни MCTS —
    только группа сэмплов, verifier и поводок KL.
    """
    probs = softmax(logits_row)
    advs = group_advantages(rewards)
    n = len(logits_row)
    grad = [0.0] * n

    for action, adv in zip(samples, advs):
        for i in range(n):
            # grad log pi(a) по логитам = onehot(a) - probs
            grad[i] += adv * ((1.0 if i == action else 0.0) - probs[i]) / len(samples)

    if beta and ref_probs is not None:
        pull = kl_penalty_gradient(probs, ref_probs)
        for i in range(n):
            # KL штрафуется, значит из максимизируемой цели она вычитается
            grad[i] -= beta * pull[i]

    return [logits_row[i] + lr * grad[i] for i in range(n)]


def winner(board):
    """Итог партии: "x", "o", "draw" или None, если игра ещё идёт.

    winner("xxx......")  ->  "x"
    winner("xxoxo.o..")  ->  "o"    (антидиагональ: o на 2, 4, 6)
    winner("xoxoxoox.")  ->  None   (есть пустая клетка, линий нет)
    winner("xxoooxxox")  ->  "draw"

    Восемь линий: три строки, три столбца, две диагонали.

    "draw" и None путать нельзя: None означает «продолжай искать ходы»,
    "draw" — «партия кончилась вничью». Для minimax это разные ветки.
    """
    lines = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # строки
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # столбцы
        (0, 4, 8), (2, 4, 6),             # диагонали
    )
    for a, b, c in lines:
        if board[a] != "." and board[a] == board[b] == board[c]:
            return board[a]
    return None if "." in board else "draw"


def minimax_value(board, player, memo=None):
    """Цена позиции при идеальной игре обеих сторон, с точки зрения "x".

    +1 — "x" выигрывает, -1 — выигрывает "o", 0 — ничья.
    player — чей ход сейчас ("x" или "o").

    minimax_value("xx.oo....", "x")  ->  1    (x ставит в 2 и выигрывает)
    minimax_value(".........", "x")  ->  0    (крестики-нолики — ничья)
    minimax_value("xxoo.....", "o")  -> -1    (o ходит в 5 и создаёт две угрозы)

    "x" МАКСИМИЗИРУЕТ, "o" МИНИМИЗИРУЕТ — обе стороны считаются одной и той
    же функцией, просто с разным экстремумом.

    Без memo обход занимает ~550k узлов и секунды; с memo по ключу
    (board, player) — около 5.5k узлов. Именно на этом месте AlphaZero
    заменяет полный обход на MCTS с обученным приором.
    """
    done = winner(board)
    if done is not None:
        return {"x": 1, "o": -1, "draw": 0}[done]

    if memo is None:
        memo = {}
    key = (board, player)
    if key in memo:
        return memo[key]

    other = "o" if player == "x" else "x"
    values = [
        minimax_value(board[:i] + player + board[i + 1:], other, memo)
        for i in range(9)
        if board[i] == "."
    ]
    value = max(values) if player == "x" else min(values)
    memo[key] = value
    return value


def best_move(board, player):
    """Индекс лучшего хода для player при идеальной игре. При равенстве — меньший.

    best_move("xx.oo....", "x")  ->  2   (немедленная победа)
    best_move("xx.o.....", "o")  ->  2   (блокировать, иначе x выиграет)
    best_move(".........", "x")  ->  0   (все ходы равны, берём первый)

    "x" выбирает ход с максимальной minimax_value, "o" — с минимальной.

    Полезно осознать: на первой доске в примере есть и другие ходы, но
    выигрышный ровно один, и minimax обязан выбрать именно его. Если
    функция вернула что-то другое — почти всегда перепутан экстремум для
    одной из сторон.
    """
    other = "o" if player == "x" else "x"
    memo = {}  # общий memo на все ветки: одинаковые позиции считаются один раз
    scored = [
        (i, minimax_value(board[:i] + player + board[i + 1:], other, memo))
        for i in range(9)
        if board[i] == "."
    ]
    pick = max if player == "x" else min
    # key только по значению: max/min при равенстве вернут первый, то есть
    # ход с меньшим индексом
    return pick(scored, key=lambda pair: pair[1])[0]


def puct_score(q, prior, parent_visits, child_visits, c=1.4):
    """PUCT из AlphaZero: Q + c * prior * sqrt(N) / (1 + n).

    q             — средняя оценка узла (эксплуатация),
    prior         — вероятность хода от policy-сети,
    parent_visits — N, сколько раз заходили в родителя,
    child_visits  — n, сколько раз заходили в этого ребёнка.

    puct_score(0.0, 0.5, 4, 0, c=1.0)   ->  1.0
    puct_score(0.5, 0.5, 4, 0, c=1.0)   ->  1.5
    puct_score(0.0, 0.5, 4, 3, c=1.0)   ->  0.25

    Единица в знаменателе — не косметика: без неё непосещённый ребёнок дал
    бы деление на ноль, а именно его и надо разведать первым.

    Приор от сети — это и есть весь смысл AlphaZero: сеть подсказывает, какие
    ветки вообще стоит разворачивать, и обход перестаёт быть экспоненциальным.
    """
    return q + c * prior * math.sqrt(parent_visits) / (1 + child_visits)
