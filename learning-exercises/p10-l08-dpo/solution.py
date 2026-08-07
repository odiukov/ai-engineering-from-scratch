"""
DPO: Direct Preference Optimization — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def sigmoid(x):
    """Сигмоида: сжимает любое число в интервал (0, 1).

    sigmoid(0.0)     ->  0.5
    sigmoid(-1000.0) ->  0.0   (без OverflowError)

    Ловушка: наивная 1/(1+exp(-x)) переполняется при большом отрицательном x.
    Разбери случай x < 0 отдельно через e^x / (1 + e^x).

    В DPO sigmoid стоит там же, где в RLHF: превращает разницу неявных наград
    в вероятность того, что chosen действительно лучше rejected.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def log_softmax(logits):
    """Логарифмы вероятностей softmax, посчитанные устойчиво.

    log_softmax([0.0, 0.0])     ->  [-0.6931..., -0.6931...]   (== log 0.5)
    log_softmax([1000.0, 0.0])  ->  [0.0, -1000.0]             (без OverflowError)

    Ловушка: log(softmax(x)) в лоб — это сначала exp (переполнение), потом
    log почти-нуля (потеря точности). Считай сразу:
        log_softmax(x)_i = x_i - max(x) - log(sum_j exp(x_j - max(x)))

    Сумма exp от результата равна 1, а сами значения всегда <= 0.
    """
    m = max(logits)
    shifted = [v - m for v in logits]
    log_sum = math.log(sum(math.exp(v) for v in shifted))
    return [v - log_sum for v in shifted]


def sequence_logprob(logits_per_step, target_ids):
    """Суммарный log-вероятность последовательности под моделью.

    logits_per_step — список логитов по шагам, target_ids — какой токен
    реально стоял на каждом шаге.

    sequence_logprob([[0.0, 0.0]], [0])              ->  -0.6931...
    sequence_logprob([[0.0, 0.0], [0.0, 0.0]], [0, 1])  ->  -1.3862...
    sequence_logprob([], [])                         ->  0.0

    Складываем ЛОГАРИФМЫ, а не перемножаем вероятности: произведение сотни
    чисел меньше единицы обнулится в double, а сумма логарифмов — нет.

    Это ровно та величина, которую DPO считает четыре раза на каждую пару:
    policy на chosen, policy на rejected, reference на chosen, reference
    на rejected.
    """
    total = 0.0
    for step_logits, target in zip(logits_per_step, target_ids):
        total += log_softmax(step_logits)[target]
    return total


def dpo_logit(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Аргумент сигмоиды в лоссе DPO — «насколько политика уже права».

    Формула: beta * ((pi_w - ref_w) - (pi_l - ref_l)), где все четыре
    аргумента — суммарные log-вероятности ответов.

    dpo_logit(-2.0, -3.0, -2.0, -3.0)        ->  0.0   (политика == reference)
    dpo_logit(-1.0, -4.0, -2.0, -3.0, 0.1)   ->  0.2

    Разности log pi - log pi_ref — это и есть неявная награда из статьи
    Rafailov et al.: отдельная reward model не нужна, награда уже лежит
    в отношении вероятностей двух моделей.

    Ловушка на порядке аргументов: chosen первым в каждой паре. Перепутаешь
    местами — знак logit'а перевернётся, и обучение поедет в другую сторону.
    """
    chosen_ratio = policy_chosen - ref_chosen
    rejected_ratio = policy_rejected - ref_rejected
    return beta * (chosen_ratio - rejected_ratio)


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Лосс DPO на одной паре предпочтений: -log(sigmoid(dpo_logit(...))).

    dpo_loss(-2.0, -3.0, -2.0, -3.0)        ->  0.6931...  (== log 2)
    dpo_loss(-1.0, -9.0, -2.0, -3.0, 0.1)   ->  0.4031...  (лучше, чем log 2)

    Пока политика совпадает с reference, лосс ровно log 2 при ЛЮБЫХ данных:
    стартовая точка обучения одинакова для всех пар.

    Это то, что считает trl.DPOTrainer внутри, только там четыре
    log-вероятности приходят из forward-пассов трансформера, а не аргументами.

    Ловушка: -log(sigmoid(z)) в лоб даёт ValueError при сильно отрицательном z
    (sigmoid обнулится). Считай через softplus: log(1 + e^(-z)).
    """
    z = dpo_logit(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    # та же softplus-форма, что и в Bradley-Terry: лосс конечен всегда,
    # переполниться может только промежуточная экспонента
    if z >= 0:
        return math.log1p(math.exp(-z))
    return -z + math.log1p(math.exp(z))


def dpo_grad(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Аналитический градиент dpo_loss по обеим log-вероятностям политики.

    Вернуть кортеж (dL/d policy_chosen, dL/d policy_rejected).

    dpo_grad(-2.0, -3.0, -2.0, -3.0)  ->  (-0.05, 0.05)   при beta = 0.1
    dpo_grad(-1.0, -9.0, -2.0, -3.0)  ->  примерно (-0.0332, 0.0332)

    Вывод: L = -log s, s = sigmoid(z), z = beta * (ratio_w - ratio_l).
    dL/dz = -(1 - s), dz/d pi_w = +beta, dz/d pi_l = -beta.

    Множитель (1 - s) — это «насколько модель ещё неправа». Когда пара уже
    решена (s близко к 1), градиент затухает сам: DPO перестаёт тратить шаги
    на то, что и так выучено.

    По reference-модели градиента нет вовсе — она заморожена.
    """
    z = dpo_logit(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    s = sigmoid(z)
    g = -beta * (1.0 - s)
    return (g, -g)


def implicit_rewards(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Неявные награды DPO и зазор между ними.

    Вернуть словарь с ключами "chosen", "rejected", "margin".

    implicit_rewards(-2.0, -3.0, -2.0, -3.0)       ->  chosen 0.0, margin 0.0
    implicit_rewards(-1.0, -4.0, -2.0, -3.0, 0.1)  ->  margin 0.2

    Награда ответа y — это beta * (log pi(y) - log pi_ref(y)). Никакой
    reward model не обучалось, но число ведёт себя как награда: положительное
    значит, что политика полюбила ответ сильнее, чем reference.

    Margin — главный индикатор здоровья обучения. Растёт по эпохам — DPO
    работает; стоит на месте — либо lr слишком мал, либо beta слишком велика.
    """
    chosen = beta * (policy_chosen - ref_chosen)
    rejected = beta * (policy_rejected - ref_rejected)
    return {"chosen": chosen, "rejected": rejected, "margin": chosen - rejected}


def preference_accuracy(rows, beta=0.1):
    """Доля пар, где неявная награда chosen выше, чем у rejected.

    rows — список кортежей (policy_chosen, policy_rejected, ref_chosen,
    ref_rejected).

    preference_accuracy([(-1.0, -4.0, -2.0, -3.0)])  ->  1.0
    preference_accuracy([(-4.0, -1.0, -3.0, -2.0)])  ->  0.0
    preference_accuracy([])                          ->  0.0

    До обучения политика равна reference, все margin нулевые и точность
    равна нулю — это нормальная стартовая точка, а не поломка.

    Обрати внимание: при любом beta > 0 ответ один и тот же. beta — это
    положительный множитель, знак margin он поменять не может.
    """
    if not rows:
        return 0.0
    correct = 0
    for policy_chosen, policy_rejected, ref_chosen, ref_rejected in rows:
        margin = implicit_rewards(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
        )["margin"]
        if margin > 0:
            correct += 1
    return correct / len(rows)
