"""
Proximal Policy Optimization (PPO) — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def action_probs(theta, x):
    """Политика линейного актора: логиты theta[a] . x, затем стабильный softmax.

    action_probs([[0.0], [0.0]], [1.0])        ->  [0.5, 0.5]
    action_probs([[1.0], [0.0]], [1.0])        ->  [0.731..., 0.268...]
    action_probs([[1.0], [0.0]], [1000.0])     ->  [1.0, 0.0]  (без OverflowError)

    theta — матрица [n_actions][n_features].

    Ловушка: math.exp от большого логита переполняется. Вычти максимум логитов
    перед экспонентой.

    Это `softmax(theta @ x)`, то есть `torch.nn.functional.softmax` над выходом
    одного линейного слоя.
    """
    logits = [sum(row[j] * x[j] for j in range(len(x))) for row in theta]
    # сдвиг на максимум: exp(z - m) <= 1, переполниться нечему
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def importance_ratio(log_p_new, log_p_old):
    """Importance ratio r_t = pi_new(a|s) / pi_old(a|s), посчитанный через логи.

    importance_ratio(-1.0, -1.0)   ->  1.0    (политика не изменилась)
    importance_ratio(0.0, -math.log(2))  ->  2.0

    Считать надо exp(log_new - log_old), а НЕ new/old. Вероятность длинного
    ответа LLM это произведение сотен множителей: в линейном виде она
    занулится, а разность логарифмов останется конечной.
    """
    return math.exp(log_p_new - log_p_old)


def clipped_surrogate(ratio, advantage, eps=0.2):
    """Клиппованный surrogate PPO: min(r*A, clip(r, 1-eps, 1+eps)*A).

    clipped_surrogate(1.0, 3.0)   ->  3.0    (ничего не изменилось)
    clipped_surrogate(2.0, 3.0)   ->  3.6    (A>0 упёрлось в потолок 1.2*A)
    clipped_surrogate(0.5, -3.0)  ->  -2.4   (A<0 упёрлось в пол 0.8*A)
    clipped_surrogate(2.0, -3.0)  ->  -6.0   (A<0, движение полезное — не режем)

    Тонкость, из-за которой берётся именно min, а не clip: коридор режет
    только ту сторону, где мы УЖЕ выиграли. Если политика ушла в невыгодную
    сторону, градиент проходит целиком — иначе алгоритм не смог бы
    исправить ошибку.
    """
    clipped = min(max(ratio, 1.0 - eps), 1.0 + eps)
    return min(ratio * advantage, clipped * advantage)


def surrogate_gradient_scale(ratio, advantage, eps=0.2):
    """Множитель перед grad log pi. Ноль там, где clip срезал градиент.

    surrogate_gradient_scale(1.0, 3.0)    ->  3.0
    surrogate_gradient_scale(2.0, 3.0)    ->  0.0    (A>0 и r>1+eps: обрезано)
    surrogate_gradient_scale(0.5, -3.0)   ->  0.0    (A<0 и r<1-eps: обрезано)
    surrogate_gradient_scale(2.0, -3.0)   ->  -6.0   (полезная сторона, режем не мы)

    Плоская крыша surrogate означает нулевую производную: если сэмпл попал за
    коридор с выгодной стороны, он НЕ должен двигать параметры. Иначе PPO
    вырождается в обычный policy gradient на многих эпохах и разъезжается.

    Вне коридора-обрезки множитель равен ratio * advantage — это производная
    r_t(theta) * A_t по логитам, поделённая на grad log pi.
    """
    if (advantage > 0 and ratio > 1.0 + eps) or (advantage < 0 and ratio < 1.0 - eps):
        return 0.0
    return ratio * advantage


def clip_fraction(ratios, advantages, eps=0.2):
    """Доля сэмплов, у которых clip сработал. Главная диагностика PPO.

    clip_fraction([1.0, 1.0], [1.0, 1.0])   ->  0.0
    clip_fraction([2.0, 1.0], [1.0, 1.0])   ->  0.5
    clip_fraction([], [])                   ->  0.0   (пустой батч, не ZeroDivisionError)

    Здоровый диапазон 0.1-0.3. Ноль означает, что шаги слишком робкие
    (поднимай lr или число эпох), 0.5+ — что роллаут переучивается.

    Считай через уже написанный clipped_surrogate, а не копируй условие
    второй раз: два экземпляра одной формулы обязательно разъедутся. Сэмпл
    обрезан ровно тогда, когда clipped_surrogate отличается от ratio * A.
    """
    if not ratios:
        return 0.0
    hits = sum(
        1
        for r, a in zip(ratios, advantages)
        if clipped_surrogate(r, a, eps) != r * a
    )
    return hits / len(ratios)


def approx_kl(old_log_probs, new_log_probs):
    """Дешёвая оценка KL(pi_old || pi_new) = среднее (log_old - log_new).

    approx_kl([-1.0, -2.0], [-1.0, -2.0])  ->  0.0    (политика та же)
    approx_kl([-1.0], [-2.0])              ->  1.0
    approx_kl([], [])                      ->  0.0

    Это не настоящая KL, а её оценка по сэмплам той же старой политики. Может
    выйти слегка отрицательной на конечном батче — это нормально и не повод
    брать abs().

    В PPO её держат в [0, 0.02]. Улетело за 0.1 — режь K_EPOCHS или lr.
    """
    if not old_log_probs:
        return 0.0
    return sum(o - n for o, n in zip(old_log_probs, new_log_probs)) / len(old_log_probs)


def ppo_actor_step(theta, x, action, log_pi_old, advantage, lr=0.05, eps=0.2):
    """Один шаг актора PPO по одному сэмплу. Вернуть новую матрицу theta.

    theta[i][j] += lr * scale * (onehot(action) - probs)[i] * x[j],
    где scale — это surrogate_gradient_scale текущего ratio.

    ppo_actor_step([[0.0], [0.0]], [1.0], 0, -math.log(2), 1.0)
        ->  [[0.025], [-0.025]]    (ratio == 1, коридор не тронут)

    Порядок важен: ratio считается по ТЕКУЩЕЙ theta против замороженного
    log_pi_old, снятого во время роллаута. log_pi_old не пересчитывается.

    Функция ничего не мутирует: theta на входе остаётся прежней.
    """
    probs = action_probs(theta, x)
    logp = math.log(max(probs[action], 1e-12))
    ratio = importance_ratio(logp, log_pi_old)
    scale = surrogate_gradient_scale(ratio, advantage, eps)
    grad = [(1.0 if i == action else 0.0) - p for i, p in enumerate(probs)]
    # новая матрица, а не += на месте: вызывающий часто хранит theta_old
    return [
        [theta[i][j] + lr * scale * grad[i] * x[j] for j in range(len(x))]
        for i in range(len(theta))
    ]


def ppo_update(theta, batch, lr=0.05, eps=0.2, epochs=4):
    """K эпох по одному роллауту. Вернуть (new_theta, mean_kl, clip_frac).

    batch — список словарей {"x": features, "a": action, "log_pi_old": float,
    "adv": float}. advantage уже посчитан и заморожен: пересчитывать его
    внутри эпох нельзя, он должен остаться константой.

    Диагностика снимается по ПОСЛЕДНЕЙ эпохе — именно там политика дальше
    всего от pi_old, и именно там KL опасен.

    Ровно этим PPO и отличается от A2C: A2C делает один проход и выбрасывает
    данные, PPO проходит их epochs раз, а коридор clip держит политику рядом
    с той, что собрала роллаут.

    Проверь себя: при epochs=1 и одном сэмпле clip_frac обязан быть 0.0, а
    mean_kl == 0.0 — ratio на первом сэмпле в точности единица.
    """
    current = [row[:] for row in theta]
    mean_kl, clip_frac = 0.0, 0.0
    for _ in range(epochs):
        ratios, advs, olds, news = [], [], [], []
        for rec in batch:
            probs = action_probs(current, rec["x"])
            logp = math.log(max(probs[rec["a"]], 1e-12))
            ratios.append(importance_ratio(logp, rec["log_pi_old"]))
            advs.append(rec["adv"])
            olds.append(rec["log_pi_old"])
            news.append(logp)
            current = ppo_actor_step(
                current, rec["x"], rec["a"], rec["log_pi_old"], rec["adv"], lr, eps
            )
        # диагностика считается ДО шага по каждому сэмплу, но пишется каждую
        # эпоху заново: наружу уходят числа последней, самой рискованной
        mean_kl = approx_kl(olds, news)
        clip_frac = clip_fraction(ratios, advs, eps)
    return current, mean_kl, clip_frac
