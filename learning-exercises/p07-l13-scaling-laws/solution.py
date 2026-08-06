"""
Законы масштабирования — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Закон Хоффманна: L(N, D) = A/N^alpha + B/D^beta + E.

    chinchilla_loss(70e9, 1400e9)   ->  1.9366...   (Chinchilla 70B)
    chinchilla_loss(175e9, 300e9)   ->  2.0022...   (GPT-3, недотренирован)
    chinchilla_loss(1e30, 1e30)     ->  1.6900...   (упёрлись в потолок E)

    N — число параметров (без эмбеддингов), D — число обучающих токенов.
    Три слагаемых: не хватает ёмкости, не хватает данных, и E — энтропия
    самого текста, ниже которой не опустится никто и никогда.

    Ловушка: N и D здесь абсолютные числа, а не миллиарды. 70e9, не 70.
    """
    return A / N ** alpha + B / D ** beta + E


def compute_flops(N, D):
    """Бюджет обучения в FLOPs: C = 6 * N * D.

    compute_flops(70e9, 1400e9)  ->  5.88e+23
    compute_flops(8e9, 15e12)    ->  7.2e+23

    Откуда шестёрка: 2 FLOPs на умножение-с-накоплением, forward — 2ND,
    backward — вдвое дороже, итого 6ND на весь шаг обучения. Это грубая
    оценка, но она держится в пределах десятков процентов на любом
    стандартном трансформере.
    """
    return 6.0 * N * D


def tokens_for_budget(C, N):
    """Сколько токенов влезает в бюджет C при заданном размере модели N.

    tokens_for_budget(6e23, 1e10)  ->  1e+13
    tokens_for_budget(1e24, 1e9)   ->  1.666...e+14

    Это compute_flops, развёрнутая относительно D. Ровно та связь, вдоль
    которой идёт весь спор Kaplan против Chinchilla: бюджет фиксирован,
    и каждый лишний параметр отнимает токены.
    """
    return C / (6.0 * N)


def compute_optimal(C, n_grid=400, log_n_min=5.0, log_n_max=14.0):
    """Перебором по сетке найти (N, D, loss), минимизирующие loss при 6ND = C.

    compute_optimal(1e23)  ->  (примерно 1.46e10, 1.14e12, 2.005)

    Вернуть кортеж (N, D, loss).

    Сетка равномерна по log10(N), а не по N: оптимум ищется на восьми
    порядках, и линейная сетка потратила бы все точки на самые большие
    модели. Точность ответа — один шаг сетки, поэтому n_grid влияет на
    третий знак.

    Ловушка: D = C/(6N), а не C/N. И D не может быть меньше нескольких
    миллионов токенов — такие точки просто выкинь, иначе формула выдаст
    фантастически низкий loss на модели, обученной на десяти примерах.
    """
    best_N, best_D, best_loss = None, None, float("inf")
    for i in range(n_grid):
        log_N = log_n_min + (log_n_max - log_n_min) * i / (n_grid - 1)
        N = 10.0 ** log_N
        D = tokens_for_budget(C, N)
        if D < 1e6:
            continue
        loss = chinchilla_loss(N, D)
        if loss < best_loss:
            best_N, best_D, best_loss = N, D, loss
    return best_N, best_D, best_loss


def optimal_exponents(alpha=0.34, beta=0.28):
    """Показатели степени, с которыми оптимальные N и D растут по бюджету C.

    optimal_exponents()            ->  (0.4516..., 0.5483...)
    optimal_exponents(0.5, 0.5)    ->  (0.5, 0.5)

    Вернуть кортеж (a, b): N_opt ~ C^a, D_opt ~ C^b.

    Выводится из условия dL/dN = 0 при D = C/(6N): a = beta/(alpha+beta),
    b = alpha/(alpha+beta). Сумма a + b всегда ровно 1 — иначе произведение
    N*D перестало бы быть пропорционально C.

    Отсюда же видно, что отношение D/N растёт как C^(b-a), а вовсе не
    держится на 20: правило «20 токенов на параметр» — это снимок для
    того диапазона компьюта, который изучала Chinchilla, а не следствие
    формулы.
    """
    total = alpha + beta
    return beta / total, alpha / total


def overtraining_tradeoff(C, shrink=10.0):
    """Цена перетренировки: модель в `shrink` раз меньше, данных во столько же больше.

    overtraining_tradeoff(1e24, 10.0)  ->  (N_opt/10, D_opt*10, ~0.061, 10.0)

    Вернуть кортеж (N_small, D_big, loss_penalty, inference_speedup):
      N_small = N_opt / shrink, D_big = D_opt * shrink  — бюджет 6ND тот же;
      loss_penalty = loss(N_small, D_big) - loss_opt    — всегда >= 0;
      inference_speedup = N_opt / N_small               — во столько раз
      дешевле каждый токен на инференсе.

    Это вся логика Llama 3 8B на 15T токенов: заплатить долей нáта в
    качестве, чтобы навсегда сэкономить на инференсе. Обучение платишь
    один раз, инференс — каждый запрос.
    """
    N_opt, D_opt, loss_opt = compute_optimal(C)
    N_small = N_opt / shrink
    D_big = D_opt * shrink
    penalty = chinchilla_loss(N_small, D_big) - loss_opt
    return N_small, D_big, penalty, N_opt / N_small


def min_compute_for_loss(target_loss, log_c_min=15.0, log_c_max=40.0, n_grid=150):
    """Наименьший бюджет C на компьют-оптимальной границе, дающий loss <= target.

    min_compute_for_loss(2.5)  ->  примерно 2.2e+20
    min_compute_for_loss(1.5)  ->  None

    None означает «недостижимо в заданном диапазоне»: любой loss ниже E
    не берётся никаким компьютом, потому что E — энтропия данных, а не
    недостаток модели.

    Так и планируют бюджеты: не «сколько GPU у нас есть», а «сколько
    нужно, чтобы срезать ещё 0.1 кросс-энтропии» — и очень быстро
    выясняется, что следующие 0.1 стоят на порядок дороже предыдущих.
    """
    for i in range(n_grid + 1):
        log_C = log_c_min + (log_c_max - log_c_min) * i / n_grid
        if compute_optimal(10.0 ** log_C)[2] <= target_loss:
            return 10.0 ** log_C
    return None


def emergence_curves(budgets, threshold):
    """Один и тот же прогресс двумя метриками: непрерывной и пороговой.

    Вернуть кортеж (smooth, stepped):
      smooth[i]  = loss компьют-оптимальной модели при budgets[i];
      stepped[i] = 1.0, если smooth[i] < threshold, иначе 0.0.

    emergence_curves([1e18, 1e22, 1e26], 2.2)  ->  ([3.54, 2.14, 1.80],
                                                    [0.0, 1.0, 1.0])

    Это возражение Schaeffer 2023 к «эмерджентности»: smooth падает
    гладко и предсказуемо, а stepped прыгает с 0 на 1 за один шаг. Скачок
    живёт в метрике (exact match, accuracy по порогу), а не в модели.
    Планируй бюджет по smooth, не по бенчмарку с порогом.
    """
    smooth = [compute_optimal(C)[2] for C in budgets]
    stepped = [1.0 if loss < threshold else 0.0 for loss in smooth]
    return smooth, stepped
