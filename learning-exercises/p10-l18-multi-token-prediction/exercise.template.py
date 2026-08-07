"""
Multi-Token Prediction (MTP)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l18-multi-token-prediction
Разбор:  /check-code p10-l18-multi-token-prediction
"""

import math


def matvec(M, x):
    """Матрица на вектор: out[i] = sum_j M[i][j] * x[j].

    matvec([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0])  ->  [3.0, 7.0]
    matvec([[0.0, 0.0]], [5.0, 5.0])              ->  [0.0]

    Число столбцов матрицы обязано совпасть с длиной вектора, иначе zip
    молча обрежет и ошибка всплывёт где-то далеко -> ValueError.
    """
    raise NotImplementedError


def softmax(logits):
    """Логиты -> распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([0.0, 1000.0])    ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум перед
    экспонентой.
    """
    raise NotImplementedError


def cross_entropy(logits, target):
    """Кросс-энтропия одного токена: -log(softmax(logits)[target]).

    cross_entropy([0.0, 0.0], 0)      ->  log(2) = 0.693...
    cross_entropy([0.0] * 32, 7)      ->  log(32) = 3.466...
    cross_entropy([100.0, 0.0], 0)    ->  почти 0

    Считай через log-sum-exp, а не как -log(softmax(...)[target]):
    у уверенно неправильного предсказания вероятность уходит в 0.0,
    и math.log(0.0) падает с ValueError. Формула безопасная:
        lse = m + log(sum(exp(z - m))),  loss = lse - logits[target]

    Равномерные логиты дают log(V) — та самая «потеря необученной модели»,
    с которой начинается любая кривая обучения.
    """
    raise NotImplementedError


def rms_norm(x, eps=1e-6):
    """RMSNorm без обучаемого масштаба: x_i / sqrt(mean(x^2) + eps).

    rms_norm([3.0, 4.0], 0.0)  ->  [0.848..., 1.131...]
    rms_norm([0.0, 0.0])       ->  [0.0, 0.0]   (спасает eps)

    Среднее НЕ вычитается — это RMSNorm, а не LayerNorm. В MTP через неё
    прогоняют оба слагаемых перед склейкой, чтобы скрытое состояние и
    эмбеддинг вошли в проекцию с сопоставимым масштабом.
    """
    raise NotImplementedError


def depth_hidden(prev_hidden, token_embed, M):
    """Скрытое состояние следующей глубины: h^(k) из h^(k-1) и E(t_{i+k}).

    Модуль модели: MTP module k (проекция M_k; трансформерный блок T_k в
    этой игрушке опущен, он ничего не добавляет к разбираемой механике).

    Порядок: rms_norm обоих векторов -> склейка в один длиной 2h ->
    умножение на M размера h x 2h.

    depth_hidden([1.0], [1.0], [[0.5, 0.5]])  ->  [1.0]

    Ловушка: склеивать надо ПОСЛЕ нормировки каждого куска по отдельности.
    Нормировать склейку целиком — другая операция и другой результат.

    Здесь и живёт вся разница с параллельным MTP Gloeckle: у него глубина
    k смотрит на h^(0), а тут — на h^(k-1) плюс уже известный токен.
    """
    raise NotImplementedError


def mtp_depth_losses(h0, targets, projections, embedding, W_out):
    """Потери по всем глубинам: L_1..L_D для последовательного MTP.

    h0          — скрытое состояние основной модели, h^(0);
    targets     — список истинных токенов t_{i+1}, t_{i+2}, ...;
    projections — матрицы M_k, нужно как минимум len(targets) - 1 штук;
    embedding   — общая таблица эмбеддингов (та же, что у основной модели);
    W_out       — общая выходная голова (та же, что у основной модели).

    Цикл по глубинам:
      logits = W_out * h,  L_k = CE(logits, targets[k-1]),
      затем h = depth_hidden(h, embedding[targets[k-1]], M_k).

    Обрати внимание: в h^(k) подставляется ИСТИННЫЙ токен, а не свой
    прогноз. Это teacher forcing; на инференсе туда идёт черновой токен,
    и модуль превращается в EAGLE-подобный драфтер.

    Проверяемое свойство: смена targets[0] меняет L_2. У параллельного
    MTP не менялась бы — там все глубины смотрят на один и тот же h^(0).
    """
    raise NotImplementedError


def joint_loss(main_loss, depth_losses, lam):
    """Полная потеря обучения: L_main + (lambda / D) * sum(L_k).

    joint_loss(2.0, [1.0, 3.0], 0.3)  ->  2.0 + 0.3 * 2.0 = 2.6
    joint_loss(2.0, [], 0.3)          ->  2.0

    lambda у DeepSeek-V3 равна 0.3 на первых 10% обучения и 0.1 дальше:
    сначала плотный сигнал полезен, потом начинает мешать основной задаче.

    Пустой список глубин -> просто L_main, без деления на ноль.
    """
    raise NotImplementedError


def mtp_extra_params(hidden, depths=1, ffn_hidden=None):
    """Сколько параметров добавляет MTP поверх основной модели.

    Возвращает словарь: projection, attention, mlp, per_module, shared, total.

    На один модуль:
      projection M_k = 2 * hidden^2   (склейка длиной 2h -> h)
      attention      = 4 * hidden^2   (Wq, Wk, Wv, Wo)
      mlp            = 3 * hidden * ffn_hidden, по умолчанию ffn = 8h/3
    Итого примерно 14 * hidden^2 на модуль — та самая оценка из урока.

    shared всегда 0: эмбеддинг и выходная голова переиспользуются, это
    буквально те же тензоры, а не копии.

    Для hidden = 7168 и одного модуля выходит около 720M параметров.
    DeepSeek-V3 отчитался про 14B — разница в том, что у них MLP внутри
    модуля тоже MoE.
    """
    raise NotImplementedError
