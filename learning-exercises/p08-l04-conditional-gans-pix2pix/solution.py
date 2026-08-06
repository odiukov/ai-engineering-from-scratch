"""
Условные GAN и Pix2Pix — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def one_hot(c, num_classes):
    """One-hot вектор длины num_classes с единицей на позиции c.

    one_hot(0, 3)  ->  [1.0, 0.0, 0.0]
    one_hot(2, 3)  ->  [0.0, 0.0, 1.0]

    Самый простой способ подать условие в сеть. Большие модели вместо этого
    используют обучаемые эмбеддинги, FiLM или cross-attention, но идея та
    же: условие обязано доехать до входа.

    Ловушка: список в Python молча примет c = -1 и поставит единицу с
    конца. Индекс вне 0..num_classes-1 — ValueError.
    """
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive int")
    if not 0 <= c < num_classes:
        raise ValueError(f"class {c!r} out of range 0..{num_classes - 1}")
    v = [0.0] * num_classes
    v[c] = 1.0
    return v


def conditioned_input(x, c, num_classes):
    """Вход условной сети: вектор x, к которому приклеен one_hot(c).

    conditioned_input([0.5], 1, 2)       ->  [0.5, 0.0, 1.0]
    conditioned_input([1.0, 2.0], 0, 2)  ->  [1.0, 2.0, 1.0, 0.0]

    Ровно это отличает conditional GAN от обычного: условие получают И
    генератор, И дискриминатор. Если условие видит только G, то D не может
    наказать за несоответствие, и G быстро научится условие игнорировать —
    это первый пункт в списке ловушек урока.

    В Pix2Pix на месте one_hot стоит целая картинка-условие, а D судит
    именно пару (условие, выход).
    """
    return list(x) + one_hot(c, num_classes)


def linear_generator(z, c, W, b, num_classes):
    """Крошечный условный генератор: G(z, c) = W @ conditioned_input(z, c) + b.

    W — список строк длиной len(z) + num_classes, b — длиной len(W).

    linear_generator([1.0], 1, [[0.0, 0.0, 5.0]], [0.0], 2)  ->  [5.0]

    Столбцы W, приходящиеся на one_hot, и есть весь механизм условия:
    обнули их — и G(z, 0) станет равен G(z, 1), соответствие вход-выход
    исчезнет, а лосс этого даже не заметит.

    Несовпадение размеров — ValueError, иначе zip молча обрежет.
    """
    inp = conditioned_input(z, c, num_classes)
    if len(W) != len(b):
        raise ValueError(f"W has {len(W)} rows but b has {len(b)} items")
    out = []
    for row, bi in zip(W, b):
        if len(row) != len(inp):
            raise ValueError(
                f"row of length {len(row)} does not match input of length {len(inp)}"
            )
        out.append(sum(w * v for w, v in zip(row, inp)) + bi)
    return out


def l1_loss(y, y_hat):
    """Средняя абсолютная ошибка между двумя векторами.

    l1_loss([1.0, 2.0], [1.0, 4.0])  ->  1.0
    l1_loss([3.0], [3.0])            ->  0.0

    В Pix2Pix L1 стоит рядом с adversarial-членом с весом lambda = 100.
    L1 даёт более резкие края, чем L2, и причина не мистическая: оптимум L1
    — медиана, а медиана не размазывает несколько правдоподобных ответов в
    один нереальный средний. См. best_constant.

    Разная длина — ValueError.
    """
    if len(y) != len(y_hat):
        raise ValueError("y and y_hat must have the same length")
    if not y:
        raise ValueError("vectors must not be empty")
    return sum(abs(a - b) for a, b in zip(y, y_hat)) / len(y)


def best_constant(targets, norm="l1"):
    """Константа, минимизирующая суммарную ошибку до всех targets.

    best_constant([0.0, 0.0, 0.0, 100.0], "l1")  ->  0.0    (медиана)
    best_constant([0.0, 0.0, 0.0, 100.0], "l2")  ->  25.0   (среднее)

    Вот откуда «L1 режет резче, а L2 мылит»: когда на один вход есть
    несколько одинаково правдоподобных ответов, L2 выдаст их среднее —
    ответ, которого в данных не было ни разу. L1 выдаст медиану, то есть
    настоящий ответ.

    Для чётного числа targets медиан целый отрезок; верни его середину
    (так же, как statistics.median). norm не "l1" и не "l2" — ValueError.
    """
    if not targets:
        raise ValueError("targets must not be empty")
    if norm == "l2":
        return sum(targets) / len(targets)
    if norm == "l1":
        s = sorted(targets)
        mid = len(s) // 2
        if len(s) % 2:
            return s[mid]
        return (s[mid - 1] + s[mid]) / 2
    raise ValueError(f"unknown norm {norm!r}, expected 'l1' or 'l2'")


def pix2pix_generator_loss(d_fake, y, y_hat, lam=100.0):
    """Лосс генератора Pix2Pix: -log D(x, G(x)) + lam * L1(y, G(x)).

    d_fake — вероятность, которую дискриминатор дал ПАРЕ (условие, выход
    генератора). Именно пара, а не один выход: без условия на входе D не
    может проверить соответствие.

    pix2pix_generator_loss(0.5, [1.0], [1.0])  ->  0.6931...
    pix2pix_generator_loss(0.5, [1.0], [0.0])  ->  100.6931...

    lam = 100 — дефолт из статьи. Слишком мало — G рисует правдоподобное,
    но не то, что просили. Слишком много — L1 перевешивает adversarial и
    выход мылится; тогда lam понижают уже после стабилизации.

    Зажми d_fake, иначе log(0) уронит прогон на середине обучения.
    """
    if lam < 0:
        raise ValueError("lam must be non-negative")
    eps = 1e-12
    p = min(max(d_fake, eps), 1.0 - eps)
    return -math.log(p) + lam * l1_loss(y, y_hat)


def patchgan_score(image, patch_size, stride, score_fn):
    """Средняя оценка PatchGAN: картинка режется на патчи, score_fn судит каждый.

    image — список строк (список списков чисел), квадратные патчи
    patch_size x patch_size берутся с шагом stride. score_fn получает патч
    (список строк) и возвращает число. Вернуть среднее по всем патчам.

    Для картинки 4x4 при patch_size=2, stride=2 патчей будет 4,
    при patch_size=2, stride=1 — уже 9 (они перекрываются).

    Идея PatchGAN: правдоподобие локально. Вместо одной оценки на всю
    картинку D выдаёт сетку оценок и они усредняются — параметров меньше,
    обучение быстрее, детали резче. Заодно один плохой патч уже нельзя
    спрятать за девятью хорошими: он тянет среднее вниз.

    patch_size больше картинки или stride <= 0 — ValueError.
    """
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    height = len(image)
    width = len(image[0]) if height else 0
    if patch_size > height or patch_size > width:
        raise ValueError("patch_size does not fit into the image")
    scores = []
    for top in range(0, height - patch_size + 1, stride):
        for left in range(0, width - patch_size + 1, stride):
            patch = [row[left : left + patch_size] for row in image[top : top + patch_size]]
            scores.append(score_fn(patch))
    return sum(scores) / len(scores)


def cycle_consistency_loss(x, forward_fn, backward_fn):
    """Цикл-лосс CycleGAN: L1 между x и backward_fn(forward_fn(x)).

    cycle_consistency_loss([1.0, 2.0], lambda v: v, lambda v: v)  ->  0.0
    cycle_consistency_loss([1.0], lambda v: [a + 1 for a in v],
                                  lambda v: v)                    ->  1.0

    Так CycleGAN обходится без парных данных: пары нет, зато есть
    требование, чтобы путь X -> Y -> X возвращал исходное. Одного этого
    хватает, чтобы выучить перевод между двумя доменами.

    Ноль цикл-лосса ещё не значит, что перевод осмысленный: тождественная
    пара функций даёт ноль и ничего не переводит. Поэтому цикл-лосс всегда
    идёт в паре с adversarial-членом, а не вместо него.
    """
    return l1_loss(x, backward_fn(forward_fn(x)))
