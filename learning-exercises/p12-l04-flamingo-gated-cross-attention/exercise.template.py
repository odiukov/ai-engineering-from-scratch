"""
Flamingo и gated cross-attention

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l04-flamingo-gated-cross-attention
Разбор:  /check-code p12-l04-flamingo-gated-cross-attention
"""

import math

IMAGE = "image"
TEXT = "text"


def cross_attention(queries, keys, values):
    """Cross-attention: каждый запрос собирает взвешенную сумму values.

    Вернуть список выходов, по одному на запрос (веса наружу не отдаём —
    здесь они не нужны).

    scores[j] = dot(query, keys[j]) / sqrt(dim_key), дальше softmax, дальше
    взвешенная сумма values.

    cross_attention([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], [[4.0], [0.0]])
        ->  [[2.679]]

    Ловушка: softmax наивным exp падает на больших логитах. Вычитай максимум.

    Пустые queries, keys или values, разное число keys и values — ValueError.
    """
    raise NotImplementedError


def perceiver_resampler(patches, latents, blocks=1):
    """Сжать любое число патчей в фиксированное число латентов.

    Один блок = cross-attention латентов по патчам плюс residual:
        latents = latents + cross_attention(latents, patches, patches)
    Повторить blocks раз. Flamingo использует 6 блоков и 64 латента.

    perceiver_resampler(patches_900, latents_64)  ->  64 вектора
    perceiver_resampler(patches_196, latents_64)  ->  64 вектора

    В этом весь смысл: картинка 224x224 и картинка 480x480 выходят одной и
    той же длины, поэтому слой gated cross-attention в LLM всегда видит
    одну и ту же форму, сколько бы картинок ни было в промпте.

    В игрушечной версии латенты и патчи одной размерности — иначе residual
    не сложится. blocks=0 возвращает копию латентов, blocks<0 — ValueError.
    Входной список латентов не менять.
    """
    raise NotImplementedError


def gated_residual(hidden, cross, alpha):
    """Затворённый residual Flamingo: out = hidden + tanh(alpha) * cross.

    gated_residual([[1.0, 2.0]], [[10.0, 10.0]], 0.0)  ->  [[1.0, 2.0]]

    Единственная по-настоящему важная строчка всей архитектуры. alpha
    инициализируется нулём, tanh(0) = 0, значит на шаге 0 новый слой —
    точный no-op, и замороженная LLM ведёт себя ровно как до вставки.
    Дальше alpha уезжает от нуля, и визуальный сигнал вливается плавно.

    Два следствия, которые стоит проверить:
      * при alpha = 0 равенство ТОЧНОЕ, а не приблизительное;
      * |tanh| <= 1, поэтому вклад никогда не превосходит сам cross —
        визуальная ветка не может затереть текстовое представление.

    Несовпадение форм hidden и cross — ValueError.
    """
    raise NotImplementedError


def gated_cross_attention_step(text_hidden, visual_tokens, alpha):
    """Полный вставной блок Flamingo между двумя слоями замороженной LLM.

    Текстовые скрытые состояния становятся запросами, визуальные токены —
    ключами и значениями, результат подмешивается через затвор.

    gated_cross_attention_step(hidden, visual, 0.0)   ->  hidden без изменений
    gated_cross_attention_step(hidden, visual, 2.0)   ->  hidden + 0.964*cross

    Обрати внимание, чего здесь НЕТ: входная последовательность LLM не
    меняется. Визуальные токены не подставляются в промпт, они живут сбоку
    и подмешиваются в скрытые состояния. Поэтому Flamingo умеет глотать
    сколько угодно картинок, не съедая контекст.
    """
    raise NotImplementedError


def most_recent_image(sequence):
    """Для каждой позиции — индекс ближайшей картинки слева (или None).

    sequence — список меток IMAGE и TEXT в порядке чтения.

    most_recent_image([TEXT, IMAGE, TEXT, IMAGE, TEXT])
        ->  [None, 0, 0, 1, 1]

    Индекс считается по картинкам, а не по позициям: первая картинка — 0,
    вторая — 1. Сама картинка «видит» себя.

    None у текста до первой картинки — не мелочь: это ровно те токены,
    которым визуальной информации ещё неоткуда взяться.

    Любая метка кроме IMAGE и TEXT — ValueError.
    """
    raise NotImplementedError


def interleaved_cross_mask(sequence, tokens_per_image):
    """Маска cross-attention для перемешанной последовательности картинок и текста.

    Матрица len(sequence) x (число картинок * tokens_per_image) из True/False.
    True — этой позиции разрешено смотреть на этот визуальный токен.

    interleaved_cross_mask([IMAGE, TEXT, IMAGE, TEXT], 2)
        ->  [[T, T, F, F],
             [T, T, F, F],
             [F, F, T, T],
             [F, F, T, T]]

    Правило Flamingo: позиция видит ТОЛЬКО последнюю предшествующую
    картинку — не все предыдущие. Ограничение сознательное: так модель
    учится связывать подпись с ближайшей к ней картинкой, и few-shot
    примеры не сливаются в кашу.

    Позиции до первой картинки не видят ничего — целая строка False.
    tokens_per_image <= 0 — ValueError.
    """
    raise NotImplementedError


def build_few_shot_prompt(examples, query_image):
    """Собрать few-shot промпт Flamingo: пары (картинка, подпись) и запрос.

    examples — список пар (имя картинки, подпись). Вернуть список пар
    (метка, содержимое), где метка это IMAGE или TEXT.

    build_few_shot_prompt([("cat.jpg", "A photo of a cat.")], "bird.jpg")
        ->  [(IMAGE, "cat.jpg"), (TEXT, "A photo of a cat."), (IMAGE, "bird.jpg")]

    Хвост принципиален: последняя картинка идёт БЕЗ подписи. Именно
    незакрытый шаблон заставляет модель продолжить его — никаких
    градиентных шагов, чистое in-context обучение.

    Ноль примеров — это zero-shot, ровно одна картинка на выходе.
    Пустая подпись — ValueError: демонстрация без ответа ничему не учит.
    """
    raise NotImplementedError
