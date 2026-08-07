"""
Gradient checkpointing и пересчёт активаций

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l34-gradient-checkpointing
Разбор:  /check-code p10-l34-gradient-checkpointing
"""

import math

ACTS_PER_LAYER = 3  # сколько промежуточных тензоров на слой держит настоящий autograd


def layer_forward(x, layer):
    """Прямой проход одного слоя: y = w2 * relu(w1 * x + b1) + b2.

    layer_forward([1.0, 1.0], ([2.0, 1.0], [0.0, -3.0], [3.0, 1.0], [1.0, 0.0]))
        ->  [7.0, 0.0]

    Всё поэлементно: j-я координата выхода зависит только от j-й координаты
    входа. Разбор примера по второй координате: pre = 1*1 - 3 = -2, relu
    его гасит, значит y = 1*0 + 0 = 0.

    Промежуточные pre и h НЕ возвращаются: этот слой ничего не хранит. Всё,
    что backward-у понадобится, он посчитает сам из x.
    """
    raise NotImplementedError


def layer_backward(x, layer, grad_y):
    """Обратный проход одного слоя. Кортеж (grad_x, (gw1, gb1, gw2, gb2)).

    layer_backward([1.0, 1.0], ([2.0, 1.0], [0.0, -3.0], [3.0, 1.0], [1.0, 0.0]), [1.0, 1.0])
        ->  ([6.0, 0.0], ([3.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 1.0]))

    Цепное правило по одной координате:
        gw2 = h * grad_y            gb2 = grad_y
        gpre = w2 * grad_y, если pre > 0, иначе 0
        gw1 = x * gpre              gb1 = gpre
        grad_x = w1 * gpre

    Обрати внимание: на входе только x, pre и h пересчитываются внутри. Это
    уже checkpointing, только на уровне одного слоя — платим один лишний
    forward слоя и не храним ничего.

    Ловушка: relu в нуле. Берём gpre = 0 при pre <= 0 — то же соглашение,
    что и у d_relu из Фазы 1.
    """
    raise NotImplementedError


def forward_store_all(x, params):
    """Прямой проход без экономии. Кортеж (выход, список активаций).

    forward_store_all([1.0], [([1.0], [0.0], [1.0], [0.0])])
        ->  ([1.0], [[1.0], [1.0]])

    activations[i] — вход i-го слоя, activations[-1] — выход сети. Длина
    списка всегда len(params) + 1: это и есть та память, которую
    checkpointing будет резать.

    Использует layer_forward.
    """
    raise NotImplementedError


def backward_store_all(grad_y, activations, params):
    """Обратный проход по сохранённым активациям. Кортеж (grad_x, список градиентов).

    Идём с последнего слоя к первому, каждому отдаём его сохранённый вход
    activations[i]. grad_y — градиент по выходу сети (для loss = sum(y) это
    единицы).

    Список градиентов идёт в порядке слоёв, а не в порядке обхода: grads[0]
    относится к params[0]. Перепутать — классика, тесты это ловят.

    Использует layer_backward.
    """
    raise NotImplementedError


def forward_checkpointed(x, params, segment):
    """Прямой проход, хранящий только входы сегментов. Кортеж (выход, saved).

    forward_checkpointed([1.0], [([1.0], [0.0], [1.0], [0.0])] * 4, 2)
        ->  выход тот же, но saved короче: 2 записи вместо 5

    Проходим все слои как обычно, но в saved кладём вход только каждого
    segment-го. Выход сети не сохраняется — он и так возвращается.

    len(saved) равен ceil(len(params) / segment). При segment = 1 экономии
    нет: сохраняются все входы, ровно как в forward_store_all (минус
    выход). При segment = len(params) хранится один-единственный вход.

    segment меньше 1 — ValueError: сегмент нулевой длины бессмысленен и
    даёт деление на ноль ниже по стеку.

    Использует layer_forward.
    """
    raise NotImplementedError


def backward_checkpointed(grad_y, saved, params, segment):
    """Обратный проход с пересчётом сегментов. Кортеж (grad_x, список градиентов).

    Возвращает РОВНО то же, что backward_store_all на тех же данных, —
    это главное свойство checkpointing и главный тест урока.

    Как: идём по сегментам с конца. Для каждого берём его сохранённый вход,
    прогоняем forward_store_all по слоям сегмента (вот он, лишний forward),
    получаем активации и отдаём их backward_store_all.

    Пик памяти: len(saved) сохранённых плюс активации ОДНОГО сегмента.
    Пересчитанные активации умирают, как только сегмент отработал.

    Использует forward_store_all и backward_store_all.
    """
    raise NotImplementedError


def checkpoint_budget(n_layers, width, segment=None, per_layer=ACTS_PER_LAYER,
                      recompute_fraction=1.0):
    """Память и FLOPs при заданной нарезке. Словарь floats, flops, overhead.

    checkpoint_budget(64, 1000)["floats"]     ->  192000
    checkpoint_budget(64, 1000, 8)["floats"]  ->  32000
    checkpoint_budget(64, 1000, 8)["overhead"]  ->  примерно 0.333

    Без checkpointing (segment=None) живут активации всех слоёв:
    n_layers * per_layer * width. FLOPs — forward плюс двойной backward, то
    есть 3 * n_layers условных единиц.

    С сегментом живут входы сегментов (ceil(n_layers / segment)) плюс
    активации одного пересчитываемого сегмента (segment * per_layer).
    FLOPs добавляют recompute_fraction * n_layers.

    recompute_fraction = 1.0 — пересчитываем слой целиком, накладные 33%.
    Selective checkpointing (Korthikanti) пересчитывает только внимание,
    это примерно 0.15, и накладные падают до 5%. Обрати внимание: память
    зависит от segment, а FLOPs — нет. Это два независимых рычага.
    """
    raise NotImplementedError


def optimal_segment(n_layers, per_layer=ACTS_PER_LAYER):
    """Длина сегмента с наименьшим пиком памяти. Число от 1 до n_layers.

    optimal_segment(64, per_layer=1)  ->  8    (это ровно sqrt(64))
    optimal_segment(64)               ->  4

    Перебираем все длины и берём минимум floats из checkpoint_budget.
    При равенстве — меньший сегмент: меньше слоёв придётся пересчитывать
    за один заход, а значит ниже пик внутри сегмента.

    Правило sqrt(L) из статьи Chen 2016 верно ровно при per_layer = 1:
    ceil(L/k) + k минимизируется на sqrt(L). Чем больше промежуточных
    тензоров на слое, тем короче выгодный сегмент — потому что хранить
    границы дешевле, чем держать живой длинный сегмент.

    Использует checkpoint_budget.
    """
    raise NotImplementedError
