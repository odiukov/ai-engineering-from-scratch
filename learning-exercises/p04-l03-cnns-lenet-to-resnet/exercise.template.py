"""
CNN от LeNet до ResNet

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p04-l03-cnns-lenet-to-resnet
Разбор:  /check-code p04-l03-cnns-lenet-to-resnet
"""


def spatial_out(size, kernel, stride=1, padding=0):
    """Пространственный размер после conv или pool: (size - k + 2p)//s + 1.

    spatial_out(32, 5)             ->  28   (conv 5x5 в LeNet)
    spatial_out(28, 2, stride=2)   ->  14   (avg pool 2x2)
    spatial_out(32, 3, padding=1)  ->  32   (same padding)

    Pool — это та же формула: окно 2x2 с шагом 2. Отдельной математики для
    пулинга не существует, поэтому и функция одна.
    """
    raise NotImplementedError


def conv_params(c_in, c_out, kernel, bias=True):
    """Число параметров свёрточного слоя: c_out*c_in*k*k (+ c_out на смещения).

    conv_params(1, 6, 5)               ->  156
    conv_params(3, 64, 3)              ->  1792
    conv_params(3, 64, 3, bias=False)  ->  1728

    bias=False — не микрооптимизация, а конвенция ResNet: сразу за conv идёт
    BatchNorm, у которого есть собственный сдвиг beta, и второе смещение
    просто дублирует первое.
    """
    raise NotImplementedError


def dense_params(in_features, out_features, bias=True):
    """Число параметров полносвязного слоя: in*out (+ out на смещения).

    dense_params(400, 120)  ->  48120
    dense_params(84, 10)    ->  850

    Сравни с conv_params: у свёртки размер картинки в формулу не входит
    вообще, у dense — входит целиком. Отсюда и 138M параметров у VGG-16,
    почти все в трёх последних слоях.
    """
    raise NotImplementedError


def lenet5_shapes(input_size=32):
    """Формы карт признаков LeNet-5 по шагам. Список пар (имя, (C, H, W)).

    Схема: conv 5x5 -> avg pool 2x2 -> conv 5x5 -> avg pool 2x2.
    Каналы: 1 -> 6 -> 6 -> 16 -> 16.

    lenet5_shapes()[0]   ->  ("input", (1, 32, 32))
    lenet5_shapes()[-1]  ->  ("pool2", (16, 5, 5))

    Именно поэтому LeNet требует вход 32x32, а не родные для MNIST 28x28:
    на 28 после всех четырёх шагов останется 4x4, flatten даст 256 вместо
    400, и первый dense-слой откажется принимать вход. Классическая ошибка
    при переносе чужого кода.

    Считай через spatial_out — переписывать формулу заново незачем.
    """
    raise NotImplementedError


def lenet5_params():
    """Параметры LeNet-5 послойно. Список пар (имя, число).

    Слои: conv1 (1->6, 5x5), conv2 (6->16, 5x5), fc1 (400->120),
    fc2 (120->84), fc3 (84->10).

    lenet5_params()[0]           ->  ("conv1", 156)
    sum(n for _, n in ...)       ->  61706

    Число 400 — это не константа из воздуха, а 16*5*5 из lenet5_shapes():
    возьми последнюю форму и перемножь.

    Разложение поучительно: 78% всех параметров сидят в fc1, а обе свёртки
    вместе занимают меньше 5%. Вся история CNN после 1998 — про то, как
    убрать этот жирный dense-слой.
    """
    raise NotImplementedError


def shortcut_kind(in_c, out_c, stride):
    """Какой обходной путь нужен BasicBlock: "identity" или "projection".

    shortcut_kind(64, 64, 1)   ->  "identity"
    shortcut_kind(64, 128, 1)  ->  "projection"
    shortcut_kind(64, 64, 2)   ->  "projection"

    Правило простое: складывать можно только тензоры одинаковой формы.
    Если stride поменял пространственный размер или число каналов выросло,
    вместо тождества ставят conv 1x1 с тем же stride плюс BatchNorm.

    Забыть про этот случай — ошибка номер один при ручной сборке ResNet:
    код упадёт на сложении форм, а если размеры случайно совпадут, то
    молча сложит несопоставимые карты признаков.
    """
    raise NotImplementedError


def residual_forward(x, f):
    """Residual-блок: y = F(x) + x. x — список чисел, f — функция над ним.

    residual_forward([1.0, 2.0], lambda v: [0.0, 0.0])   ->  [1.0, 2.0]
    residual_forward([1.0, 2.0], lambda v: [10.0, 20.0]) ->  [11.0, 22.0]

    Ровно одно сложение отделяет обычный блок от residual, и оно даёт блоку
    аварийный выход: занулив F, слой превращается в тождество. Значит сеть
    из ста блоков в худшем случае не хуже сети из одного — и оптимизатор
    соглашается делать каждый блок хотя бы чуть-чуть полезным.
    """
    raise NotImplementedError


def gradient_scale(gains, residual=False):
    """Во сколько раз градиент изменится, пройдя стопку слоёв назад.

    gains — список коэффициентов усиления, по одному на слой.

    gradient_scale([0.5, 0.5])                  ->  0.25
    gradient_scale([0.5, 0.5], residual=True)   ->  2.25
    gradient_scale([0.0] * 50, residual=True)   ->  1.0

    Обычная стопка перемножает сами коэффициенты: dy/dx = g_L * ... * g_1.
    Полсотни множителей меньше единицы — и до первого слоя доходит ноль,
    это и есть degradation problem, из-за которой VGG-32 не обучалась.

    У residual-блока производная другая: y = F(x) + x даёт dy/dx = g + 1.
    Даже полностью мёртвый блок (g = 0) пропускает градиент как есть,
    множитель ровно 1. Вот почему ResNet-152 обучается, а plain-34 нет.
    """
    raise NotImplementedError
