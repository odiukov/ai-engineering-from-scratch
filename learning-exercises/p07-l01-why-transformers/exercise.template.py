"""
Почему трансформеры

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p07-l01-why-transformers
Разбор:  /check-code p07-l01-why-transformers
"""

import math


def rnn_state(xs, decay=0.9):
    """Рекуррентный проход: h = decay * h + x, по одному элементу за шаг.

    Начальное состояние — 0.0.

    rnn_state([1.0, 0.0, 0.0], decay=0.5)  ->  0.25
    rnn_state([0.0, 0.0, 1.0], decay=0.5)  ->  1.0
    rnn_state([], 0.5)                     ->  0.0

    Сравни два первых примера: одна и та же единица даёт 0.25 в начале и
    1.0 в конце. Порядок решает всё, а старое затухает как decay^k — это
    vanishing gradient в миниатюре, на трёх числах.
    """
    raise NotImplementedError


def attention_mean(xs):
    """Параллельная свёртка: среднее по всем элементам сразу.

    attention_mean([1.0, 0.0, 0.0])  ->  1/3
    attention_mean([0.0, 0.0, 1.0])  ->  1/3

    Пустой вход — ValueError: среднего ни у чего не бывает.

    Ни один элемент не ждёт другого, поэтому результат вообще не зависит от
    порядка. Это и достоинство (GPU считает всё разом), и проблема:
    порядок слов приходится доносить отдельно — см. урок 04.
    """
    raise NotImplementedError


def serial_scan(xs):
    """Префиксные суммы «в лоб»: out[i] = xs[0] + ... + xs[i].

    serial_scan([1.0, 2.0, 3.0])  ->  [1.0, 3.0, 6.0]
    serial_scan([])               ->  []

    Каждый шаг ждёт предыдущего: глубина зависимостей равна длине входа.
    """
    raise NotImplementedError


def hillis_steele_scan(xs):
    """Те же префиксные суммы, но параллельным алгоритмом Hillis-Steele.

    hillis_steele_scan([1.0, 2.0, 3.0])       ->  [1.0, 3.0, 6.0]
    hillis_steele_scan([1.0, 1.0, 1.0, 1.0])  ->  [1.0, 2.0, 3.0, 4.0]

    Схема: пока сдвиг step меньше длины, каждый элемент с индексом i >= step
    получает out[i] + out[i - step], потом step удваивается. Раундов выходит
    ceil(log2(n)) вместо n.

    Ловушка: внутри одного раунда читать надо СТАРЫЙ массив, а писать в
    новый. Если править на месте, элементы начнут видеть уже обновлённых
    соседей и суммы поедут.

    Сложений здесь БОЛЬШЕ, чем у serial_scan, а на GPU он всё равно быстрее.
    Решает глубина графа зависимостей, а не число операций.
    """
    raise NotImplementedError


def scan_rounds(n):
    """Сколько раундов делает Hillis-Steele на входе длины n.

    scan_rounds(1)     ->  0
    scan_rounds(1024)  ->  10
    scan_rounds(1000)  ->  10

    Это ceil(log2(n)) при n > 1 и 0 при n <= 1.

    Сравни с последовательным сканом, где глубина ровно n: на 1024 элементах
    10 против 1024. Вот из чего складывается разрыв в wall-clock между RNN и
    трансформером на длинной последовательности.
    """
    raise NotImplementedError


def attention_memory_cells(seq_len, n_heads=1, n_layers=1):
    """Сколько ячеек займут ЯВНО материализованные матрицы внимания.

    attention_memory_cells(4)                         ->  16
    attention_memory_cells(4, n_heads=8)              ->  128
    attention_memory_cells(8, n_heads=8, n_layers=2)  ->  1024

    Обычная реализация хранит n_layers * n_heads * seq_len^2 ячеек: удвоил
    контекст — получил вчетверо больше памяти под score/weight-матрицы.

    FlashAttention (урок 12) вычисляет ТО ЖЕ точное внимание IO-aware блоками
    и не материализует всю матрицу: его дополнительная память линейна по
    длине. Квадратичным остаётся объём вычислений, а не память этих матриц.
    """
    raise NotImplementedError


def pick_architecture(seq_len, streaming=False, has_matmul_accelerator=True):
    """Выбор архитектуры под задачу — таблица из раздела Use It.

    Правила проверяются строго в этом порядке:
      1. streaming=True (по токену за раз, память константная)   ->  "rnn"
      2. seq_len > 1_000_000 (память внимания взрывается)        ->  "linear-attention"
      3. has_matmul_accelerator=False (edge без матричного блока) ->  "rnn"
      4. всё остальное                                           ->  "transformer"

    pick_architecture(2048)                  ->  "transformer"
    pick_architecture(2048, streaming=True)  ->  "rnn"
    pick_architecture(5_000_000)             ->  "linear-attention"

    Порядок важен: стриминг с длиной 5 млн всё равно "rnn", потому что при
    потоковой выдаче вся последовательность в память и не кладётся.
    """
    raise NotImplementedError
