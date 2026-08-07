"""
Нейронные аудиокодеки

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p06-l13-neural-audio-codecs
Разбор:  /check-code p06-l13-neural-audio-codecs
"""

import math


def nearest_code(codebook, value):
    """Индекс ближайшего кода в codebook к числу value.

    nearest_code([-1.0, 0.0, 1.0], 0.4)   ->  1
    nearest_code([-1.0, 0.0, 1.0], 0.6)   ->  2
    nearest_code([0.0, 1.0], 0.5)         ->  0   (ничья — берём меньший индекс)

    Возвращать нужно ИНДЕКС, а не сам код: кодек передаёт по сети именно
    номера, 10 бит на кодовое слово при codebook из 1024 элементов.

    На пустом codebook — ValueError, иначе ошибка вылезет позже и в другом месте.
    """
    raise NotImplementedError


def uniform_codebook(size, scale=1.0):
    """Равномерный codebook из size кодов, симметричный, от -scale до +scale.

    uniform_codebook(3, 1.0)  ->  [-1.0, 0.0, 1.0]
    uniform_codebook(5, 2.0)  ->  [-2.0, -1.0, 0.0, 1.0, 2.0]
    uniform_codebook(1, 7.0)  ->  [0.0]

    Настоящие кодеки codebook не задают формулой, а обучают. Но у
    равномерного есть свойство, ради которого он здесь: при нечётном size
    в нём есть ровно 0.0, и тогда квантование НИКОГДА не увеличивает
    остаток. На этом держится вся каскадная схема RVQ.

    size < 1 — ValueError.
    """
    raise NotImplementedError


def quantize_layer(values, codebook):
    """Один слой квантования: вернуть (indices, residuals).

    quantize_layer([0.4, -0.9], [-1.0, 0.0, 1.0])  ->  ([1, 0], [0.4, 0.1])

    indices — что кодек передаёт, residuals — то, что он НЕ передал.
    Именно residuals уходят в следующий codebook каскада: отсюда и
    название Residual Vector Quantization.
    """
    raise NotImplementedError


def rvq_encode(values, codebooks):
    """Каскад RVQ: первый codebook кодирует сигнал, каждый следующий — остаток.

    rvq_encode([0.4], [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]])  ->  [[1], [1]]

    Возвращает список списков индексов — по одному списку на codebook.
    Длина результата равна len(codebooks), длина каждого списка — len(values).

    Ловушка: в следующий слой уходит ОСТАТОК, а не исходный сигнал. Если
    подать исходный, все слои посчитают одно и то же и качество не вырастет.
    """
    raise NotImplementedError


def rvq_decode(all_indices, codebooks, length):
    """Декодер RVQ: сумма выбранных кодов по всем слоям.

    rvq_decode([[1], [2]], [[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], 1)  ->  [0.5]
    rvq_decode([], [], 3)                                            ->  [0.0, 0.0, 0.0]

    Декодер не «выбирает лучший» слой — он складывает все. Поэтому можно
    оборвать каскад на любом слое: получится тот же сигнал, только грубее.
    Это и есть переменный битрейт у EnCodec одной и той же моделью.
    """
    raise NotImplementedError


def reconstruction_mse(original, reconstructed):
    """Средняя квадратичная ошибка восстановления.

    reconstruction_mse([1.0, 2.0], [1.0, 2.0])  ->  0.0
    reconstruction_mse([0.0, 0.0], [1.0, -1.0]) ->  1.0

    Разная длина — ValueError: молчаливое обрезание по zip спрячет баг,
    когда декодер вернул не столько отсчётов, сколько было на входе.
    """
    raise NotImplementedError


def codec_cost(seconds, frame_rate_hz, n_codebooks, codebook_size):
    """Цена клипа в кодеке: сколько фреймов, токенов и бит в секунду.

    Вернуть словарь с ключами "frames", "tokens", "bitrate_bps".

    codec_cost(10, 12.5, 8, 1024)
        ->  {"frames": 125.0, "tokens": 1000.0, "bitrate_bps": 1000.0}
    codec_cost(1, 75.0, 8, 1024)
        ->  {"frames": 75.0, "tokens": 600.0, "bitrate_bps": 6000.0}

    frames = seconds * frame_rate_hz
    tokens = frames * n_codebooks          — длина последовательности для LM
    bitrate_bps = frame_rate_hz * n_codebooks * log2(codebook_size)

    Ради этого числа и придумали Mimi: 10 секунд речи — 1000 токенов,
    трансформер такой контекст не замечает. У EnCodec-24k на 75 Hz те же
    10 секунд дали бы 6000 токенов.

    codebook_size < 2 — ValueError: один код не несёт ни бита.
    """
    raise NotImplementedError


def split_semantic_acoustic(frames):
    """Разрезать последовательность фреймов на semantic и acoustic части.

    frames — список фреймов, каждый фрейм — список кодов всех codebook'ов.

    split_semantic_acoustic([[5, 1, 2], [7, 3, 4]])  ->  ([5, 7], [[1, 2], [3, 4]])

    В Mimi codebook 0 дистиллирован из WavLM и несёт содержание — что
    сказано. Codebook'и 1..7 несут тембр, просодию, шум. Text-to-semantic
    модель предсказывает первый, acoustic-декодер — остальные; отсюда
    zero-shot клонирование голоса.

    Пустой фрейм — ValueError: semantic-кода взять неоткуда.
    """
    raise NotImplementedError
