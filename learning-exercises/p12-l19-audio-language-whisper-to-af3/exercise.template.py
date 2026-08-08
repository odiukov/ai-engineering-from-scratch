"""
Audio-language модели: от Whisper до Audio Flamingo 3

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l19-audio-language-whisper-to-af3
Разбор:  /check-code p12-l19-audio-language-whisper-to-af3
"""

import math

LEXICAL_NEEDS = frozenset({
    "transcription",
    "keywords",
    "summarization",
    "translation",
})
ACOUSTIC_NEEDS = frozenset({
    "emotion",
    "music",
    "speaker_id",
    "environment",
    "temporal_grounding",
    "deepfake",
})


def hz_to_mel(f):
    """Перевод частоты из герц в мелы (шкала восприятия высоты звука).

    hz_to_mel(0)     ->  0.0
    hz_to_mel(700)   ->  781.18   (2595 * log10(2))

    Формула (та же, что в librosa.hz_to_mel(..., htk=True)):
        mel = 2595 * log10(1 + f / 700)

    Смысл: ухо различает 100 Гц и 200 Гц гораздо лучше, чем 7000 Гц и
    7100 Гц. Мел-шкала растягивает низы и сжимает верха, поэтому
    мел-фильтры внизу узкие, а наверху широкие.
    """
    raise NotImplementedError


def mel_to_hz(m):
    """Обратное преобразование: из мелов в герцы.

    mel_to_hz(0)        ->  0.0
    mel_to_hz(781.18)   ->  700.0

    Формула: f = 700 * (10^(m / 2595) - 1).

    Нужно, чтобы разложить диапазон 0..sr/2 на РАВНЫЕ куски в мелах, а
    потом вернуться в герцы и получить неравные куски по частоте.
    """
    raise NotImplementedError


def mel_filterbank(n_bins, n_mels, sr):
    """Банк из n_mels треугольных мел-фильтров над n_bins спектральными отсчётами.

    Возвращает матрицу n_mels x n_bins неотрицательных весов.

    len(mel_filterbank(64, 20, 16000))      ->  20
    len(mel_filterbank(64, 20, 16000)[0])   ->  64

    Считаем, что отсчёт k покрывает частоту k * (sr / 2) / (n_bins - 1),
    то есть n_bins отсчётов равномерно закрывают 0..Найквист (так ведёт
    себя rfft длины 2 * (n_bins - 1)).

    Как строится фильтр m:
      1. взять n_mels + 2 точки, РАВНОМЕРНО разложенных в мелах от
         hz_to_mel(0) до hz_to_mel(sr / 2);
      2. вернуть их в герцы, а герцы — в дробный индекс отсчёта;
      3. фильтр m — треугольник с нулями в точках m и m+2 и вершиной 1.0
         в точке m+1.

    Ловушка: точки дробные, делить на (center - left) можно только если
    оно не ноль. При маленьком n_bins соседние точки схлопываются.

    Соответствует librosa.filters.mel — только там это одна строка.
    """
    raise NotImplementedError


def frame_signal(x, sr, win_ms=25, hop_ms=10):
    """Нарезка сигнала на перекрывающиеся окна: список кадров одинаковой длины.

    frame_signal([0.0] * 16000, 16000)          ->  98 кадров по 400 отсчётов
    frame_signal([0.0] * 100, 16000)            ->  []   (короче одного окна)

    win = int(sr * win_ms / 1000), hop = int(sr * hop_ms / 1000). Кадр
    берётся, только пока он целиком помещается в сигнал; хвост короче окна
    выбрасывается.

    Ловушка: 30 секунд при 16 кГц дают 2998 кадров, а не 3000. Whisper
    печатает 3000, потому что дополняет вход нулями до ровных 30 секунд —
    padding, а не другая формула.
    """
    raise NotImplementedError


def dft_magnitude(frame, n_bins):
    """Модули n_bins отсчётов спектра от DC до частоты Найквиста.

    dft_magnitude([1.0, 1.0, 1.0, 1.0], 3)  ->  [4.0, 0.0, 0.0]

    Отсчёт k соответствует нормализованной частоте k / (2 * (n_bins - 1))
    цикла на отсчёт. Так n_bins точек равномерно покрывают весь односторонний
    rfft-спектр, включая Найквист, а не берут первые n_bins точек полного DFT.
    Когда n_bins = len(frame) // 2 + 1, это обычные бины rfft. Считаем их руками
    за O(n_bins * N) вместо O(N log N).

    Ловушка: модуль, а не действительная часть. Сдвиг сигнала во времени
    меняет фазу, но не модуль — на этом и держится вся спектрограмма.
    """
    raise NotImplementedError


def log_mel_spectrogram(x, sr, n_mels=20, n_bins=64, win_ms=25, hop_ms=10):
    """Полная log-Mel спектрограмма: список из T строк по n_mels чисел.

    Конвейер урока целиком:
      кадры -> модуль ДПФ -> мел-фильтры -> log(1 + x).

    len(log_mel_spectrogram([0.0] * 400, 8000, n_mels=4, n_bins=16))  ->  3

    Вход у Whisper и у AF3 выглядит именно так: (T, 80) для 80 мел-полос.

    Ловушка: банк фильтров зависит только от (n_bins, n_mels, sr), а не от
    кадра. Считать его внутри цикла — это T лишних построений банка.
    """
    raise NotImplementedError


def qformer_attend(queries, frames):
    """Audio Q-former: N обучаемых запросов кросс-аттендятся к кадрам энкодера.

    Возвращает len(queries) векторов той же размерности, что и кадры.

    qformer_attend([[1.0, 0.0]], [[3.0, 3.0], [3.0, 3.0]])  ->  [[3.0, 3.0]]

    Для каждого запроса q:
      scores[j] = dot(q, frames[j]) / sqrt(d)
      weights   = softmax(scores)
      out       = sum_j weights[j] * frames[j]

    Здесь K = V = frames без матриц проекции — настоящий Q-former добавил бы
    W_q, W_k, W_v и несколько голов; это torch.nn.MultiheadAttention с
    одной головой и единичными проекциями.

    Ловушка: softmax считается через вычитание максимума. Наивный
    math.exp(score) на больших числах даёт OverflowError, а именно большие
    скоры и получаются, когда запрос уже научился на что-то смотреть.

    Смысл: 3000 кадров тридцатисекундного клипа сжимаются в 64 токена,
    которые влезают в контекст LLM.
    """
    raise NotImplementedError


def pick_pipeline(needs):
    """Выбор пайплайна по списку требований: "cascaded" или "end-to-end".

    pick_pipeline(["transcription", "summarization"])  ->  "cascaded"
    pick_pipeline(["transcription", "emotion"])        ->  "end-to-end"
    pick_pipeline([])                                  ->  "cascaded"

    Правило урока: cascaded (Whisper транскрибирует, LLM рассуждает по
    тексту) годится, пока весь нужный сигнал лежит в словах. Хоть одно
    требование из ACOUSTIC_NEEDS — и нужен end-to-end audio-LLM.

    Неизвестное требование — ValueError: молча вернуть "cascaded" на
    опечатке "emotoin" значит потерять всю акустику в проде.
    """
    raise NotImplementedError
