"""
Whisper: аудио как последовательность кадров — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


FRAME_SIZE = 400  # 25 мс при 16 кГц — ширина кадра Whisper по умолчанию


def sine_wave(freq, duration_s, sample_rate=16000):
    """Синусоида заданной частоты: список отсчётов длиной duration_s секунд.

    len(sine_wave(440, 1.0))       ->  16000   (16 кГц, ровно как у Whisper)
    sine_wave(440, 1.0)[0]         ->  0.0     (sin 0 = 0)

    Отсчёт i это sin(2 * pi * freq * i / sample_rate). Число отсчётов —
    int(duration_s * sample_rate), дробный хвост отбрасывается.

    Зачем: это самый простой сигнал, на котором видно всю дальнейшую
    арифметику кадров. Настоящий Whisper ест ровно такой же список чисел,
    только в нём записана речь, а не чистый тон.
    """
    n = int(duration_s * sample_rate)
    return [math.sin(2 * math.pi * freq * i / sample_rate) for i in range(n)]


def n_frames(n_samples, frame_size=FRAME_SIZE, hop=160):
    """Сколько окон нарежется из сигнала. Кадры не выходят за конец сигнала.

    n_frames(16000)  ->  98     (1 секунда при 16 кГц — «около 100 кадров»)
    n_frames(400)    ->  1      (ровно на одно окно)
    n_frames(399)    ->  0      (на окно не хватило)

    Формула: 1 + (n_samples - frame_size) // hop, а если сигнал короче
    окна — ноль. Классическая ошибка — посчитать n_samples // hop: это даст
    100 вместо 98, потому что последние два окна вылезли бы за край.

    Whisper: frame_size = 400 (25 мс), hop = 160 (10 мс). Тридцать секунд
    дают 2998 кадров, а на вход энкодера подаётся ровно 3000 — разницу
    добирают паддингом, см. pad_or_clip.
    """
    if n_samples < frame_size:
        return 0
    return 1 + (n_samples - frame_size) // hop


def frame_signal(x, frame_size=FRAME_SIZE, hop=160):
    """Нарезать сигнал на перекрывающиеся окна.

    frame_signal([1, 2, 3, 4, 5], frame_size=3, hop=1)
        ->  [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

    Все кадры полные: неполный хвост отбрасывается, а не добивается нулями.
    Соседние кадры перекрываются на frame_size - hop отсчётов — при
    настройках Whisper это 240 из 400, то есть 60% каждого окна повторяется
    в следующем. Перекрытие нужно, чтобы спектр не «мигал» на границах.

    Длина результата обязана совпадать с n_frames для тех же аргументов.
    """
    return [
        list(x[start:start + frame_size])
        for start in range(0, len(x) - frame_size + 1, hop)
    ]


def frame_energy(frame):
    """Логарифм энергии кадра — заглушка вместо мел-полос.

    frame_energy([0.0, 0.0])   ->  -20.72...  (log 1e-9, тишина)
    frame_energy([1.0, 1.0])   ->  0.693...   (log 2)

    Энергия это сумма квадратов отсчётов. Логарифм берётся от energy + 1e-9:
    без этой добавки тишина даёт log(0) и весь пайплайн падает на -inf.

    Настоящий Whisper вместо одного числа считает 80 мел-полос через FFT,
    то есть librosa.feature.melspectrogram(n_mels=80) плюс логарифм. Форма
    данных при этом та же: одно число (у нас) или 80 (у него) на кадр, и
    именно эту матрицу «частота x время» видит энкодер.
    """
    energy = sum(v * v for v in frame)
    return math.log(energy + 1e-9)


def pad_or_clip(frames, target_frames):
    """Привести число кадров ровно к target_frames: добить тишиной или срезать.

    pad_or_clip([[1.0], [2.0]], 4)  ->  [[1.0], [2.0], [0.0], [0.0]]
    pad_or_clip([[1.0], [2.0]], 1)  ->  [[1.0]]
    len(pad_or_clip([], 1)[0])      ->  400

    Whisper всегда работает окном ровно 30 секунд = 3000 кадров, независимо
    от длины записи. Короче — паддинг тишиной, длиннее — режется, и всё, что
    не влезло, требует отдельного чанкинга снаружи модели.

    Добивочный кадр — нули той же ширины, что у существующих кадров. Когда
    аудио пустое и ширину подсмотреть неоткуда, берём FRAME_SIZE = 400:
    пустой аудиоклип всё равно должен стать корректной матрицей Whisper.
    Входной список не портим: возвращаем новый.
    """
    if target_frames < 0:
        raise ValueError("target_frames must not be negative")
    if len(frames) >= target_frames:
        return [list(f) for f in frames[:target_frames]]
    width = len(frames[0]) if frames else FRAME_SIZE
    padded = [list(f) for f in frames]
    padded.extend([0.0] * width for _ in range(target_frames - len(frames)))
    return padded


def conv_stem_length(n_in, kernel=3, stride=2, padding=1):
    """Длина последовательности после одного слоя Conv1D.

    conv_stem_length(3000, stride=1)  ->  3000   (первый слой длину хранит)
    conv_stem_length(3000, stride=2)  ->  1500   (второй слой её делит)

    Формула: (n_in + 2 * padding - kernel) // stride + 1.

    Свёрточный «стем» Whisper это два слоя с ядром 3: первый со шагом 1,
    второй со шагом 2. Вместе они превращают 3000 кадров в 1500 токенов
    энкодера. Смысл: attention стоит квадрат от длины, и деление длины
    вдвое экономит вчетверо, а параметров добавляет всего ничего.

    Осторожно: ДВА слоя со шагом 2 дали бы 750, а не 1500.
    stride < 1 — ValueError.
    """
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return (n_in + 2 * padding - kernel) // stride + 1


def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    """Собрать префикс декодера, который управляет поведением Whisper.

    whisper_prompt()                        ->  ["<|startoftranscript|>",
                                                 "<|en|>", "<|transcribe|>"]
    whisper_prompt("fr", "translate", False) ->  ["<|startoftranscript|>",
                                                 "<|fr|>", "<|translate|>",
                                                 "<|notimestamps|>"]

    Порядок токенов не произволен: модель обучалась именно на этой
    последовательности, и перестановка ломает предсказание.

    Токен <|notimestamps|> ДОБАВЛЯЕТСЯ, когда таймкоды НЕ нужны — логика
    обратная имени флага, на этом легко ошибиться.

    Это вся поверхность управления моделью: три-четыре токена вместо
    отдельной модели на каждую пару язык-задача. Задача бывает только
    transcribe (в язык записи) или translate (в английский) — иначе
    ValueError.
    """
    if task not in ("transcribe", "translate"):
        raise ValueError("task must be 'transcribe' or 'translate'")
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens


def parse_whisper_prompt(tokens):
    """Разобрать префикс обратно в словарь настроек.

    parse_whisper_prompt(["<|startoftranscript|>", "<|de|>", "<|translate|>"])
        ->  {"lang": "de", "task": "translate", "timestamps": True}

    Обратна whisper_prompt: разбор собранного префикса обязан вернуть те же
    аргументы, с которыми его собирали.

    Зачем это в жизни: логи и датасеты для дообучения Whisper хранят
    префикс строкой, и его приходится читать обратно, чтобы понять, на
    какой задаче модель ошиблась.

    Префикс без <|startoftranscript|> или короче трёх токенов — ValueError.
    """
    if len(tokens) < 3 or tokens[0] != "<|startoftranscript|>":
        raise ValueError("prompt must start with <|startoftranscript|>")
    inner = [t[2:-2] for t in tokens]  # снимаем обёртку <| |>
    return {
        "lang": inner[1],
        "task": inner[2],
        "timestamps": "notimestamps" not in inner,
    }
