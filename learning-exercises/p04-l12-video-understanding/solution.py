"""
Понимание видео — моделирование времени — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def sample_uniform(num_frames_total, T):
    """Равномерно выбрать T кадров по всему клипу. Список индексов длины T.

    sample_uniform(10, 5)  ->  [0, 2, 4, 6, 8]
    sample_uniform(3, 5)   ->  [0, 1, 2, 2, 2]   (кадров меньше, чем нужно)
    sample_uniform(300, 3) ->  [0, 100, 200]

    Шаг дробный: step = num_frames_total / T, индекс = int(i * step).
    Целочисленное деление здесь испортит равномерность на длинных клипах.

    Ловушка: коротких видео в датасете всегда больше, чем кажется. Когда
    кадров меньше T, добери список повторами ПОСЛЕДНЕГО кадра — длина обязана
    быть ровно T, иначе батч не соберётся.

    Это дефолт для схемы 2D+pool: важно покрыть весь клип, а не соседние кадры.
    """
    if num_frames_total <= T:
        pad = [num_frames_total - 1] * (T - num_frames_total)
        return list(range(num_frames_total)) + pad
    step = num_frames_total / T
    return [int(i * step) for i in range(T)]


def sample_dense(num_frames_total, T, rng):
    """Случайное непрерывное окно из T подряд идущих кадров.

    rng — экземпляр random.Random. Глобальный random брать нельзя: выборка
    обязана быть воспроизводимой по сиду.

    sample_dense(10, 3, random.Random(0))  ->  три ПОДРЯД идущих индекса
    sample_dense(3, 5, random.Random(0))   ->  [0, 1, 2, 2, 2]

    Начало берётся из диапазона [0, num_frames_total - T] включительно —
    правый конец тоже валиден, иначе последние кадры видео никогда не
    попадут в обучение.

    Зачем подряд: 3D-свёртке нужны соседние кадры, иначе движения между ними
    просто нет — прыжок через 100 кадров это уже смена сцены, а не движение.
    """
    if num_frames_total <= T:
        pad = [num_frames_total - 1] * (T - num_frames_total)
        return list(range(num_frames_total)) + pad
    start = rng.randint(0, num_frames_total - T)
    return list(range(start, start + T))


def multi_clip_indices(num_frames_total, T, num_clips):
    """Несколько окон по T кадров, равномерно разложенных по всему видео.

    Возвращает список из num_clips списков индексов.

    multi_clip_indices(10, 4, 2)  ->  [[0, 1, 2, 3], [6, 7, 8, 9]]
    multi_clip_indices(10, 4, 1)  ->  [[3, 4, 5, 6]]   (одно окно — по центру)
    multi_clip_indices(3, 4, 2)   ->  [[0, 1, 2, 2], [0, 1, 2, 2]]

    Первое окно начинается с нуля, последнее упирается в конец: старты берутся
    как i * (num_frames_total - T) // (num_clips - 1).

    Так считают video-level accuracy: классифицируют каждое окно и усредняют
    предсказания. Цифра выходит выше и стабильнее, чем clip-level по одному
    окну, и оба числа принято показывать рядом.
    """
    if num_frames_total <= T:
        return [sample_uniform(num_frames_total, T) for _ in range(num_clips)]
    span = num_frames_total - T
    if num_clips == 1:
        return [list(range(span // 2, span // 2 + T))]
    clips = []
    for i in range(num_clips):
        start = i * span // (num_clips - 1)
        clips.append(list(range(start, start + T)))
    return clips


def temporal_mean_pool(frame_features):
    """Усреднить эмбеддинги кадров по времени. Один вектор на клип.

    frame_features — список векторов одинаковой длины (по одному на кадр).

    temporal_mean_pool([[1.0, 2.0], [3.0, 4.0]])  ->  [2.0, 3.0]
    temporal_mean_pool([[5.0]])                   ->  [5.0]

    Это весь «pool» в схеме 2D+pool: прогнали ResNet по каждому кадру,
    усреднили, отдали в классификатор.

    И ровно здесь её потолок: среднее не зависит от порядка кадров. «Открыть
    дверь» и «закрыть дверь» дают один и тот же вектор. Всё, что определяется
    движением, такой моделью не решается в принципе.
    """
    n = len(frame_features)
    dim = len(frame_features[0])
    return [sum(f[i] for f in frame_features) / n for i in range(dim)]


def temporal_conv(signal, kernel):
    """Свёртка одного признака вдоль времени. Режим valid, без паддинга.

        out[i] = sum_j signal[i + j] * kernel[j]

    Длина результата: len(signal) - len(kernel) + 1.

    temporal_conv([1.0, 2.0, 3.0], [1.0])            ->  [1.0, 2.0, 3.0]
    temporal_conv([0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]) ->  [2.0]
    temporal_conv([2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]) ->  [-2.0]

    Как и в свёрточных сетях, ядро НЕ переворачивается: это корреляция,
    хотя все зовут её свёрткой.

    Обрати внимание на последние два примера: та же последовательность
    задом наперёд дала противоположный знак. Вот чего среднее по времени
    не умеет — различать направление движения.
    """
    k = len(kernel)
    out_len = len(signal) - k + 1
    return [
        sum(signal[i + j] * kernel[j] for j in range(k)) for i in range(out_len)
    ]


def top_k_accuracy(scores, labels, k=1):
    """Доля примеров, где верный класс попал в топ-k по скорам.

    scores — список списков (скоры по классам для каждого примера),
    labels — список номеров верных классов.

    top_k_accuracy([[0.1, 0.9]], [1])              ->  1.0
    top_k_accuracy([[0.1, 0.9]], [0])              ->  0.0
    top_k_accuracy([[0.5, 0.3, 0.2]], [1], k=2)    ->  1.0

    Не нужно сортировать весь список: достаточно посчитать, сколько классов
    имеют скор строго больше верного. Если таких меньше k — попали в топ-k.
    Заодно это честно разруливает совпадающие скоры.

    Считают её дважды: clip-level (одно окно на видео) и video-level
    (усреднённые предсказания по нескольким окнам). Большой разрыв между
    ними означает, что модель держится на усреднении, а не на кадрах.
    """
    hits = 0
    for row, label in zip(scores, labels):
        better = sum(1 for s in row if s > row[label])
        if better < k:
            hits += 1
    return hits / len(labels)


def inflate_2d_to_3d(kernel2d, time_kernel=3):
    """Раздуть 2D-ядро в 3D по трюку I3D: повторить вдоль времени и поделить.

    kernel2d — список строк. Возвращает список из time_kernel одинаковых
    срезов, каждый — kernel2d, поделённый на time_kernel.

    inflate_2d_to_3d([[1.0]], 2)  ->  [[[0.5]], [[0.5]]]
    inflate_2d_to_3d([[4.0]], 1)  ->  [[[4.0]]]

    Деление на time_kernel — не косметика. Без него отклик на статичном
    видео вырастет в time_kernel раз, статистики batch norm поедут, и
    предобученные веса перестанут работать с первого же прохода.

    Смысл всего трюка: 3D-сеть стартует не со случайных весов, а с
    ImageNet-предобучения. Это и сделало I3D первой рабочей 3D-моделью.
    """
    return [[[v / time_kernel for v in row] for row in kernel2d] for _ in range(time_kernel)]


def conv2plus1d_mid_channels(in_c, out_c, k=3):
    """Сколько промежуточных каналов взять в (2+1)D-блоке. Целое число.

        M = (in_c * out_c * k^3) // (in_c * k^2 + out_c * k)

    conv2plus1d_mid_channels(3, 64, 3)   ->  23
    conv2plus1d_mid_channels(64, 64, 3)  ->  144

    Формула из статьи R(2+1)D подобрана так, чтобы у факторизованного блока
    было примерно столько же параметров, сколько у обычной 3D-свёртки
    in_c x out_c x k x k x k. Проверить легко:

        (2+1)D: in_c * M * k * k  +  M * out_c * k
        3D:     in_c * out_c * k * k * k

    Выигрыш не в размере, а в том, что между пространственной и временной
    частями появляется дополнительная нелинейность — та же ёмкость учится
    лучше.
    """
    return (in_c * out_c * k ** 3) // (in_c * k ** 2 + out_c * k)
