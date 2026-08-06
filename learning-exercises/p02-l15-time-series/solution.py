"""
Временные ряды — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def difference(series):
    """Первые разности: список изменений между соседними значениями.

    difference([100, 102, 106, 112, 120])  ->  [2, 4, 6, 8]
    difference([5])                        ->  []

    Результат короче входа на 1 — у первого элемента нет предыдущего.
    Дифференцирование убирает тренд: линейный тренд превращается в константу,
    квадратичный — в линейный, и тогда разности берут дважды.
    """
    return [series[i] - series[i - 1] for i in range(1, len(series))]


def is_stationary(series, mean_tol=0.5, var_ratio=2.0):
    """Груба ли проверка стационарности: похожи ли половины ряда друг на друга.

    Ряд считается стационарным, если ОБА условия выполнены:
      * |среднее первой половины - среднее второй| <= mean_tol * std всего ряда;
      * отношение большей дисперсии половины к меньшей <= var_ratio.

    is_stationary([1, 2, 3, ..., 100])  ->  False  (среднее уезжает)
    is_stationary([7, 7, 7, 7])         ->  True   (ничего не меняется)

    Дисперсии считаем по всей выборке (делим на n). Ловушка: нулевая
    дисперсия у одной половины — делить на неё нельзя.

    Это не тест Дики-Фуллера, а его дешёвая замена: она ловит уехавшее
    среднее и разъехавшийся разброс, а именно из-за них модель, обученная
    на январе, систематически ошибается в феврале.
    """
    half = len(series) // 2
    first, second = series[:half], series[half:]

    def mean(xs):
        return sum(xs) / len(xs)

    def var(xs):
        m = mean(xs)
        return sum((x - m) ** 2 for x in xs) / len(xs)

    overall_std = var(series) ** 0.5
    if abs(mean(first) - mean(second)) > mean_tol * overall_std:
        return False
    v1, v2 = var(first), var(second)
    hi, lo = max(v1, v2), min(v1, v2)
    # lo == 0 сравнивать отношением нельзя: либо обе нулевые, либо разъехались
    if lo == 0:
        return hi == 0
    return hi / lo <= var_ratio


def autocorrelation(series, max_lag):
    """Автокорреляция для лагов 0..max_lag включительно. Список длины max_lag+1.

    autocorrelation([1, 2, 3, 4], 1)  ->  [1.0, 0.333...]
    Для ряда с периодом 7 максимум по ненулевым лагам будет на лаге 7.

    acf[k] = среднее произведение отклонений (x[i]-m) * (x[i+k]-m) по n-k парам,
    делённое на дисперсию всего ряда. Поэтому acf[0] всегда равен 1.

    Ловушка: константный ряд даёт нулевую дисперсию и деление на ноль —
    возвращай для него нули.

    ACF отвечает на вопрос "сколько лагов брать в признаки": берут столько,
    пока корреляция заметно отличается от нуля.
    """
    n = len(series)
    m = sum(series) / n
    var = sum((x - m) ** 2 for x in series) / n
    if var == 0:
        return [0.0] * (max_lag + 1)
    acf = []
    for k in range(max_lag + 1):
        # делим на n-k, а не на n: пар всего n-k
        cov = sum((series[i] - m) * (series[i + k] - m) for i in range(n - k)) / (n - k)
        acf.append(cov / var)
    return acf


def rolling_mean(series, window):
    """Скользящее среднее по окну window. Длина результата len(series)-window+1.

    rolling_mean([1, 2, 3, 4], 2)  ->  [1.5, 2.5, 3.5]
    rolling_mean([1, 2], 5)        ->  []   (окно длиннее ряда)

    Окно ЗАКАНЧИВАЕТСЯ на текущей точке, а не центрируется на ней. Центрировать
    нельзя: в момент t будущих значений ещё не существует, и признак,
    подсмотревший вперёд, даст отличный бэктест и провал в проде.
    """
    if window > len(series):
        return []
    # накопительная сумма: одно вычитание и одно сложение на шаг вместо
    # пересчёта всего окна, O(n) вместо O(n * window)
    out = []
    total = sum(series[:window])
    out.append(total / window)
    for i in range(window, len(series)):
        total += series[i] - series[i - window]
        out.append(total / window)
    return out


def make_lag_features(series, n_lags):
    """Превратить ряд в задачу обучения с учителем. Вернуть (X, y).

    make_lag_features([10, 12, 14, 13, 15], 2)
        ->  ([[10, 12], [12, 14], [14, 13]], [14, 13, 15])

    Строка X — это n_lags значений, идущих ПЕРЕД целевым, в хронологическом
    порядке (самое старое слева). y — само целевое значение.

    Главная ловушка урока: значение в момент t не имеет права попасть в
    признаки строки t. Такой признак даёт идеальный прогноз и абсолютно
    бесполезную модель. Первые n_lags значений ряда становятся только
    историей и своей строки не получают.
    """
    X, y = [], []
    for t in range(n_lags, len(series)):
        X.append(series[t - n_lags : t])
        y.append(series[t])
    return X, y


def time_split(X, y, test_size):
    """Хронологический сплит: последние test_size наблюдений — тест.

    time_split([[1], [2], [3], [4]], [1, 2, 3, 4], 1)
        ->  ([[1], [2], [3]], [[4]], [1, 2, 3], [4])

    Вернуть (X_train, X_test, y_train, y_test). Порядок НЕ перемешивается:
    случайное перемешивание отдаёт модели куски будущего и завышает метрику
    в разы. Если test_size не меньше длины выборки — ValueError.
    """
    if test_size <= 0 or test_size >= len(X):
        raise ValueError("test_size должен быть в диапазоне 1..len(X)-1")
    cut = len(X) - test_size
    return X[:cut], X[cut:], y[:cut], y[cut:]


def walk_forward_splits(n_samples, n_splits=5, min_train=50):
    """Разбиения walk-forward: список пар (индексы train, индексы test).

    walk_forward_splits(8, 2, 4)  ->  [([0,1,2,3], [4,5]), ([0,1,2,3,4,5], [6,7])]

    Обучающее окно расширяется: каждый фолд учится на всём, что было до него,
    и проверяется на следующем куске. Шаг = max(1, (n_samples-min_train)//n_splits).
    Фолды, у которых не осталось тестовых точек, просто не выдаются.
    Если min_train не меньше n_samples — ValueError.

    Инвариант, который стоит держать в голове: максимальный индекс train
    всегда меньше минимального индекса test. Обычный k-fold это нарушает.
    """
    if min_train >= n_samples:
        raise ValueError("min_train должен быть меньше n_samples")
    step = max(1, (n_samples - min_train) // n_splits)
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * step
        if train_end >= n_samples:
            break
        test_end = min(train_end + step, n_samples)
        splits.append((list(range(train_end)), list(range(train_end, test_end))))
    return splits


def seasonal_naive_forecast(series, period, horizon):
    """Базовый прогноз: повторить последний полный период.

    seasonal_naive_forecast([1, 2, 3, 4, 5, 6], 3, 4)  ->  [4, 5, 6, 4]
    seasonal_naive_forecast([1, 2, 3], 1, 2)           ->  [3, 3]

    period=1 — это persistence, "завтра будет как сегодня". Если period
    больше длины ряда или меньше 1 — ValueError.

    Смысл: пока модель не обыгрывает этот прогноз, она не выучила ничего
    сверх календаря. Проигрыш сезонному наиву — почти всегда баг, чаще
    всего утечка будущего в признаках.
    """
    if period < 1 or period > len(series):
        raise ValueError("period должен быть в диапазоне 1..len(series)")
    tail = series[len(series) - period :]
    # h-й шаг вперёд берёт то же место в цикле, что и period шагов назад
    return [tail[h % period] for h in range(horizon)]
