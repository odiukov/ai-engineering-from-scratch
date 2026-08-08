"""
Знакомство с JAX: функциональный стиль руками — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import random


def prng_key(seed):
    """Ключ генератора из целого seed. Аналог jax.random.PRNGKey.

    prng_key(0) == prng_key(0)  ->  True
    prng_key(0) != prng_key(1)  ->  True

    В JAX нет глобального состояния генератора: ключ — обычное значение,
    которое передаётся в функцию явно. Никакого random.seed(), никакого
    «кто-то до меня уже дёрнул генератор».

    Ключ — просто число. Вся его магия в том, что от него детерминированно
    выводятся и новые ключи (split_key), и сами числа (normal).
    """
    # константы линейного конгруэнтного генератора (PCG); годится любая
    # перемешивающая функция, лишь бы соседние seed давали далёкие ключи
    return (int(seed) * 6364136223846793005 + 1442695040888963407) % (1 << 64)


def split_key(key, n=2):
    """Разбить ключ на n новых, не пересекающихся с исходным. Аналог jax.random.split.

    len(split_key(prng_key(0), 3))  ->  3
    split_key(prng_key(0))[0] != split_key(prng_key(0))[1]  ->  True

    Возвращает кортеж. Один и тот же key всегда даёт один и тот же набор —
    иначе воспроизводимости не будет.

    Зачем: использовать один ключ дважды нельзя, иначе два «независимых»
    слоя получат одинаковые веса. Правило JAX — расщепил и раздал.
    """
    return tuple(
        ((key + (i + 1) * 1442695040888963407) * 6364136223846793005 + i) % (1 << 64)
        for i in range(n)
    )


def normal(key, n, scale=1.0):
    """n чисел из N(0, scale), выведенных из ключа. Аналог jax.random.normal.

    normal(prng_key(0), 3) == normal(prng_key(0), 3)  ->  True
    len(normal(prng_key(0), 5))                       ->  5

    Функция ЧИСТАЯ: тот же ключ — те же числа, сколько ни зови. Заводи
    локальный random.Random(key); глобальный random здесь всё сломает,
    потому что второй вызов вернёт другое.
    """
    rng = random.Random(key)
    return [rng.gauss(0.0, scale) for _ in range(n)]


def tree_map(fn, *trees):
    """Применить fn к каждому листу pytree, сохранив структуру. Аналог jax.tree.map.

    tree_map(lambda v: v * 2, [1.0, 2.0])            ->  [2.0, 4.0]
    tree_map(lambda a, b: a + b, [1.0], [10.0])      ->  [11.0]
    tree_map(lambda v: -v, {"w": [1.0], "b": 2.0})   ->  {'w': [-1.0], 'b': -2.0}

    pytree — это вложенные списки, кортежи и словари; листья — числа.
    Структура берётся из ПЕРВОГО дерева, остальные обходятся параллельно.
    Все деревья обязаны иметь одинаковые типы контейнеров, длины и ключи;
    несовпадение бросает ValueError вместо тихого отбрасывания листьев zip-ом.

    Ради этой одной функции JAX и обходится без метода .parameters():
    обновление всех весов сразу — это tree_map(lambda p, g: p - lr * g,
    params, grads), и работает оно с любой формой модели.
    """
    if not trees:
        raise ValueError("tree_map needs at least one pytree")

    first = trees[0]
    if isinstance(first, dict):
        if any(not isinstance(t, dict) or t.keys() != first.keys() for t in trees[1:]):
            raise ValueError("pytree structure mismatch: dictionary keys differ")
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in first}
    if isinstance(first, (list, tuple)):
        if any(type(t) is not type(first) or len(t) != len(first) for t in trees[1:]):
            raise ValueError("pytree structure mismatch: sequence shape differs")
        mapped = [tree_map(fn, *items) for items in zip(*trees)]
        return type(first)(mapped)
    if any(isinstance(t, (dict, list, tuple)) for t in trees[1:]):
        raise ValueError("pytree structure mismatch: leaf and container differ")
    return fn(*trees)


def grad(f, h=1e-5):
    """Из функции f(params) -> число сделать функцию params -> градиент.

    df = grad(lambda p: p[0] ** 2)
    df([3.0])  ->  [6.0]  (с точностью до 1e-6)

    df = grad(lambda p: p[0] * p[1])
    df([2.0, 5.0])  ->  [5.0, 2.0]

    params — плоский список чисел, результат — список той же длины.
    Считай ЦЕНТРАЛЬНОЙ разностью: (f(x+h) - f(x-h)) / (2h), её ошибка
    падает как h^2, а у односторонней — как h.

    Разница с настоящим jax.grad: там автодифференцирование, точное до
    последнего бита и стоящее один проход, здесь — численное, 2*len(params)
    проходов. Общее — сама идея: grad превращает функцию в функцию, ничего
    не записывая ни в какие тензоры.

    Копию params делай на каждой координате: править входной список
    запрещено, иначе следующая частная производная посчитается не в той точке.
    """
    def gradient(params):
        out = []
        for i in range(len(params)):
            up, down = list(params), list(params)
            up[i] += h
            down[i] -= h
            out.append((f(up) - f(down)) / (2 * h))
        return out

    return gradient


def value_and_grad(f, h=1e-5):
    """Функция, возвращающая пару (значение, градиент). Аналог jax.value_and_grad.

    vg = value_and_grad(lambda p: p[0] ** 2)
    vg([3.0])  ->  (9.0, [6.0])

    В настоящем JAX это дешевле, чем считать f и grad(f) по отдельности:
    значение достаётся из того же прямого прохода. Здесь выигрыш тот же по
    смыслу, хоть и скромнее по цене.
    """
    gradient = grad(f, h=h)

    def wrapped(params):
        return f(params), gradient(params)

    return wrapped


def vmap(f):
    """Из функции одного примера сделать функцию батча. Аналог jax.vmap.

    batched = vmap(lambda x: x * 2)
    batched([1.0, 2.0, 3.0])  ->  [2.0, 4.0, 6.0]

    batched = vmap(lambda row: sum(row))
    batched([[1.0, 2.0], [3.0, 4.0]])  ->  [3.0, 7.0]

    Ты пишешь функцию для ОДНОГО примера и не думаешь про батч-измерение.
    Настоящий vmap не крутит цикл, а перестраивает вычисление в
    векторизованное — отсюда и ускорение в десятки раз. Наш вариант
    честно возвращает список, но интерфейс тот же.
    """
    def batched(batch):
        return [f(item) for item in batch]

    return batched


def predict(params, x):
    """Линейная модель: dot(w, x) + b. Чистая функция, состояния нет.

    predict({"w": [2.0, 3.0], "b": 1.0}, [1.0, 1.0])  ->  6.0

    Веса приходят аргументом, а не лежат в self. В этом вся разница между
    JAX и PyTorch: тут нет объекта модели, есть функция и pytree чисел.
    """
    return sum(w * v for w, v in zip(params["w"], x)) + params["b"]


def mse(params, xs, ys):
    """Средний квадрат ошибки predict на батче. Чистая функция.

    mse({"w": [1.0], "b": 0.0}, [[1.0], [2.0]], [1.0, 2.0])  ->  0.0

    Батч прогоняется через vmap(...), а не через ручной цикл — ровно так
    это пишут в JAX.
    """
    predicted = vmap(lambda x: predict(params, x))(xs)
    return sum((p - y) ** 2 for p, y in zip(predicted, ys)) / len(ys)


def train_linear(key, xs, ys, steps=300, lr=0.1):
    """Обучить predict на (xs, ys). Вернуть НОВЫЙ pytree params.

    Начальные веса берутся из normal по ключу; на каждом шаге градиент
    считается через grad по плоскому списку [*w, b], а обновление — через
    tree_map(lambda p, g: p - lr * g, ...).

    key = prng_key(0)
    params = train_linear(key, [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0])
    abs(params["w"][0] - 2.0) < 0.01  ->  True
    abs(params["b"] - 1.0) < 0.01     ->  True

    Функция обязана быть ЧИСТОЙ по отношению к аргументам: xs и ys не
    трогаем, возвращаем новый словарь. В JAX иначе и не получится —
    массивы там неизменяемы, а тут это дисциплина, которую держишь ты.
    """
    n_features = len(xs[0])
    flat = normal(key, n_features + 1, scale=0.1)

    def loss(theta):
        return mse({"w": theta[:-1], "b": theta[-1]}, xs, ys)

    gradient = grad(loss)
    for _ in range(steps):
        flat = tree_map(lambda p, g: p - lr * g, flat, gradient(flat))

    return {"w": flat[:-1], "b": flat[-1]}
