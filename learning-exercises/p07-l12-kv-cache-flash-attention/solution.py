"""
KV-cache, Flash Attention и оптимизация инференса — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def softmax(scores):
    """Softmax, устойчивый к большим числам: вычти максимум перед exp.

    softmax([0.0, 0.0])        ->  [0.5, 0.5]
    softmax([1.0, 0.0, 0.0])   ->  [0.5761..., 0.2119..., 0.2119...]
    softmax([1000.0, 999.0])   ->  [0.7310..., 0.2689...]

    Ловушка: math.exp(1000) бросает OverflowError. Сдвиг на максимум ничего
    не меняет математически (exp(a-m)/sum exp(b-m) = exp(a)/sum exp(b)), но
    делает самый большой аргумент exp равным нулю.

    В attention логиты легко уходят за 100 на длинном контексте, поэтому
    ни одна реальная реализация softmax не считает exp «в лоб».
    """
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def attention_full(q, Ks, Vs):
    """Внимание одного запроса ко всем ключам сразу: softmax(qK^T/sqrt(d)) V.

    attention_full([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]], [[2.0], [4.0]])
        ->  [3.0]        (оба ключа одинаковы -> веса 0.5/0.5 -> среднее V)

    attention_full([10.0, 0.0], [[1.0, 0.0], [-1.0, 0.0]], [[2.0], [4.0]])
        ->  [2.0000...]  (первый ключ выигрывает с огромным отрывом)

    Делить на sqrt(len(q)) обязательно: без масштаба дисперсия скалярного
    произведения растёт с размерностью, softmax насыщается и градиент
    умирает. Это тот самый 1/sqrt(d_k) из «Attention Is All You Need».
    """
    scale = 1.0 / math.sqrt(len(q))
    scores = [sum(qi * ki for qi, ki in zip(q, k)) * scale for k in Ks]
    weights = softmax(scores)
    out = [0.0] * len(Vs[0])
    for w, v in zip(weights, Vs):
        for j in range(len(out)):
            out[j] += w * v[j]
    return out


def tiled_softmax_dot(q, Ks, Vs, tile=4):
    """То же внимание, но блоками по `tile` ключей — алгоритм Flash Attention.

    Ответ обязан совпасть с attention_full до ошибок округления:
    tiled_softmax_dot(q, Ks, Vs, tile=1) == tiled_softmax_dot(q, Ks, Vs, tile=99)

    Идея: не материализовать все N логитов. Держим бегущий максимум m,
    бегущую сумму s и незавершённый выход out. На новом блоке пересчитываем
    их так, будто максимум всегда был новым:
        new_m = max(m, максимум логитов блока)
        поправка = exp(m - new_m)     # во сколько раз «сжать» накопленное
        s   = s * поправка + сумма exp(логиты блока - new_m)
        out = out * поправка + сумма exp(...) * V блока
    В конце делим out на s.

    Ловушка: на первом блоке m = -inf, и exp(-inf - new_m) даёт 0.0 — но
    только если написать это руками. math.exp(float("-inf") - x) вернёт 0.0,
    а вот float("-inf") - float("-inf") — уже nan. Разбери первый блок явно.

    Зачем: рабочий набор здесь tile x d_head, а не N x d_head. Именно так
    Flash Attention помещает всё внимание в SRAM и не гоняет матрицу N x N
    в HBM. Ответ не приближённый — он точный.
    """
    d_v = len(Vs[0])
    scale = 1.0 / math.sqrt(len(q))
    running_max = float("-inf")
    running_sum = 0.0
    out = [0.0] * d_v
    for start in range(0, len(Ks), tile):
        k_block = Ks[start:start + tile]
        v_block = Vs[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) * scale for k in k_block]
        new_max = max(scores)
        if running_max > new_max:
            new_max = running_max
        # на первом блоке накопленного ещё нет, поправка ровно 0.0
        fix = 0.0 if running_max == float("-inf") else math.exp(running_max - new_max)
        exps = [math.exp(s - new_max) for s in scores]
        running_sum = running_sum * fix + sum(exps)
        for j in range(d_v):
            out[j] = out[j] * fix + sum(e * v[j] for e, v in zip(exps, v_block))
        running_max = new_max
    return [o / running_sum for o in out]


class KVCache:
    """Хранилище K и V всех уже обработанных токенов префикса.

    c = KVCache()
    c.append([1.0], [2.0])
    c.append([3.0], [4.0])
    len(c)      ->  2
    c.read()    ->  ([[1.0], [3.0]], [[2.0], [4.0]])

    Весь KV-cache — это два растущих списка. Никакой магии; магия в том,
    что благодаря им декодер перестаёт пересчитывать префикс на каждом шаге.
    """

    def __init__(self):
        """Пустой кэш: два пустых списка."""
        self.K = []
        self.V = []

    def append(self, k, v):
        """Дописать K и V одного нового токена в конец."""
        self.K.append(k)
        self.V.append(v)

    def read(self):
        """Вернуть кортеж (список ключей, список значений) целиком."""
        return self.K, self.V

    def __len__(self):
        """Сколько токенов уже лежит в кэше."""
        return len(self.K)


def decode_naive(states, project, queries):
    """Наивный декодер: на каждом шаге K и V пересчитываются для всего префикса.

    `project(state)` возвращает кортеж (k, v) для одного токена.
    Вернуть кортеж (список выходов, число вызовов project).

    Для 3 шагов project зовётся 1 + 2 + 3 = 6 раз.
    Для 100 шагов — 5050 раз. Это и есть O(N^2).

    Именно эта функция — «до». Выходы у неё правильные, но работа
    квадратичная: k и v префиксного токена не меняются никогда, а мы
    считаем их заново на каждом новом токене.
    """
    outputs = []
    calls = 0
    for t, q in enumerate(queries):
        Ks, Vs = [], []
        for state in states[:t + 1]:
            k, v = project(state)
            calls += 1
            Ks.append(k)
            Vs.append(v)
        outputs.append(attention_full(q, Ks, Vs))
    return outputs, calls


def decode_cached(states, project, queries):
    """Тот же декодер, но с KV-cache: project зовётся ровно один раз на токен.

    Вернуть кортеж (список выходов, число вызовов project).

    Для 3 шагов project зовётся 3 раза, для 100 — 100. Это O(N).

    Главное свойство, которое надо доказать тестом: выходы совпадают с
    decode_naive до последнего бита. KV-cache — не приближение и не
    компромисс по качеству, а вычёркивание повторной работы.
    """
    cache = KVCache()
    outputs = []
    calls = 0
    for state, q in zip(states, queries):
        k, v = project(state)
        calls += 1
        cache.append(k, v)
        Ks, Vs = cache.read()
        outputs.append(attention_full(q, Ks, Vs))
    return outputs, calls


def kv_cache_bytes(seq_len, n_layers, n_kv_heads, d_head, dtype_bytes=2):
    """Размер KV-cache в байтах: двойка за K и V, дальше просто произведение.

    kv_cache_bytes(1, 1, 1, 128, 2)            ->  512
    kv_cache_bytes(32768, 80, 8, 128, 2)       ->  10737418240   (~10.7 GB)

    dtype_bytes: 2 для fp16/bf16, 1 для fp8/int8, 4 для fp32.

    Ловушка: n_kv_heads — это число KV-голов, а не голов внимания. GQA с
    8 KV-головами против MHA с 64 даёт ровно восьмикратную экономию кэша,
    и это главная причина, почему длинный контекст в 2026 вообще посилен.
    """
    return 2 * seq_len * n_layers * n_kv_heads * d_head * dtype_bytes
