"""
Janus-Pro: раздельные энкодеры для понимания и генерации — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Тезис урока в одну строку: один визуальный энкодер не может быть хорош
одновременно для understanding и для generation, потому что эти задачи
требуют противоположного. Понимание хочет семантики — чтобы две разные
фотографии кота лежали рядом. Генерация хочет реконструкции — чтобы код
разворачивался обратно в чёткие пиксели.

Здесь этот тезис не пересказывается, а измеряется. Два числа:

  * reconstruction_error — насколько точно codebook восстанавливает вектор;
  * semantic_margin      — насколько сильно представление группирует
                           объекты одного класса.

Codebook из классовых средних выигрывает по margin и проигрывает по error.
Codebook из самих векторов — ровно наоборот. Отсюда и решение Janus-Pro:
перестать выбирать и завести два энкодера при одном общем transformer body.

Только стандартная библиотека.
"""

import math

# Ключевые слова роутера. На уровне модуля, а не внутри функции: тесты
# импортируют их и проверяют, что классификация опирается именно на них.
UNDERSTAND_WORDS = ("describe", "what", "why", "caption", "explain", "how many")
GENERATE_WORDS = ("draw", "generate", "sketch", "render", "create", "paint")

# Маршруты Janus-Pro: какой энкодер обслуживает какую задачу.
ENCODERS = {
    "understand": "siglip",
    "generate": "vq",
    "ambiguous": "both",
}


def cosine_similarity(a, b):
    """Косинус угла между векторами: мера «про одно ли это».

    cosine_similarity([1.0, 0.0], [0.0, 1.0])  ->  0.0
    cosine_similarity([1.0, 0.0], [3.0, 0.0])  ->  1.0
    cosine_similarity([1.0, 0.0], [-1.0, 0.0]) -> -1.0

    Именно эту метрику оптимизирует CLIP/SigLIP-претрейн, поэтому и
    семантику мы меряем ею же.

    Ловушка: нулевой вектор не имеет направления, косинус для него не
    определён. Бросай ValueError, а не возвращай 0.0 — иначе ошибка в
    данных бесшумно превратится в «эти картинки не похожи».
    """
    if len(a) != len(b):
        raise ValueError(f"разная длина: {len(a)} и {len(b)}")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        raise ValueError("косинус с нулевым вектором не определён")
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def nearest_code(vector, codebook):
    """Индекс ближайшей записи codebook по евклидову расстоянию. Это VQ-энкодер.

    nearest_code([0.9, 0.1], [[1.0, 0.0], [0.0, 1.0]])  ->  0
    nearest_code([0.5, 0.5], [[1.0, 0.0], [0.0, 1.0]])  ->  0   (ничья -> меньший индекс)

    Расстояние сравниваем в квадрате: корень монотонен, а значит на порядок
    не влияет и считать его незачем.

    Ничья решается в пользу меньшего индекса — иначе один и тот же вектор
    квантовался бы по-разному от запуска к запуску.

    Пустой codebook — ValueError.
    """
    if not codebook:
        raise ValueError("пустой codebook: квантовать не во что")
    best_i, best_d = None, None
    for i, code in enumerate(codebook):
        if len(code) != len(vector):
            raise ValueError(f"размерность кода {i} не совпадает с вектором")
        d = sum((x - y) ** 2 for x, y in zip(vector, code))
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def vq_encode(vectors, codebook):
    """Список векторов -> список индексов codebook. Картинка становится токенами.

    vq_encode([[0.9, 0.1], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [0, 1]

    Ровно то, что делает VQ-токенизатор перед тем, как отдать картинку
    в transformer: непрерывные патчи превращаются в целые id.
    """
    return [nearest_code(v, codebook) for v in vectors]


def vq_reconstruct(indices, codebook):
    """Индексы -> векторы codebook. Это VQ-декодер.

    vq_reconstruct([1, 0], [[1.0, 0.0], [0.0, 1.0]])  ->  [[0.0, 1.0], [1.0, 0.0]]

    Возвращаем КОПИИ строк codebook, а не сами строки: иначе правка
    восстановленной картинки задним числом испортит сам codebook.

    Индекс вне диапазона — ValueError.
    """
    out = []
    for i in indices:
        if not 0 <= i < len(codebook):
            raise ValueError(f"индекс {i} вне codebook размера {len(codebook)}")
        out.append(list(codebook[i]))
    return out


def reconstruction_error(vectors, codebook):
    """Средний квадрат ошибки round-trip encode -> decode. Метрика генерации.

    reconstruction_error([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  0.0
    reconstruction_error([[1.0, 1.0]], [[1.0, 0.0]])              ->  0.5

    Разбор второго примера: ближайший код [1, 0], ошибки 0 и 1,
    (0 + 1) / 2 = 0.5.

    Делим на размерность, чтобы число не зависело от неё и картинки разного
    разрешения можно было сравнивать.

    Пустой список векторов — ValueError.
    """
    if not vectors:
        raise ValueError("нечего реконструировать: пустой список векторов")
    codes = vq_encode(vectors, codebook)
    restored = vq_reconstruct(codes, codebook)
    total = 0.0
    for original, approx in zip(vectors, restored):
        total += sum((x - y) ** 2 for x, y in zip(original, approx)) / len(original)
    return total / len(vectors)


def semantic_margin(vectors, labels):
    """Насколько представление группирует свой класс. Метрика понимания.

    Средний косинус внутри классов МИНУС средний косинус между классами.

    semantic_margin([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                    ["a", "a", "b", "b"])              ->  1.0

    Единица — идеал: свои совпадают, чужие ортогональны. Ноль означает, что
    представление про классы не знает ничего.

    Считаем по всем НЕУПОРЯДОЧЕННЫМ парам i < j: пара (i, j) и (j, i) — это
    один и тот же косинус, и учитывать его дважды незачем.

    Ловушка: если все объекты одного класса, межклассовых пар нет и разность
    не определена. ValueError.
    """
    if len(vectors) != len(labels):
        raise ValueError(f"разная длина: {len(vectors)} и {len(labels)}")
    same, other = [], []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sim = cosine_similarity(vectors[i], vectors[j])
            (same if labels[i] == labels[j] else other).append(sim)
    if not same or not other:
        raise ValueError("нужны и внутриклассовые, и межклассовые пары")
    return sum(same) / len(same) - sum(other) / len(other)


def route(prompt):
    """Классификация запроса: 'understand', 'generate' или 'ambiguous'.

    route("Describe what's in this image")   ->  'understand'
    route("Render a cyberpunk city")         ->  'generate'
    route("Sketch a cat and describe it")    ->  'ambiguous'
    route("hello there")                     ->  'ambiguous'

    Считаем, сколько слов из UNDERSTAND_WORDS и GENERATE_WORDS встретилось в
    запросе (регистр не важен), и берём сторону с большим счётом.

    Ничья — 'ambiguous', причём и «оба сразу», и «ни одного». Второе выглядит
    странно, но честно: роутер, который по умолчанию угадывает, молча уводит
    половину трафика в неправильный энкодер, и это никак не диагностируется.
    """
    low = prompt.lower()
    u = sum(1 for w in UNDERSTAND_WORDS if w in low)
    g = sum(1 for w in GENERATE_WORDS if w in low)
    if g > u:
        return "generate"
    if u > g:
        return "understand"
    return "ambiguous"


def encoder_for(task):
    """Какой энкодер обслуживает задачу. Здесь и живёт всё «разделение».

    encoder_for("understand")  ->  'siglip'
    encoder_for("generate")    ->  'vq'
    encoder_for("ambiguous")   ->  'both'

    В Chameleon, Show-o и Transfusion эта функция вернула бы одно и то же на
    любой вход — там токенизатор один. У Janus-Pro ответы разные, и в этом
    вся архитектурная идея.

    Незнакомая задача — ValueError, а не тихий 'siglip' по умолчанию.
    """
    if task not in ENCODERS:
        raise ValueError(f"неизвестная задача {task!r}")
    return ENCODERS[task]
