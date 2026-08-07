"""
Janus-Pro: раздельные энкодеры для понимания и генерации

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l15-janus-pro-decoupled-encoders
Разбор:  /check-code p12-l15-janus-pro-decoupled-encoders
"""

import math

UNDERSTAND_WORDS = ("describe", "what", "why", "caption", "explain", "how many")
GENERATE_WORDS = ("draw", "generate", "sketch", "render", "create", "paint")
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
    raise NotImplementedError


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
    raise NotImplementedError


def vq_encode(vectors, codebook):
    """Список векторов -> список индексов codebook. Картинка становится токенами.

    vq_encode([[0.9, 0.1], [0.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [0, 1]

    Ровно то, что делает VQ-токенизатор перед тем, как отдать картинку
    в transformer: непрерывные патчи превращаются в целые id.
    """
    raise NotImplementedError


def vq_reconstruct(indices, codebook):
    """Индексы -> векторы codebook. Это VQ-декодер.

    vq_reconstruct([1, 0], [[1.0, 0.0], [0.0, 1.0]])  ->  [[0.0, 1.0], [1.0, 0.0]]

    Возвращаем КОПИИ строк codebook, а не сами строки: иначе правка
    восстановленной картинки задним числом испортит сам codebook.

    Индекс вне диапазона — ValueError.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
