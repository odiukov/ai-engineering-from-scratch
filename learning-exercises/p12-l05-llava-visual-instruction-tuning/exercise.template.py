"""
LLaVA и visual instruction tuning

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l05-llava-visual-instruction-tuning
Разбор:  /check-code p12-l05-llava-visual-instruction-tuning
"""

import math

IMAGE_TOKEN = "<image>"
PATCH_TOKEN = "<patch>"
TILE = 336
PATCH = 14
GRID_CANDIDATES = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2))


def gelu(x):
    """GELU в tanh-приближении — активация между двумя слоями проектора.

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    gelu(0.0)   ->  0.0
    gelu(3.0)   ->  2.9964    почти сам x
    gelu(-1.0)  ->  -0.1588   ОТРИЦАТЕЛЬНОЕ, не ноль

    Главное отличие от ReLU видно на третьем примере: около нуля GELU
    проваливается в минус, а не отрезает. Из-за этого у неё есть градиент
    там, где ReLU уже мертва, и проектор LLaVA обучается за часы.

    Симпатичное тождество для самопроверки: gelu(x) - gelu(-x) == x.
    """
    raise NotImplementedError


def mlp_projector(patches, W1, b1, W2, b2):
    """Двухслойный MLP-проектор LLaVA: vit_dim -> hidden -> llm_dim.

    Каждый патч проходит linear -> gelu -> linear. Веса общие для всех
    патчей. Вернуть список токенов в размерности LLM.

    W1 — hidden строк длиной vit_dim, W2 — llm_dim строк длиной hidden.

    mlp_projector([[1.0]], [[1.0]], [0.0], [[2.0]], [0.0])  ->  [[1.6824]]

    Ключевое отличие от Q-Former: сколько патчей пришло, столько токенов и
    уйдёт в LLM. Никакого сжатия. 576 патчей ViT-L/14@336 — это 576 токенов
    контекста. Взамен LLM получает сырые патчи, а не чужой пересказ.

    Несовпадение размерностей — ValueError.
    """
    raise NotImplementedError


def projector_param_count(vit_dim, hidden_dim, llm_dim):
    """Сколько обучаемых параметров у двухслойного проектора (с bias).

    vit_dim*hidden_dim + hidden_dim + hidden_dim*llm_dim + llm_dim

    projector_param_count(1024, 4096, 4096)  ->  20_979_712   (около 21M)

    Сравни с 188M у Q-Former из урока 12.03 и с 7B у самой LLM. Проектор —
    это меньше трети процента модели, и именно поэтому первую стадию LLaVA
    (выравнивание на 558k подписей) гоняют за несколько часов.

    Любой неположительный аргумент — ValueError.
    """
    raise NotImplementedError


def build_llava_prompt(system, user):
    """Собрать промпт в формате LLaVA. Одна строка.

    "{system} USER: <image> {user} ASSISTANT:"

    build_llava_prompt("A chat.", "Describe this image.")
        ->  "A chat. USER: <image> Describe this image. ASSISTANT:"

    Плейсхолдер IMAGE_TOKEN стоит ПЕРЕД вопросом: LLM причинная, и токены
    вопроса должны видеть картинку, а не наоборот.

    Хвост "ASSISTANT:" без пробела в конце — это точка, с которой начинается
    генерация. Обучение шло ровно на таком формате; лишний пробел или
    перенос строки заметно портит качество.

    Пустой system или user — ValueError.
    """
    raise NotImplementedError


def expand_image_placeholder(prompt, image_tokens):
    """Заменить каждый <image> на image_tokens штук PATCH_TOKEN. Список слов.

    Промпт режется по пробелам; слово, равное IMAGE_TOKEN, разворачивается
    в image_tokens одинаковых PATCH_TOKEN, остальные слова идут как есть.

    expand_image_placeholder("USER: <image> hi", 3)
        ->  ["USER:", "<patch>", "<patch>", "<patch>", "hi"]

    Именно это происходит перед подачей в LLM: на месте одного плейсхолдера
    оказывается 576 (или 2880 при AnyRes) визуальных эмбеддингов. Токенизатор
    видит последовательность длиннее той, на которой учился базовый LLM, —
    и это нормально, стадия 1 научила его такое переваривать.

    Промпт без плейсхолдера — ValueError: картинка просто не доедет до модели,
    а модель всё равно что-нибудь ответит, и баг найдётся нескоро.
    image_tokens <= 0 — тоже ValueError.
    """
    raise NotImplementedError


def pick_anyres_grid(height, width, tile=TILE):
    """Выбрать раскладку плиток AnyRes под форму картинки. Кортеж (rows, cols).

    Перебираем GRID_CANDIDATES. Для каждой раскладки холст имеет размер
    (rows*tile, cols*tile). Вписываем картинку в холст с сохранением
    пропорций (scale = min(canvas_h/height, canvas_w/width)) и считаем:

      effective = min(площадь вписанной картинки, площадь оригинала)
      wasted    = площадь холста - effective

    Побеждает максимум effective; при равенстве — минимум wasted; при
    полном равенстве — кандидат, который встретился раньше.

    pick_anyres_grid(336, 336)    ->  (1, 1)
    pick_anyres_grid(672, 672)    ->  (2, 2)
    pick_anyres_grid(1344, 672)   ->  (2, 1)   высокая картинка -> два ряда

    Второй критерий не декоративный: без него квадратная картинка 336x336
    выбрала бы холст 2x2 и заплатила вчетверо больше токенов за апскейл,
    который не добавляет ни пикселя информации.
    """
    raise NotImplementedError


def anyres_token_count(height, width, tile=TILE, patch=PATCH, thumbnail=True):
    """Сколько визуальных токенов съест картинка при AnyRes.

    Плиток rows*cols по (tile/patch)^2 токенов плюс, если thumbnail=True,
    ещё одна такая же порция на превью всей картинки целиком.

    anyres_token_count(672, 672)                   ->  2880   (4 плитки + превью)
    anyres_token_count(336, 336)                   ->  1152   (1 плитка + превью)
    anyres_token_count(336, 336, thumbnail=False)  ->  576    (базовый LLaVA)

    Превью — это глобальный контекст: плитки видят детали, но не видят, как
    они соотносятся друг с другом. За него платят 576 токенами.

    Число 2880 стоит запомнить: столько стоит одна картинка в LLaVA-NeXT.
    В контексте 2048 она не помещается вообще.
    """
    raise NotImplementedError


def context_usage(image_tokens, text_tokens, context_window):
    """Что останется от контекста после картинки. Словарь.

    Ключи: used, free, image_share, fits.
      used        = image_tokens + text_tokens
      free        = context_window - used   (может быть отрицательным!)
      image_share = image_tokens / used
      fits        = used <= context_window

    context_usage(576, 100, 2048)
        ->  {"used": 676, "free": 1372, "image_share": 0.852..., "fits": True}

    free специально не обрезается по нулю: минус тридцать три показывает,
    насколько именно ты промахнулся, а ноль — нет.

    Отрицательные счётчики токенов или неположительное окно — ValueError.
    """
    raise NotImplementedError
