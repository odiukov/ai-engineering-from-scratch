"""
Vision Transformers: патчи вместо пикселей — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def patch_grid(height, width, patch_size):
    """Сколько патчей влезает в картинку: (строк патчей, столбцов патчей).

    patch_grid(224, 224, 16)  ->  (14, 14)    196 патчей, как в ViT-B/16
    patch_grid(24, 24, 6)     ->  (4, 4)

    Патчи не пересекаются и не выходят за край, поэтому размеры обязаны
    делиться на patch_size без остатка — иначе ValueError. Настоящие
    пайплайны сначала ресайзят картинку именно под это условие.

    patch_size <= 0 — тоже ValueError.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if height % patch_size or width % patch_size:
        raise ValueError("image size must be divisible by patch_size")
    return height // patch_size, width // patch_size


def patchify(image, patch_size):
    """Порезать картинку на патчи и сплющить каждый в один вектор.

    image — список строк, строка — список пикселей, пиксель — список
    каналов. Результат — список плоских патчей длиной patch_size^2 * C
    в растровом порядке: слева направо, сверху вниз.

    patchify([[[1], [2]], [[3], [4]]], 1)  ->  [[1], [2], [3], [4]]
    patchify([[[1], [2]], [[3], [4]]], 2)  ->  [[1, 2, 3, 4]]

    Внутри патча порядок тоже растровый: сначала вся первая строка патча
    (со всеми каналами каждого пикселя), потом вторая. Так делают все ViT,
    и путать этот порядок нельзя — обученная матрица проекции ждёт именно его.

    Это единственное «зрительное» место в ViT. Дальше идёт обычный
    трансформерный энкодер, тот же, что в BERT.
    """
    grid_h, grid_w = patch_grid(len(image), len(image[0]), patch_size)
    patches = []
    for gi in range(grid_h):
        for gj in range(grid_w):
            patch = []
            for di in range(patch_size):
                row = image[gi * patch_size + di]
                for dj in range(patch_size):
                    patch.extend(row[gj * patch_size + dj])
            patches.append(patch)
    return patches


def unpatchify(patches, patch_size, height, width):
    """Собрать картинку обратно из плоских патчей. Обратна patchify.

    unpatchify([[1, 2, 3, 4]], 2, 2, 2)  ->  [[[1], [2]], [[3], [4]]]

    Число каналов не передаётся — оно вычисляется:
    C = len(patches[0]) / patch_size^2. Если делится не нацело — ValueError.

    Зачем: если round-trip совпал с исходником, значит порядок обхода не
    перепутан. Ошибка в порядке — самая частая и самая незаметная в ViT:
    сеть всё равно чему-то обучится, только хуже.
    """
    grid_h, grid_w = patch_grid(height, width, patch_size)
    if len(patches) != grid_h * grid_w:
        raise ValueError("patch count does not match the requested image size")
    flat_size = len(patches[0])
    channels, rest = divmod(flat_size, patch_size * patch_size)
    if rest or channels < 1:
        raise ValueError("flat patch length is not patch_size^2 * C")

    image = [[None] * width for _ in range(height)]
    for idx, patch in enumerate(patches):
        gi, gj = divmod(idx, grid_w)
        cursor = 0
        for di in range(patch_size):
            for dj in range(patch_size):
                pixel = patch[cursor:cursor + channels]
                cursor += channels
                image[gi * patch_size + di][gj * patch_size + dj] = list(pixel)
    return image


def linear_project(patches, W):
    """Линейная проекция патчей в d_model: одна общая матрица на все патчи.

    W — матрица (patch_flat_size, d_model). Результат — список токенов,
    по одному на патч, каждый длиной d_model.

    linear_project([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [[1.0, 2.0]]
    linear_project([[1.0, 1.0]], [[2.0], [3.0]])            ->  [[5.0]]

    В PyTorch то же самое пишется как nn.Conv2d(C, d_model, kernel_size=P,
    stride=P): свёртка с шагом, равным размеру ядра, — это ровно «порезать
    на патчи и умножить каждый на общую матрицу», никакой магии.

    Ширина патча обязана совпадать с числом строк W — иначе ValueError.
    """
    if len(patches[0]) != len(W):
        raise ValueError("patch length must match the number of rows in W")
    d_model = len(W[0])
    tokens = []
    for patch in patches:
        row = [0.0] * d_model
        for value, w_row in zip(patch, W):
            if value == 0.0:
                continue  # ноль ничего не добавляет, а картинки разрежены
            for j in range(d_model):
                row[j] += value * w_row[j]
        tokens.append(row)
    return tokens


def pos_2d(grid_h, grid_w, d_model):
    """Двумерное синусоидальное позиционное кодирование для сетки патчей.

    Возвращает список grid_h * grid_w векторов длиной d_model в растровом
    порядке — тот же порядок, что у patchify.

    pos_2d(1, 1, 4)  ->  [[0.0, 1.0, 0.0, 1.0]]   (sin 0 = 0, cos 0 = 1)

    Устройство: d_model делится на две половины. Первая кодирует НОМЕР
    СТРОКИ патча, вторая — номер столбца, каждая обычными парами sin/cos с
    частотами 10000^(2k/half). Поэтому два патча из одной строки имеют
    одинаковую первую половину, а из одного столбца — одинаковую вторую.

    d_model обязан делиться на 4: по две половины, в каждой пары sin/cos.
    Иначе ValueError.

    В оригинальном ViT позиционные эмбеддинги были обучаемыми, а не
    синусоидальными; здесь мы собираем более поздний вариант, потому что у
    него нет параметров и его можно проверить руками.
    """
    if d_model % 4:
        raise ValueError("d_model must be divisible by 4")
    half = d_model // 2
    out = []
    for i in range(grid_h):
        for j in range(grid_w):
            vec = [0.0] * d_model
            for k in range(half // 2):
                freq = 10000 ** (2 * k / half)
                vec[2 * k] = math.sin(i / freq)
                vec[2 * k + 1] = math.cos(i / freq)
                vec[half + 2 * k] = math.sin(j / freq)
                vec[half + 2 * k + 1] = math.cos(j / freq)
            out.append(vec)
    return out


def add_cls_and_pos(tokens, cls_token, pos):
    """Добавить [CLS] в начало и прибавить позиционное кодирование к патчам.

    add_cls_and_pos([[1.0], [2.0]], [0.0], [[10.0], [20.0]])
        ->  [[0.0], [11.0], [22.0]]

    Длина последовательности становится len(tokens) + 1. Само [CLS]
    позиционного кодирования не получает: у него нет места в картинке, его
    работа — собрать в себя всю картинку через attention. Его финальный
    вектор и есть представление изображения для классификатора.

    Функция ничего не портит на входе: tokens и pos остаются прежними.
    len(pos) обязан равняться len(tokens) — иначе ValueError.
    """
    if len(pos) != len(tokens):
        raise ValueError("need one positional vector per patch token")
    out = [list(cls_token)]
    for token, p in zip(tokens, pos):
        out.append([t + pv for t, pv in zip(token, p)])
    return out


def attention_pairs(height, width, patch_size):
    """Сколько пар «запрос-ключ» посчитает self-attention на одной картинке.

    attention_pairs(224, 224, 16)  ->  38809    (197 токенов в квадрате)
    attention_pairs(4, 4, 2)       ->  25       (4 патча + CLS)

    Токенов ровно patch_grid + 1 на [CLS], а attention стоит квадрат от
    числа токенов. Отсюда главный рычаг ViT: уменьшил патч вдвое — токенов
    стало вчетверо больше, а attention подорожал примерно в шестнадцать раз.
    Потому детекция и сегментация с патчем 8x8 такие дорогие.
    """
    grid_h, grid_w = patch_grid(height, width, patch_size)
    n_tokens = grid_h * grid_w + 1  # +1 на [CLS]
    return n_tokens * n_tokens


def vit_param_count(d_model, n_layers, n_patches, patch_size, channels=3,
                    ffn_expansion=4, n_classes=1000):
    """Приблизительное число параметров ViT.

    vit_param_count(768, 12, 196, 16)   ->  86482944   (ViT-Base/16, ~86M)
    vit_param_count(192, 12, 196, 16)   ->  5695488    (ViT-Tiny/16, ~5.7M)

    Из чего складывается:
      * блок: 4 * d^2 на Q, K, V, O + 2 * d * (expansion * d) на FFN
        + 4 * d на два LayerNorm;
      * patch embedding: patch_size^2 * channels * d;
      * позиционные эмбеддинги: (n_patches + 1) * d;
      * сам [CLS]: d;
      * финальный LayerNorm: 2 * d;
      * голова классификатора: d * n_classes.

    Смещения (bias) для краткости не считаем — они дают меньше процента.

    Ориентиры: ViT-Base ~86M против ResNet-50 ~25M. ViT-Large ~307M.
    Видно и главное свойство: параметры растут КВАДРАТИЧНО по d_model и
    линейно по числу слоёв, а от размера картинки почти не зависят.
    """
    per_layer = 4 * d_model ** 2 + 2 * d_model * int(ffn_expansion * d_model) + 4 * d_model
    patch_embed = patch_size * patch_size * channels * d_model
    pos_embed = (n_patches + 1) * d_model
    head = d_model * n_classes
    return per_layer * n_layers + patch_embed + pos_embed + d_model + 2 * d_model + head
