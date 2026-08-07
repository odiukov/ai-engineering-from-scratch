"""
Vision Transformer и патч-токены

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p12-l01-vision-transformer-patch-tokens
Разбор:  /check-code p12-l01-vision-transformer-patch-tokens
"""


def grid_shape(height, width, patch):
    """Сколько патчей по вертикали и по горизонтали помещается в картинку.

    grid_shape(224, 224, 16)  ->  (14, 14)
    grid_shape(336, 224, 14)  ->  (24, 16)

    Классический ViT работает на фиксированном разрешении и требует, чтобы
    сторона делилась на patch НАЦЕЛО. Если не делится — ValueError, а не
    молчаливое округление: обрезанная полоска пикселей это потерянный
    текст на скриншоте и потерянный край на снимке.

    patch <= 0 тоже ValueError.
    """
    raise NotImplementedError


def sequence_length(height, width, patch, cls=True, registers=0):
    """Длина последовательности, которую увидит трансформер.

    sequence_length(224, 224, 16)                 ->  197   (196 патчей + CLS)
    sequence_length(224, 224, 16, cls=False)      ->  196   (mean pooling)
    sequence_length(384, 384, 14, registers=4)    ->  ...

    Патч-токены плюс необязательный CLS плюс register-токены. Register-токены
    (DINOv2, SigLIP 2) — это «мусорка» для высоконормных артефактов attention;
    перед передачей в LLM их выбрасывают, но по сети они едут наравне со всеми.

    Именно это число умножается на само себя внутри self-attention, поэтому
    оно и есть цена разрешения.
    """
    raise NotImplementedError


def extract_patches(image, patch):
    """Нарезать картинку на непересекающиеся патчи и расплющить каждый.

    Картинка — список строк, строка — список пикселей, пиксель — список
    каналов. То есть image[r][c][ch].

    Для картинки 4x4 с одним каналом, где пиксель равен r*10+c:
      extract_patches(image, 2)[0]  ->  [0, 1, 10, 11]
      extract_patches(image, 2)[1]  ->  [2, 3, 12, 13]

    Два порядка обхода, и оба важны:
      * патчи идут по сетке слева направо, сверху вниз (row-major);
      * внутри патча — сначала строка, потом столбец, потом канал.
    Перепутаешь любой из них — позиционные эмбеддинги лягут не туда, и сеть
    будет учить перестановку вместо картинки.

    Длина каждого патча равна patch * patch * channels.
    """
    raise NotImplementedError


def project_patches(patches, W_E, bias=None):
    """Общая линейная проекция патчей в скрытую размерность D.

    W_E — список из D строк, каждая длиной patch*patch*channels.
    token[d] = dot(W_E[d], patch) + bias[d].

    project_patches([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  [[1.0, 2.0]]

    Проекция ОДНА на все патчи — те же веса для верхнего левого угла и для
    нижнего правого. Отсюда два следствия, которые проверяют тесты:
    одинаковые патчи дают одинаковые токены, и без bias отображение линейно.

    В продакшене это пишут как nn.Conv2d(C, D, kernel_size=P, stride=P) —
    математически ровно то же самое, просто быстрее.
    """
    raise NotImplementedError


def add_position_embeddings(tokens, table):
    """Прибавить к каждому токену его позиционный вектор. Новый список.

    add_position_embeddings([[1.0], [1.0]], [[0.0], [5.0]])  ->  [[1.0], [6.0]]

    До этого шага трансформер видит патчи как мешок без порядка: два
    одинаковых куска неба в разных углах для него неразличимы. Позиционный
    эмбеддинг — единственное, что их разводит.

    Входные списки не менять: их ещё используют выше по стеку.
    Несовпадение длин — ValueError.
    """
    raise NotImplementedError


def mean_pool(tokens):
    """Усреднить токены покоординатно — image-level представление.

    mean_pool([[1.0, 2.0], [3.0, 4.0]])  ->  [2.0, 3.0]

    Альтернатива CLS-токену; так делают SigLIP, DINOv2 и большинство
    современных VLM. Свойство, ради которого это работает: результат не
    зависит от порядка токенов — pooling симметричен.

    Пустой список — ValueError: среднего у ничего нет.
    """
    raise NotImplementedError


def vit_param_count(image_size, patch, dim, depth, channels=3, registers=0):
    """Разложить число параметров ViT по слагаемым. Словарь.

    Ключи: patch_embed, extra_tokens, position, blocks, final_norm, total.

    vit_param_count(224, 16, 768, 12)["total"]  ->  85_798_656   (это ViT-B/16)

    Формулы, чтобы все считали одинаково:
      patch_embed  = channels*patch*patch*dim + dim
      extra_tokens = (1 + registers) * dim            (CLS и регистры)
      position     = sequence_length * dim            (CLS и регистры тоже)
      блок         = 12*dim^2 + 13*dim
                     (QKVO: 4*dim^2 + 4*dim; MLP 4x: 8*dim^2 + 5*dim;
                      два LayerNorm: 4*dim)
      blocks       = depth * блок
      final_norm   = 2 * dim

    Прикидывай так ЛЮБОЙ энкодер до того, как качать чекпойнт: размер
    backbone задаёт нижнюю границу VRAM всей будущей VLM.
    """
    raise NotImplementedError
