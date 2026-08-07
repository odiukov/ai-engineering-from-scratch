"""
Open-vocabulary vision: CLIP — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def normalize_rows(embeddings):
    """Привести каждый эмбеддинг батча к единичной длине.

    normalize_rows([[3.0, 4.0], [0.0, 2.0]])  ->  [[0.6, 0.8], [0.0, 1.0]]

    Нулевой эмбеддинг -> ValueError: направления у него нет, а косинус с ним
    не определён. Тихий ноль в ответе означал бы «одинаково похож на всё».

    Это `F.normalize(x, dim=-1)`, она же строчка
    `feats = feats / feats.norm(dim=-1, keepdim=True)` из примера с OpenCLIP.
    Обе башни CLIP заканчиваются этой нормализацией: без неё скалярное
    произведение мерило бы длину вектора, а не смысл.
    """
    out = []
    for v in embeddings:
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0.0:
            raise ValueError("cannot normalize a zero embedding")
        out.append([x / norm for x in v])
    return out


def similarity_matrix(image_emb, text_emb):
    """Матрица косинусов «каждая картинка против каждого текста», N x C.

    similarity_matrix([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        ->  [[1.0, 0.0, -1.0]]

    Единица — то же направление, ноль — независимость, минус единица —
    противоположность. Диапазон всегда [-1, 1], потому что обе стороны
    нормализуются внутри.

    Это `image_features @ text_features.T`. Обрати внимание на порядок: строки
    — картинки, столбцы — тексты. Перепутанный порядок не упадёт, а тихо
    поменяет местами задачи image-to-text и text-to-image.
    """
    img = normalize_rows(image_emb)
    txt = normalize_rows(text_emb)
    # C обычно мало (число классов), N велико — цикл по строкам дешевле
    # транспонирования матрицы текстов
    return [[sum(a * b for a, b in zip(i, t)) for t in txt] for i in img]


def clip_loss(image_emb, text_emb, logit_scale):
    """Симметричный контрастный лосс CLIP на батче из N пар (картинка, подпись).

    Пара i — это image_emb[i] и text_emb[i]; все остальные подписи батча для
    неё негативы, и наоборот.

    clip_loss(<N ортогональных>, <N ортогональных>, 100.0)  ->  log(N)
    clip_loss([[1.0, 0.0]], [[0.0, 1.0]], 100.0)            ->  0.0

    Второй пример — та же ловушка, что в SimCLR: при N=1 в батче нет ни одного
    негатива, softmax по одному элементу равен единице, лосс тождественно ноль
    при любых эмбеддингах. CLIP учится только на больших батчах.

    Симметрия обязательна: считаем кросс-энтропию по строкам (картинка ищет
    свою подпись) и по столбцам (подпись ищет свою картинку), берём полусумму.
    Иначе поиск будет работать только в одну сторону.

    logit_scale — это exp(learned scalar), у CLIP стартует с 1/0.07 ~ 14.3.
    """
    if len(image_emb) != len(text_emb):
        raise ValueError(f"batch mismatch: {len(image_emb)} images vs {len(text_emb)} texts")
    if not image_emb:
        raise ValueError("empty batch")
    if logit_scale <= 0:
        raise ValueError(f"logit_scale must be positive, got {logit_scale}")

    n = len(image_emb)
    sim = [[logit_scale * s for s in row] for row in similarity_matrix(image_emb, text_emb)]

    def cross_entropy(row, target):
        # log-sum-exp: при logit_scale=100 голый exp(100) уже переполняется
        top = max(row)
        lse = top + math.log(sum(math.exp(x - top) for x in row))
        return lse - row[target]

    # по строкам: i-я картинка обязана выбрать i-ю подпись
    l_i2t = sum(cross_entropy(sim[i], i) for i in range(n)) / n
    # по столбцам: та же матрица, прочитанная поперёк
    l_t2i = sum(cross_entropy([sim[i][j] for i in range(n)], j) for j in range(n)) / n
    return (l_i2t + l_t2i) / 2


def siglip_loss(image_emb, text_emb, logit_scale, bias=0.0):
    """Попарный сигмоидный лосс SigLIP: mean log(1 + exp(-y * sim)).

    y = +1 для совпадающей пары (i == j) и -1 для всех остальных N*N - N пар.

    siglip_loss([[1.0, 0.0]], [[1.0, 0.0]], 10.0)   ->  примерно 0.0000454
    siglip_loss([[1.0, 0.0]], [[-1.0, 0.0]], 10.0)  ->  примерно 10.0

    Ключевое отличие от clip_loss: здесь нет softmax по батчу, каждая пара
    оценивается сама по себе. Поэтому батч из одной пары уже даёт сигнал —
    ровно то, ради чего SigLIP и придуман.

    Ловушка: exp(-y * sim) при sim = -100 это exp(100), переполнение.
    Считай через softplus(z) = max(z, 0) + log1p(exp(-|z|)).
    """
    if len(image_emb) != len(text_emb):
        raise ValueError(f"batch mismatch: {len(image_emb)} images vs {len(text_emb)} texts")
    if not image_emb:
        raise ValueError("empty batch")
    if logit_scale <= 0:
        raise ValueError(f"logit_scale must be positive, got {logit_scale}")

    n = len(image_emb)
    sim = similarity_matrix(image_emb, text_emb)
    total = 0.0
    for i in range(n):
        for j in range(n):
            z = logit_scale * sim[i][j] + bias
            y = 1.0 if i == j else -1.0
            # softplus в устойчивой форме: без max(...) exp(+100) даст inf
            arg = -y * z
            total += max(arg, 0.0) + math.log1p(math.exp(-abs(arg)))
    return total / (n * n)


def build_prompts(class_names, templates):
    """Развернуть имена классов в шаблоны: список списков строк, по одному на класс.

    build_prompts(["cat", "dog"], ["a photo of a {}", "a sketch of a {}"])
        ->  [["a photo of a cat", "a sketch of a cat"],
             ["a photo of a dog", "a sketch of a dog"]]

    Порядок внешнего списка = порядок class_names. Он же дальше становится
    порядком столбцов матрицы похожестей, поэтому перемешивать его нельзя.

    Шаблон без "{}" -> ValueError: "a photo of a cat" для класса "dog" — это
    молчаливо неверная разметка, которую потом не отловить.

    Зачем 80 шаблонов вместо одного: OpenAI намерял на ImageNet +1-3% top-1
    просто от усреднения эмбеддингов разных формулировок одного класса.
    """
    if not templates:
        raise ValueError("need at least one prompt template")
    for t in templates:
        if "{}" not in t:
            raise ValueError(f"template {t!r} has no {{}} placeholder")
    return [[t.format(name) for t in templates] for name in class_names]


def average_class_embeddings(per_class_embeddings):
    """Свернуть эмбеддинги всех шаблонов класса в один единичный вектор.

    per_class_embeddings[c] — список эмбеддингов всех шаблонов класса c.

    average_class_embeddings([[[1.0, 0.0], [1.0, 0.0]]])  ->  [[1.0, 0.0]]
    average_class_embeddings([[[3.0, 0.0], [0.0, 3.0]]])  ->  [[0.707..., 0.707...]]

    Порядок: нормализовать каждый шаблон, усреднить, нормализовать результат.
    Средний вектор короче единицы (шаблоны смотрят чуть в разные стороны) —
    без финальной нормализации класс с согласованными шаблонами получил бы
    незаслуженно большую похожесть просто из-за длины.

    Класс без шаблонов -> ValueError. Класс, шаблоны которого гасят друг друга
    в ноль, тоже -> ValueError: у такого «класса» нет направления.
    """
    out = []
    for embeddings in per_class_embeddings:
        if not embeddings:
            raise ValueError("class has no prompt embeddings")
        unit = normalize_rows(embeddings)
        dim = len(unit[0])
        mean = [sum(v[k] for v in unit) / len(unit) for k in range(dim)]
        # normalize_rows сам бросит ValueError, если шаблоны взаимно погасились
        out.append(normalize_rows([mean])[0])
    return out


def zero_shot_probabilities(image_emb, class_emb, logit_scale=100.0):
    """Вероятности классов для каждой картинки: softmax(logit_scale * косинусы).

    zero_shot_probabilities([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], 1.0)
        ->  [[0.731..., 0.268...]]

    Строка — картинка, столбец — класс, каждая строка суммируется в единицу.

    Это `(100.0 * image_features @ text_features.T).softmax(dim=-1)`. Число 100
    — не магия, а exp(logit_scale) обученной температуры: оно превращает
    косинусы из тесного диапазона [-1, 1] в разделимые логиты.

    Помни: сумма в единицу не означает уверенности. Если на картинке нет ни
    одного из классов, softmax всё равно распределит всю единицу между ними.
    """
    if logit_scale <= 0:
        raise ValueError(f"logit_scale must be positive, got {logit_scale}")
    out = []
    for row in similarity_matrix(image_emb, class_emb):
        logits = [logit_scale * s for s in row]
        top = max(logits)  # стабильный softmax: exp(100) уже на грани
        exps = [math.exp(x - top) for x in logits]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def zero_shot_classify(image_emb, class_emb, class_names):
    """Zero-shot классификация: имя ближайшего по косинусу класса для каждой картинки.

    zero_shot_classify([[1.0, 0.0], [0.0, 1.0]],
                       [[1.0, 0.0], [0.0, 1.0]], ["cat", "dog"])
        ->  ["cat", "dog"]

    Ни одного обученного на этих классах слоя: классы задаются текстом, и
    добавить новый — значит дописать строку в class_names. В этом весь смысл
    слова open-vocabulary.

    len(class_names) != len(class_emb) -> ValueError: сдвиг на единицу здесь
    даёт правдоподобный, но систематически неверный ответ.

    При равных косинусах побеждает меньший индекс — просто чтобы ответ был
    воспроизводим.
    """
    if len(class_names) != len(class_emb):
        raise ValueError(
            f"{len(class_names)} names for {len(class_emb)} class embeddings"
        )
    if not class_emb:
        raise ValueError("no classes to choose from")
    sim = similarity_matrix(image_emb, class_emb)
    # max по значению, при равенстве — первый: index() даёт ровно это
    return [class_names[row.index(max(row))] for row in sim]
