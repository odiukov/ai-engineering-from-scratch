"""
ColPali и vision-native RAG по документам — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

# Размерность эмбеддинга ColPali и число патчей на страницу у PaliGemma.
# Из этих двух чисел и считается вся смета хранилища.
COLPALI_DIM = 128
PATCHES_PER_PAGE = 729
BYTES_PER_FLOAT = 4


def cosine(a, b):
    """Косинусная близость двух векторов: от -1 до 1.

    cosine([1.0, 0.0], [1.0, 0.0])    ->  1.0
    cosine([1.0, 0.0], [0.0, 1.0])    ->  0.0
    cosine([1.0, 0.0], [5.0, 0.0])    ->  1.0   (длина роли не играет)

    Нулевой вектор направления не имеет: возвращай 0.0, а не деление на ноль.
    Пустые патчи белого поля страницы дают ровно такой вектор.
    """
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def l2_normalize(v):
    """Вектор той же длины, но с нормой 1. Нулевой вектор возвращается как есть.

    l2_normalize([3.0, 4.0])  ->  [0.6, 0.8]
    l2_normalize([0.0, 0.0])  ->  [0.0, 0.0]

    После нормализации косинус превращается в обычное скалярное
    произведение — это и есть тот трюк, ради которого векторы в индексе
    хранят уже нормализованными.
    """
    n = math.sqrt(sum(x * x for x in v))
    if n == 0.0:
        return list(v)
    return [x / n for x in v]


def maxsim(query_tokens, page_patches):
    """MaxSim из ColBERT/ColPali: сумма по токенам запроса максимумов по патчам.

    score = sum_i max_j cos(q_i, p_j)

    maxsim([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  1.0
    maxsim([[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]])  ->  1.414...

    Каждый токен запроса сам выбирает лучший патч страницы. Это late
    interaction: страница не сжимается в один вектор, сравнение происходит
    в момент запроса.

    Ловушка: сумма, а не среднее. Длинный запрос набирает больший балл — и
    это правильно, страницы сравниваются между собой при ОДНОМ запросе.
    """
    # для каждого токена запроса один проход по патчам: O(N_q * N_p * D)
    return sum(max(cosine(q, p) for p in page_patches) for q in query_tokens)


def mean_sim(query_tokens, page_patches):
    """Средняя близость по всем парам (токен запроса, патч страницы).

    mean_sim([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])  ->  0.5

    Нужна для контраста с maxsim. Среднее размывает единственный точный
    патч сотнями пустых полей страницы: чем крупнее страница, тем сильнее
    штраф за то, что на ней есть ещё что-то кроме ответа.
    """
    pairs = [cosine(q, p) for q in query_tokens for p in page_patches]
    return sum(pairs) / len(pairs)


def pool_page(page_patches):
    """Сжатие страницы в ОДИН вектор: усреднение патчей плюс нормализация.

    Так работает bi-encoder — и VisRAG из урока.

    pool_page([[1.0, 0.0], [1.0, 0.0]])  ->  [1.0, 0.0]

    Дёшево по памяти (один вектор вместо 729), но необратимо: после
    усреднения уже не узнать, был ли на странице один яркий патч или сто
    посредственных.
    """
    dim = len(page_patches[0])
    mean = [sum(p[k] for p in page_patches) / len(page_patches) for k in range(dim)]
    return l2_normalize(mean)


def bi_encoder_score(query_tokens, page_patches):
    """Балл bi-encoder: усреднить запрос, усреднить страницу, взять косинус.

    bi_encoder_score([[1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]])  ->  1.0

    Ровно то, что делает классический text-RAG: один вектор на чанк, один
    на запрос, косинус. Сравни с maxsim на странице, где ответ занимает
    один патч из многих, — разница между 80% и 55% на ViDoRe берётся
    отсюда.
    """
    return cosine(pool_page(query_tokens), pool_page(page_patches))


def retrieve(query_tokens, pages, k=3):
    """Top-k страниц по MaxSim. Возвращает список пар (page_id, score).

    pages — словарь page_id -> список патч-эмбеддингов.

    retrieve(q, {"p1": [...], "p2": [...]}, k=1)  ->  [("p1", 2.31)]

    Сортировка по убыванию балла; при равных баллах — по page_id по
    возрастанию, чтобы выдача была детерминированной. Недетерминированный
    top-k невозможно ни отладить, ни замерить на ViDoRe.

    k больше числа страниц — вернуть все, не падать.
    """
    scored = [(pid, maxsim(query_tokens, patches)) for pid, patches in pages.items()]
    # ключ (-score, pid): минус вместо reverse=True, иначе reverse перевернёт
    # и порядок page_id при равных баллах
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


def storage_bytes(n_pages, n_patches=PATCHES_PER_PAGE, dim=COLPALI_DIM,
                  compression=1):
    """Сколько байт занимает индекс ColPali. compression — коэффициент PQ.

    storage_bytes(50)                  ->  18 662 400   (примерно 18 МБ)
    storage_bytes(50, compression=8)   ->  2 332 800

    n_pages * n_patches * dim * 4 байта, делённое на compression.

    Смысл: text-RAG на тех же 50 страницах — один вектор на чанк, около
    150 КБ. ColPali дороже примерно в 30 раз до сжатия и в 5-10 после PQ.
    Обычно терпимо, но на миллионах страниц это и есть решающий аргумент.
    """
    return n_pages * n_patches * dim * BYTES_PER_FLOAT // compression
