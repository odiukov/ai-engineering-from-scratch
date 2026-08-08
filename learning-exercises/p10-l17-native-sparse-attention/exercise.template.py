"""
Native Sparse Attention (DeepSeek NSA)

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l17-native-sparse-attention
Разбор:  /check-code p10-l17-native-sparse-attention
"""

import math


def softmax(row):
    """Строка логитов -> распределение вероятностей.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([0.0, 1000.0])    ->  [0.0, 1.0]   (без OverflowError!)

    Ловушка: math.exp(1000) падает с OverflowError. Вычти максимум строки
    перед экспонентой.
    """
    raise NotImplementedError


def attention_weights(q, K):
    """Веса внимания запроса q по ключам K: softmax(q . k_i / sqrt(d)).

    attention_weights([1.0, 0.0], [[0.0, 0.0], [0.0, 0.0]])  ->  [0.5, 0.5]

    d берётся как len(q). Пустой список ключей -> ValueError: делить не на
    что, а вернуть [] значит уронить ветку тихо.
    """
    raise NotImplementedError


def attend(weights, V):
    """Взвешенная сумма строк значений: out[j] = sum_i w_i * V[i][j].

    attend([0.5, 0.5], [[1.0, 0.0], [3.0, 4.0]])  ->  [2.0, 2.0]
    """
    raise NotImplementedError


def compress_blocks(K, l):
    """Сжатая ветка: блоки по l строк, каждый усредняется в одну строку.

    compress_blocks([[1.0], [3.0], [5.0]], 2)  ->  [[2.0], [5.0]]
    compress_blocks([[1.0], [3.0]], 1)         ->  [[1.0], [3.0]]

    Хвост короче l усредняется по тому, что в нём реально есть, — делить
    на l нельзя, иначе последний блок окажется заниженным.

    В настоящей NSA здесь обучаемый MLP; среднее — честная заглушка,
    которая не мешает увидеть остальную конструкцию.
    """
    raise NotImplementedError


def top_k_blocks(weights, k):
    """Индексы k блоков с наибольшими весами, по возрастанию.

    top_k_blocks([0.1, 0.5, 0.2, 0.4], 2)  ->  [1, 3]
    top_k_blocks([0.1, 0.5], 5)            ->  [0, 1]   (k больше числа блоков)

    При равных весах побеждает меньший индекс — выбор обязан быть
    воспроизводимым.

    Тонкость урока: это единственное недифференцируемое место NSA, и оно
    ни на что не влияет в графе — top_k только решает, какие блоки грузить
    из памяти. Градиент течёт через оценки сжатой ветки.
    """
    raise NotImplementedError


def selected_branch(q, K, V, selection_block_size, k):
    """Выбранная ветка: top-k блоков selection_block_size, токены — исходные.

    Порядок: сжали K -> посчитали веса по сжатым ключам -> взяли top-k
    блоков -> собрали ИСХОДНЫЕ токены этих блоков -> обычное внимание.

    Размер блока выбора не связан с размером блока отдельной сжатой
    ветки. Свойство для проверки: при selection_block_size = 1 и
    k >= числа токенов ответ совпадает
    с плотным вниманием по всей последовательности. Так и должно быть —
    разреженность с полным окном обязана вырождаться в плотную.

    Веса пересчитываются по несжатым ключам: сжатые оценки нужны только
    чтобы выбрать блоки, а не чтобы взвешивать токены внутри них.
    """
    raise NotImplementedError


def nsa_attention(
    q, K, V, compression_block_size, k, selection_block_size, w, gates
):
    """Три ветки NSA, сложенные с гейтами (g_cmp, g_sel, g_win).

    Модуль модели: NSA-блок целиком.

    out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win

    Гейты НЕ обязаны давать в сумме единицу: в статье это выход маленького
    MLP по запросу, ветки взвешиваются независимо.

    Три полезные проверки:
      gates = (0, 0, 1), w >= len(K)  ->  обычное плотное внимание;
      gates = (0, 1, 0), selection_block_size = 1, k >= len(K)
          ->  тоже плотное внимание;
      gates = (0, 0, 0)  ->  нулевой вектор.
    """
    raise NotImplementedError


def keys_per_query(n, compression_block_size, k, selection_block_size, w):
    """Бюджет вычислений: сколько ключей видит один запрос в каждой ветке.

    Возвращает словарь с ключами compressed, selected, window, total,
    full, reduction (= full / total).

    keys_per_query(64000, 64, 16, 64, 512)["total"]  ->  2536
    keys_per_query(64000, 64, 16, 64, 512)["full"]   ->  64000

    compressed = ceil(n / compression_block_size),
    selected = min(k * selection_block_size, n), window = min(w, n).
    Ограничение по n обязательно: нельзя прочитать больше ключей, чем есть.

    Ради чего всё: на 64k выигрыш 25x, на 128k уже 36x. Экономия растёт
    вместе с длиной контекста — в этом и весь смысл.
    """
    raise NotImplementedError
