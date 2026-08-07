"""
Внутренности serving-движка: PagedAttention, continuous batching, chunked prefill

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l04-vllm-serving-internals
Разбор:  /check-code p17-l04-vllm-serving-internals
"""

import math
from collections import deque

KV_BLOCK_SIZE = 16          # столько токенов в одном блоке KV-кэша (дефолт vLLM)
PREFILL_SEC_PER_TOKEN = 0.00004   # prefill compute-bound: платим за каждый токен
DECODE_STEP_BASE_SEC = 0.010      # decode memory-bound: чтение весов из HBM,
DECODE_SEC_PER_SEQ = 0.0002       # добавка за каждую последовательность в батче
STEP_OVERHEAD_SEC = 0.0002        # фиксированные накладные одного forward-вызова
CHUNK_SIZE = 512            # размер куска prefill по умолчанию в vLLM
MIXED_WORKLOAD = (
    (0.00, 128, 80), (0.02, 256, 120), (0.04, 8192, 60), (0.06, 512, 200),
    (0.08, 128, 90), (0.10, 2048, 150), (0.12, 256, 80), (0.14, 128, 110),
    (0.16, 8192, 70), (0.18, 512, 160), (0.20, 128, 100), (0.22, 2048, 130),
    (0.24, 256, 90), (0.26, 128, 140), (0.28, 8192, 60), (0.30, 512, 180),
    (0.32, 128, 85), (0.34, 2048, 120), (0.36, 256, 95), (0.38, 128, 150),
    (0.40, 512, 110), (0.42, 128, 75), (0.44, 2048, 140), (0.46, 256, 105),
)


class OutOfKVBlocks(Exception):
    """В пуле KV-блоков не осталось места под запрошенную последовательность.

    Свой класс, а не MemoryError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, поэтому `pytest.raises(RuntimeError)` прошёл бы
    зелёным на пустой заготовке и ничего бы не проверил.
    """
    pass


def blocks_for(num_tokens, block_size):
    """Сколько блоков нужно под num_tokens токенов.

    blocks_for(16, 16)  ->  1
    blocks_for(17, 16)  ->  2    (последний блок занят одним токеном)
    blocks_for(0, 16)   ->  0

    Округление вверх — единственный источник внутренней фрагментации в
    PagedAttention: недозаполненным бывает только последний блок
    последовательности.
    """
    raise NotImplementedError


def contiguous_waste(seq_lens, max_len):
    """Доля впустую зарезервированной памяти при классической непрерывной аллокации.

    contiguous_waste([1500, 1500], 8192)  ->  0.8168...
    contiguous_waste([8192], 8192)        ->  0.0

    Классический аллокатор резервирует max_len слотов на КАЖДУЮ
    последовательность заранее — иначе KV-кэш придётся двигать. Всё, что
    сверх реальной длины, лежит мёртвым грузом.

    Это те самые 60-80% потерь из урока. Сравни с paged_waste на тех же
    данных: разница и есть смысл PagedAttention.
    """
    raise NotImplementedError


def paged_waste(seq_lens, block_size):
    """Доля впустую занятой памяти при блочной аллокации.

    paged_waste([1500, 1500], 16)  ->  0.00265...
    paged_waste([16, 32], 16)      ->  0.0        (всё легло ровно по блокам)

    Теряется только хвост последнего блока каждой последовательности, максимум
    block_size - 1 токен. От max_len результат не зависит вообще — блоки
    выдаются по мере роста.

    Ловушка: делить надо на ЗАНЯТУЮ память (число блоков * block_size), а не
    на сумму реальных длин.
    """
    raise NotImplementedError


class BlockPool:
    """Блочный аллокатор KV-кэша — PagedAttention в миниатюре.

    Пул из total_blocks блоков по block_size токенов. Последовательность
    берёт блоки по мере роста и отдаёт их все сразу, когда закончилась.

    pool = BlockPool(10, 16)
    pool.allocate("a", 20)   # 20 токенов -> 2 блока
    pool.free_blocks()       ->  8
    pool.append_token("a")   # 21-й токен, оба блока ещё не полны
    pool.used_blocks()       ->  2
    pool.free("a")
    pool.free_blocks()       ->  10

    Именно возврат блоков в общий пул позволяет планировщику принимать новые
    запросы, не дожидаясь конца батча.
    """

    def __init__(self, total_blocks, block_size):
        """Создать пул. Все блоки свободны.

        BlockPool(10, 16).free_blocks()  ->  10
        """
        raise NotImplementedError

    def used_blocks(self):
        """Сколько блоков занято сейчас.

        BlockPool(10, 16).used_blocks()  ->  0
        """
        raise NotImplementedError

    def free_blocks(self):
        """Сколько блоков свободно сейчас.

        BlockPool(10, 16).free_blocks()  ->  10
        """
        raise NotImplementedError

    def allocate(self, seq_id, num_tokens):
        """Выделить блоки под новую последовательность длины num_tokens.

        Не хватило места — OutOfKVBlocks. Такой seq_id уже есть — ValueError.
        """
        raise NotImplementedError

    def append_token(self, seq_id):
        """Добавить один токен. Новый блок берём, только когда прежние заполнены.

        Не хватило места под новый блок — OutOfKVBlocks.
        """
        raise NotImplementedError

    def free(self, seq_id):
        """Вернуть все блоки последовательности в пул."""
        raise NotImplementedError


def chunk_plan(prompt_len, chunk_size):
    """Разбить prefill на куски. chunk_size=None — один кусок целиком.

    chunk_plan(1200, 512)  ->  [512, 512, 176]
    chunk_plan(1200, None) ->  [1200]
    chunk_plan(0, 512)     ->  []

    Сумма кусков всегда равна prompt_len: chunked prefill не добавляет и не
    убавляет работу, он её РАСПРЕДЕЛЯЕТ по шагам, чтобы между кусками успел
    продвинуться decode.
    """
    raise NotImplementedError


def percentile(values, q):
    """Перцентиль по методу ближайшего ранга.

    percentile([1, 2, 3, 4], 50)          ->  2
    percentile([1, 2, 3, 4, 5], 100)      ->  5
    percentile([10] * 99 + [1000], 99)    ->  10

    Третий пример объясняет, зачем в уроке смотрят именно на P99, а не на
    среднее: один выброс из ста среднее заметно поднимет, а P99 — нет, зато
    P100 покажет его целиком.
    """
    raise NotImplementedError


def schedule_static(requests, batch_size):
    """Классический static batching: набрать батч, добить паддингом, ждать медленного.

    requests — последовательность кортежей (приход, длина промпта, длина ответа).
    Вернуть словарь метрик: makespan, output_tokens, throughput, ttft_mean,
    ttft_p99, itl_p99, itl_max, e2e_p99.

    schedule_static(MIXED_WORKLOAD, 8)["output_tokens"]  ->  2700

    Три источника потерь, все видны в коде:
      1. батч не стартует, пока не пришёл последний его участник;
      2. prefill считается по САМОМУ длинному промпту группы;
      3. decode идёт по самому длинному ответу, и короткие запросы всё это
         время занимают место в батче, уже ничего не производя.

    ITL внутри батча ровный — это единственное, в чём static хорош. Хвост он
    проигрывает на TTFT и на сквозной задержке.
    """
    raise NotImplementedError


def schedule_continuous(requests, total_blocks, chunk_size=None, block_size=KV_BLOCK_SIZE):
    """Continuous batching: приём и отпуск последовательностей на каждой итерации.

    chunk_size=None — prefill целиком за один шаг; число — chunked prefill.
    Метрики те же, что у schedule_static.

    schedule_continuous(MIXED_WORKLOAD, 1800)["output_tokens"]              ->  2700
    schedule_continuous(MIXED_WORKLOAD, 1800, CHUNK_SIZE)["output_tokens"]  ->  2700

    Порядок одной итерации ровно как в V1-планировщике vLLM:
      1. закончившиеся последовательности отдают блоки в пул;
      2. пока блоков хватает, из очереди принимаются новые;
      3. один forward: кому-то кусок prefill, кому-то один токен decode.

    Батч НЕ добивается паддингом до фиксированного размера, поэтому все
    2700 выходных токенов настоящие — сравни с static на тех же данных.

    Упрощение относительно vLLM: настоящий планировщик не знает длину ответа
    заранее и при нехватке блоков вытесняет последовательности, а мы
    резервируем весь горизонт сразу. Пул от этого работает честно, а кода
    вдвое меньше.

    Последовательность, которая не влезает в пул целиком, — OutOfKVBlocks:
    так планировщик не уходит в бесконечное ожидание.
    """
    raise NotImplementedError
