"""
Внутренности serving-движка: PagedAttention, continuous batching, chunked prefill — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
from collections import deque

KV_BLOCK_SIZE = 16          # столько токенов в одном блоке KV-кэша (дефолт vLLM)

# Учебная модель стоимости шага. Числа подобраны так, чтобы воспроизвести форму
# бенчмарков из урока, а не абсолютные миллисекунды на H100.
PREFILL_SEC_PER_TOKEN = 0.00004   # prefill compute-bound: платим за каждый токен
DECODE_STEP_BASE_SEC = 0.010      # decode memory-bound: чтение весов из HBM,
                                  # цена одна на весь батч — отсюда вся выгода батчинга
DECODE_SEC_PER_SEQ = 0.0002       # добавка за каждую последовательность в батче
STEP_OVERHEAD_SEC = 0.0002        # фиксированные накладные одного forward-вызова

CHUNK_SIZE = 512            # размер куска prefill по умолчанию в vLLM

# Смешанный трафик: (момент прихода, длина промпта, длина ответа).
# Есть и короткие запросы, и промпты на 8192 токена — именно на такой смеси
# видно разницу между режимами планировщика.
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


def blocks_for(num_tokens, block_size):
    """Сколько блоков нужно под num_tokens токенов.

    blocks_for(16, 16)  ->  1
    blocks_for(17, 16)  ->  2    (последний блок занят одним токеном)
    blocks_for(0, 16)   ->  0

    Округление вверх — единственный источник внутренней фрагментации в
    PagedAttention: недозаполненным бывает только последний блок
    последовательности.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens < 0:
        raise ValueError("num_tokens must not be negative")
    return math.ceil(num_tokens / block_size)


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
    if not seq_lens:
        raise ValueError("empty seq_lens")
    if any(n > max_len for n in seq_lens):
        raise ValueError("sequence longer than max_len")
    reserved = len(seq_lens) * max_len
    return (reserved - sum(seq_lens)) / reserved


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
    if not seq_lens:
        raise ValueError("empty seq_lens")
    allocated = sum(blocks_for(n, block_size) for n in seq_lens) * block_size
    if allocated == 0:
        return 0.0
    return (allocated - sum(seq_lens)) / allocated


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
        if total_blocks < 0 or block_size <= 0:
            raise ValueError("bad pool geometry")
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._tokens = {}   # seq_id -> сколько токенов уже лежит
        self._blocks = {}   # seq_id -> сколько блоков занято

    def used_blocks(self):
        """Сколько блоков занято сейчас.

        BlockPool(10, 16).used_blocks()  ->  0
        """
        return sum(self._blocks.values())

    def free_blocks(self):
        """Сколько блоков свободно сейчас.

        BlockPool(10, 16).free_blocks()  ->  10
        """
        return self.total_blocks - self.used_blocks()

    def allocate(self, seq_id, num_tokens):
        """Выделить блоки под новую последовательность длины num_tokens.

        Не хватило места — OutOfKVBlocks. Такой seq_id уже есть — ValueError.
        """
        if seq_id in self._blocks:
            raise ValueError(f"sequence {seq_id!r} is already allocated")
        need = blocks_for(num_tokens, self.block_size)
        if need > self.free_blocks():
            raise OutOfKVBlocks(
                f"need {need} blocks, only {self.free_blocks()} free"
            )
        self._blocks[seq_id] = need
        self._tokens[seq_id] = num_tokens

    def append_token(self, seq_id):
        """Добавить один токен. Новый блок берём, только когда прежние заполнены.

        Не хватило места под новый блок — OutOfKVBlocks.
        """
        if seq_id not in self._blocks:
            raise KeyError(seq_id)
        self._tokens[seq_id] += 1
        need = blocks_for(self._tokens[seq_id], self.block_size)
        if need > self._blocks[seq_id]:
            if self.free_blocks() < 1:
                self._tokens[seq_id] -= 1   # откат: состояние должно остаться целым
                raise OutOfKVBlocks("no free block to grow the sequence")
            self._blocks[seq_id] = need

    def free(self, seq_id):
        """Вернуть все блоки последовательности в пул."""
        if seq_id not in self._blocks:
            raise KeyError(seq_id)
        del self._blocks[seq_id]
        del self._tokens[seq_id]


def chunk_plan(prompt_len, chunk_size):
    """Разбить prefill на куски. chunk_size=None — один кусок целиком.

    chunk_plan(1200, 512)  ->  [512, 512, 176]
    chunk_plan(1200, None) ->  [1200]
    chunk_plan(0, 512)     ->  []

    Сумма кусков всегда равна prompt_len: chunked prefill не добавляет и не
    убавляет работу, он её РАСПРЕДЕЛЯЕТ по шагам, чтобы между кусками успел
    продвинуться decode.
    """
    if prompt_len < 0:
        raise ValueError("prompt_len must not be negative")
    if prompt_len == 0:
        return []
    if chunk_size is None:
        return [prompt_len]
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    plan = []
    left = prompt_len
    while left > 0:
        take = min(chunk_size, left)
        plan.append(take)
        left -= take
    return plan


def percentile(values, q):
    """Перцентиль по методу ближайшего ранга.

    percentile([1, 2, 3, 4], 50)          ->  2
    percentile([1, 2, 3, 4, 5], 100)      ->  5
    percentile([10] * 99 + [1000], 99)    ->  10

    Третий пример объясняет, зачем в уроке смотрят именно на P99, а не на
    среднее: один выброс из ста среднее заметно поднимет, а P99 — нет, зато
    P100 покажет его целиком.
    """
    if not values:
        raise ValueError("empty values")
    if not 0 < q <= 100:
        raise ValueError("q must be within (0, 100]")
    ordered = sorted(values)
    idx = math.ceil(q / 100.0 * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, idx))]


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
    now = 0.0
    ttfts, itls, e2es = [], [], []
    output_tokens = 0
    for start in range(0, len(requests), batch_size):
        group = requests[start:start + batch_size]
        # GPU освободилась и пришёл последний участник — только теперь стартуем
        now = max(now, max(r[0] for r in group))
        pad_prompt = max(r[1] for r in group)
        now += pad_prompt * PREFILL_SEC_PER_TOKEN + STEP_OVERHEAD_SEC
        pad_output = max(r[2] for r in group)
        # len(group) в цене шага — платим и за уже закончившиеся последовательности
        step = DECODE_STEP_BASE_SEC + len(group) * DECODE_SEC_PER_SEQ + STEP_OVERHEAD_SEC
        for i in range(pad_output):
            now += step
            for arrival, _prompt, out_len in group:
                if i < out_len:
                    output_tokens += 1
                    if i == 0:
                        # TTFT заканчивается первым сгенерированным токеном,
                        # как и в continuous scheduler ниже.
                        ttfts.append(now - arrival)
                    else:
                        # TPOT/ITL — только интервалы после первого токена.
                        itls.append(step)
        for arrival, _prompt, _out in group:
            e2es.append(now - arrival)   # батч уходит целиком, вместе с медленным
    return {
        "makespan": now,
        "output_tokens": output_tokens,
        "throughput": output_tokens / now if now else 0.0,
        "ttft_mean": sum(ttfts) / len(ttfts),
        "ttft_p99": percentile(ttfts, 99),
        "itl_p99": percentile(itls, 99),
        "itl_max": max(itls),
        "e2e_p99": percentile(e2es, 99),
    }


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
    pool = BlockPool(total_blocks, block_size)
    waiting = deque(sorted(
        ({"rid": i, "arrival": a, "prompt_len": p, "output_len": o,
          "prefilled": 0, "generated": 0, "ttft": None, "last_token_at": None}
         for i, (a, p, o) in enumerate(requests)),
        key=lambda r: (r["arrival"], r["rid"]),
    ))
    running = []
    now = 0.0
    ttfts, itls, e2es = [], [], []
    output_tokens = 0

    while waiting or running:
        # считать нечего — перематываем время к ближайшему приходу
        if not running and waiting and now < waiting[0]["arrival"]:
            now = waiting[0]["arrival"]
        while waiting and waiting[0]["arrival"] <= now:
            r = waiting[0]
            need = blocks_for(r["prompt_len"] + r["output_len"], block_size)
            if need > pool.total_blocks:
                raise OutOfKVBlocks(
                    f"request {r['rid']} needs {need} blocks, pool holds {pool.total_blocks}"
                )
            if need > pool.free_blocks():
                break
            # Admission уже проверил полный KV-горизонт, поэтому резервируем
            # те же блоки. Иначе несколько промптов могли пройти проверку,
            # а затем столкнуться с OutOfKVBlocks во время decode.
            pool.allocate(r["rid"], r["prompt_len"] + r["output_len"])
            running.append(waiting.popleft())
        if not running:
            now = waiting[0]["arrival"]
            continue

        prefill_tokens = 0
        decoders = []
        for r in running:
            if r["prefilled"] < r["prompt_len"]:
                remaining = r["prompt_len"] - r["prefilled"]
                # первый кусок плана и есть работа этого шага
                take = chunk_plan(remaining, chunk_size)[0]
                r["prefilled"] += take
                prefill_tokens += take
            else:
                decoders.append(r)

        # один fused forward: куски prefill и по одному токену decode вместе
        dt = (prefill_tokens * PREFILL_SEC_PER_TOKEN
              + (DECODE_STEP_BASE_SEC if decoders else 0.0)
              + len(decoders) * DECODE_SEC_PER_SEQ
              + STEP_OVERHEAD_SEC)
        now += dt

        for r in decoders:
            r["generated"] += 1
            output_tokens += 1
            if r["ttft"] is None:
                ttfts.append(now - r["arrival"])
                r["ttft"] = now
            else:
                itls.append(now - r["last_token_at"])
            r["last_token_at"] = now

        for r in [x for x in running if x["generated"] >= x["output_len"]]:
            e2es.append(now - r["arrival"])   # уходит сразу, не дожидаясь соседей
            pool.free(r["rid"])
            running.remove(r)

    return {
        "makespan": now,
        "output_tokens": output_tokens,
        "throughput": output_tokens / now if now else 0.0,
        "ttft_mean": sum(ttfts) / len(ttfts),
        "ttft_p99": percentile(ttfts, 99),
        "itl_p99": percentile(itls, 99),
        "itl_max": max(itls),
        "e2e_p99": percentile(e2es, 99),
    }
