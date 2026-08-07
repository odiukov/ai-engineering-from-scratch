"""
SGLang и RadixAttention: префиксное дерево KV-кэша и cache-aware планировщик

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l06-sglang-radixattention
Разбор:  /check-code p17-l06-sglang-radixattention
"""

POLICIES = ("fcfs", "cache_aware")
PARTS = {
    "system": tuple(range(1000, 1200)),      # 200 токенов системного промпта
    "tools": tuple(range(2000, 2120)),       # 120 токенов схем инструментов
    "context_a": tuple(range(3000, 3080)),   # 80 токенов документа A
    "context_b": tuple(range(4000, 4080)),   # 80 токенов документа B
}
CANONICAL_ORDER = ("system", "tools", "context_a")
SCRAMBLED_ORDER = ("system", "context_a", "tools")
QUESTIONS = tuple(tuple(range(9000 + 100 * i, 9020 + 100 * i)) for i in range(8))
ORDERED_WORKLOAD = tuple(
    PARTS["system"] + PARTS["tools"] + PARTS["context_a"] + QUESTIONS[i]
    for i in range(8)
)
SCRAMBLED_WORKLOAD = tuple(
    (PARTS["system"] + PARTS["tools"] + PARTS["context_a"] if i % 2 == 0
     else PARTS["system"] + PARTS["context_a"] + PARTS["tools"]) + QUESTIONS[i]
    for i in range(8)
)
MIXED_WORKLOAD = tuple(
    PARTS["system"] + PARTS["tools"]
    + PARTS["context_a" if i % 2 == 0 else "context_b"] + QUESTIONS[i]
    for i in range(8)
)
CACHE_CAPACITY = 480


class CacheTooSmall(Exception):
    """Освободить нужное число токенов нечем: живых листьев не осталось.

    Свой класс, а не MemoryError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, поэтому `pytest.raises(RuntimeError)` прошёл бы
    зелёным на пустой заготовке и ничего бы не проверил.
    """
    pass


def common_prefix_len(a, b):
    """Длина общего префикса двух последовательностей токенов.

    common_prefix_len((1, 2, 3), (1, 2, 9))  ->  2
    common_prefix_len((1, 2), (9, 2))        ->  0    <- расхождение в первом токене
    common_prefix_len((1, 2), (1, 2, 3))     ->  2

    Второй пример и есть вся арифметика префиксного кэша: расхождение в самом
    первом токене обнуляет выгоду целиком, сколько бы одинакового ни шло дальше.
    """
    raise NotImplementedError


def render_prompt(order, parts):
    """Собрать промпт из компонентов в заданном порядке. Порядок и есть ключ кэша.

    render_prompt(("system", "tools"), {"system": (1, 2), "tools": (3,)})  ->  (1, 2, 3)

    Компонент, которого нет в parts — KeyError. Пустой order — ValueError.

    Функция существует ради одного наблюдения: два вызова с одним и тем же
    parts, но разным order дают последовательности, у которых общий префикс
    заканчивается на первом же переставленном компоненте. Дереву всё равно,
    что человек считает эти промпты «одинаковыми».
    """
    raise NotImplementedError


def prefill_speedup(hit_rate):
    """Во сколько раз дешевеет prefill при данной доле переиспользованных токенов.

    prefill_speedup(0.0)    ->  1.0     (ничего не переиспользовано)
    prefill_speedup(0.844)  ->  6.41    (те самые 6.4x из урока)
    prefill_speedup(0.864)  ->  7.35    (86.4% на voice cloning)

    Prefill стоит пропорционально НОВЫМ токенам, поэтому ускорение равно
    1 / (1 - hit_rate). Зависимость нелинейная: с 50% до 75% выигрыш удваивается,
    а с 90% до 95% — тоже удваивается. Последние проценты дисциплины промпта
    стоят больше первых.

    hit_rate вне [0, 1) — ValueError. Единица означала бы, что считать нечего.
    """
    raise NotImplementedError


class RadixCache:
    """KV-кэш как префиксное дерево со сжатыми рёбрами — RadixAttention в миниатюре.

    Каждый узел владеет отрезком токенов (segment) и «своими» KV-блоками.
    Путь от корня до узла — это последовательность, для которой кэш посчитан.
    Новый запрос идёт по дереву, пока токены совпадают; всё совпавшее не
    пересчитывается, дальше выделяется только хвост.

    cache = RadixCache(1000)
    cache.insert((1, 2, 3), now=1.0)   ->  {"reused": 0, "new": 3}
    cache.insert((1, 2, 9), now=2.0)   ->  {"reused": 2, "new": 1}
    cache.match((1, 2, 3))             ->  3
    cache.used_tokens()                ->  4

    Вытеснение — LRU по ЛИСТЬЯМ: узел с живыми потомками не выкидывается
    никогда, иначе путь к его детям оборвётся и их KV станет недостижимым.
    Так форма кэша повторяет форму дерева, о чём и говорит урок.
    """

    def __init__(self, capacity_tokens):
        """Пустой кэш ёмкостью capacity_tokens токенов.

        RadixCache(1000).used_tokens()  ->  0

        Ёмкость меньше 1 — ValueError.
        """
        raise NotImplementedError

    def used_tokens(self):
        """Сколько токенов сейчас лежит в кэше.

        RadixCache(1000).used_tokens()  ->  0
        """
        raise NotImplementedError

    def match(self, tokens):
        """Сколько токенов из начала tokens уже есть в дереве. Ничего не меняет.

        Пустой кэш даёт 0 на любом входе. Совпадение считается по пути от корня
        и обрывается на первом же расхождении — в том числе В СЕРЕДИНЕ отрезка
        узла: половина отрезка тоже засчитывается.

        Это и есть «сколько токенов промпта достанется бесплатно».
        """
        raise NotImplementedError

    def insert(self, tokens, now):
        """Положить последовательность в кэш. Вернуть {"reused": int, "new": int}.

        reused совпадает с match(tokens) ДО вставки, new — остаток.
        Время подаётся параметром now и служит меткой LRU: весь путь, по
        которому прошёл запрос, помечается временем now.

        Не хватает места под новые токены — вызывается evict. Узлы, тронутые
        этим же запросом (last_used == now), из вытеснения исключены: иначе
        кэш выкинул бы ветку, которую сам только что признал горячей.

        Промпт длиннее ёмкости кэша — CacheTooSmall. Пустой промпт — ValueError.
        """
        raise NotImplementedError

    def evict(self, need_tokens, now):
        """Освободить не меньше need_tokens токенов по LRU. Вернуть сколько освободили.

        Выкидываются ТОЛЬКО листья: у узла с потомками свои KV-блоки лежат на
        пути к их блокам, и снос такого узла оборвал бы путь. Когда лист
        уходит, его родитель сам может стать листом и попасть под вытеснение
        на следующем витке — так ветка снимается послойно, снизу вверх.

        Кандидаты с last_used >= now пропускаются: это ветка текущего запроса,
        её вытеснять бессмысленно. Кандидатов не осталось — CacheTooSmall.

        Порядок при равном last_used — по пути от корня, чтобы результат не
        зависел от порядка ключей в словарях.

        need_tokens <= 0 — ничего не делаем, возвращаем 0.
        """
        raise NotImplementedError


def cache_aware_order(cache, prompts, pending):
    """Расставить ожидающие запросы: сначала те, у кого длиннее общий префикс с кэшем.

    pending — индексы в prompts. Вернуть их же, переупорядоченными.

    cache_aware_order(empty_cache, prompts, [0, 1, 2])  ->  [0, 1, 2]

    Это и есть «обход дерева в глубину» из урока: горячая ветка обслуживается
    подряд и не успевает вытесниться. При равной длине совпадения порядок
    прихода сохраняется — планировщик не должен переставлять запросы без причины.

    FCFS не сортирует вообще: берёт первого пришедшего, даже если следующий за
    ним делит с резидентной веткой на порядок больше токенов.
    """
    raise NotImplementedError


def run_workload(prompts, capacity_tokens, policy):
    """Прогнать поток промптов через кэш выбранной политикой. Вернуть метрики.

    Вернуть словарь: order, reused_tokens, prefill_tokens, prompt_tokens,
    hit_rate, speedup.

    run_workload(ORDERED_WORKLOAD, CACHE_CAPACITY, "fcfs")["hit_rate"]  ->  0.833...

    prefill_tokens — то, за что реально платит GPU: только новые токены.
    hit_rate — доля переиспользованных, speedup считается через prefill_speedup.

    fcfs берёт запросы в порядке прихода, cache_aware — через cache_aware_order,
    пересчитывая совпадения после КАЖДОЙ вставки: дерево меняется на ходу, и
    порядок, посчитанный один раз в начале, врал бы.

    Неизвестная политика — ValueError.
    """
    raise NotImplementedError
