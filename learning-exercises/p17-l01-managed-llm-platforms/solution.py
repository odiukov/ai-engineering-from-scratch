"""
Managed LLM платформы: on-demand, PTU и выбор под SLA — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30

# Каталог платформ. Цены и латентности — учебные приближения из урока
# (docs/en.md, раздел «Numbers you should remember»), а не прайс-лист.
#   in_per_mtok / out_per_mtok  — $ за миллион входных / выходных токенов
#   ptu_hourly                  — $ в час за одну единицу зарезервированной
#                                 мощности; None означает «PTU не продаётся»
#   ptu_tokens_per_hour         — сколько токенов в час тянет одна единица
#   ttft_p99_ondemand_ms        — P99 времени до первого токена на общей мощности
#   ttft_p99_ptu_ms             — то же на выделенной мощности
PLATFORMS = {
    "bedrock": {
        "in_per_mtok": 3.00,
        "out_per_mtok": 15.00,
        "ptu_hourly": 21.0,
        "ptu_tokens_per_hour": 1_200_000,
        "ttft_p99_ondemand_ms": 180.0,
        "ttft_p99_ptu_ms": 83.0,
    },
    "azure": {
        "in_per_mtok": 2.50,
        "out_per_mtok": 10.00,
        "ptu_hourly": 10.0,
        "ptu_tokens_per_hour": 2_000_000,
        "ttft_p99_ondemand_ms": 140.0,
        "ttft_p99_ptu_ms": 57.0,
    },
    "vertex": {
        "in_per_mtok": 1.25,
        "out_per_mtok": 5.00,
        "ptu_hourly": None,
        "ptu_tokens_per_hour": 0,
        "ttft_p99_ondemand_ms": 160.0,
        "ttft_p99_ptu_ms": 160.0,
    },
}


class SLAUnreachable(Exception):
    """Ни одна платформа каталога не укладывается в требуемый SLA по TTFT.

    Свой класс, а не ValueError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, поэтому тест `pytest.raises(RuntimeError)` прошёл
    бы зелёным на пустой заготовке и ничего бы не проверил.
    """


def ondemand_cost(tokens_in, tokens_out, price_in_per_mtok, price_out_per_mtok):
    """Стоимость on-demand: платим за фактические токены, вход и выход по разным ставкам.

    ondemand_cost(3_000_000, 1_000_000, 3.0, 15.0)  ->  24.0
    ondemand_cost(0, 0, 3.0, 15.0)                  ->  0.0

    Ставки на прайс-листах даны за МИЛЛИОН токенов ($/M tok), а объём приходит
    в штуках. Делить на 1e6 нужно объём, а не цену.

    Это базовая единица всех расчётов урока: любую другую модель оплаты мы
    сравниваем именно с ней.
    """
    return tokens_in / 1e6 * price_in_per_mtok + tokens_out / 1e6 * price_out_per_mtok


def ptu_units_needed(total_tokens, tokens_per_hour_per_unit, hours):
    """Сколько единиц зарезервированной мощности (PTU) нужно на объём за hours часов.

    ptu_units_needed(45_000_000, 2_000_000, 24)  ->  1   (ёмкость 48M, влезло)
    ptu_units_needed(50_000_000, 2_000_000, 24)  ->  2   (48M мало, берём вторую)
    ptu_units_needed(0, 2_000_000, 24)           ->  1   (резерв не бывает нулевым)

    Единицы дискретны: округляем ВВЕРХ, потому что половину PTU не продают.
    И минимум одна — резервирование существует, даже когда трафика нет; в этом
    вся суть PTU: платишь за простой.

    Если платформа не продаёт PTU (tokens_per_hour_per_unit == 0), считать
    нечего — ValueError.
    """
    if tokens_per_hour_per_unit <= 0:
        raise ValueError("tokens_per_hour_per_unit must be positive")
    if hours <= 0:
        raise ValueError("hours must be positive")
    capacity = tokens_per_hour_per_unit * hours
    # max(1, ...) — резерв нельзя купить в нулевом количестве
    return max(1, math.ceil(total_tokens / capacity))


def ptu_cost(total_tokens, tokens_per_hour_per_unit, hourly_price, hours):
    """Стоимость PTU-пути: число единиц * цена часа * часы.

    ptu_cost(45_000_000, 2_000_000, 10.0, 24)  ->  240.0
    ptu_cost(1_000, 2_000_000, 10.0, 24)       ->  240.0   (тот же счёт!)

    Обрати внимание на второй пример: объём упал в сорок пять тысяч раз, счёт
    не изменился. PTU — фиксированная плата за резерв, она не зависит от того,
    сколько трафика реально прошло.
    """
    units = ptu_units_needed(total_tokens, tokens_per_hour_per_unit, hours)
    return units * hourly_price * hours


def ptu_breakeven_utilization(hourly_price, tokens_per_hour_per_unit, price_per_mtok):
    """Доля утилизации одной PTU, при которой резерв сравнивается с on-demand.

    ptu_breakeven_utilization(10.0, 2_000_000, 10.0)  ->  0.5
    ptu_breakeven_utilization(21.0, 1_200_000, 15.0)  ->  1.1666...

    Смысл: за час полностью загруженная единица выдаёт tokens_per_hour токенов,
    те же токены на on-demand стоят tokens_per_hour/1e6 * price_per_mtok.
    Резерв стоит hourly_price всегда. Делим одно на другое.

    Результат больше 1 — честный ответ «не окупается никогда»: даже при
    стопроцентной загрузке резерв дороже. Второй пример как раз такой.
    """
    if price_per_mtok <= 0 or tokens_per_hour_per_unit <= 0:
        raise ValueError("price_per_mtok and tokens_per_hour_per_unit must be positive")
    ondemand_full_hour = tokens_per_hour_per_unit / 1e6 * price_per_mtok
    return hourly_price / ondemand_full_hour


def cheapest_path(platform, tokens_in, tokens_out, hours):
    """Выбрать дешёвый путь для одной платформы: вернуть (path, cost).

    path — либо "on-demand", либо "ptu".

    cheapest_path(PLATFORMS["vertex"], 3_000_000, 1_000_000, 24)
        ->  ("on-demand", 8.75)          (Vertex не продаёт PTU вообще)
    cheapest_path(PLATFORMS["azure"], 30_000_000, 15_000_000, 24)
        ->  ("on-demand", 225.0)         (45M токенов слишком мало для резерва)

    При РАВНОЙ цене выбирай "on-demand": обязательство на месяц вперёд без
    выигрыша в деньгах — это чистый минус в гибкости.

    PTU считаем по суммарному объёму (вход + выход): резерв меряется в
    пропускной способности, а не в направлении токенов.
    """
    cost_on = ondemand_cost(
        tokens_in, tokens_out, platform["in_per_mtok"], platform["out_per_mtok"]
    )
    if platform["ptu_hourly"] is None:
        return ("on-demand", cost_on)
    cost_ptu = ptu_cost(
        tokens_in + tokens_out,
        platform["ptu_tokens_per_hour"],
        platform["ptu_hourly"],
        hours,
    )
    # строгое < : при ничьей остаёмся на on-demand
    return ("ptu", cost_ptu) if cost_ptu < cost_on else ("on-demand", cost_on)


def pick_platform(platforms, tokens_in, tokens_out, hours, sla_ttft_p99_ms):
    """Самая дешёвая платформа среди тех, кто укладывается в SLA по P99 TTFT.

    Вернуть (name, path, cost). Если не проходит никто — SLAUnreachable.

    pick_platform(PLATFORMS, 3_000_000, 1_000_000, 24, 200.0)
        ->  ("vertex", "on-demand", 8.75)
    pick_platform(PLATFORMS, 3_000_000, 1_000_000, 24, 10.0)
        ->  SLAUnreachable

    Ловушка: SLA проверяется по ТОМУ пути, который выбран по цене. Дешёвый
    on-demand и быстрый PTU — разные строки в SLA, и брать латентность PTU
    для on-demand-цены нельзя.

    При равной цене выбирай платформу с меньшим именем по алфавиту: результат
    должен быть воспроизводимым, а не зависеть от порядка ключей словаря.
    """
    best = None
    for name in sorted(platforms):
        platform = platforms[name]
        path, cost = cheapest_path(platform, tokens_in, tokens_out, hours)
        p99 = (
            platform["ttft_p99_ptu_ms"] if path == "ptu" else platform["ttft_p99_ondemand_ms"]
        )
        if p99 > sla_ttft_p99_ms:
            continue
        # строгое < : sorted() уже дал алфавитный порядок, первый победитель и остаётся
        if best is None or cost < best[2]:
            best = (name, path, cost)
    if best is None:
        raise SLAUnreachable(
            f"no platform meets P99 TTFT <= {sla_ttft_p99_ms} ms"
        )
    return best


def redundancy_uplift(primary_daily_cost, gateway_pct, headroom_pct):
    """Надбавка за политику «минимум два провайдера», в день и в месяц.

    Вернуть (daily, monthly).

    redundancy_uplift(50.0, 3.0, 10.0)  ->  (6.5, 195.0)
    redundancy_uplift(50.0, 0.0, 0.0)   ->  (0.0, 0.0)

    Две статьи: накладные расходы шлюза (лишний хоп, логирование) и holding
    тёплого запаса у второго провайдера, который в обычный день простаивает.
    Обе считаются от основного счёта.

    Месяц берём как DAYS_PER_MONTH дней — грубо, зато сравнимо между уроками.
    """
    daily = primary_daily_cost * (gateway_pct + headroom_pct) / 100.0
    return (daily, daily * DAYS_PER_MONTH)
