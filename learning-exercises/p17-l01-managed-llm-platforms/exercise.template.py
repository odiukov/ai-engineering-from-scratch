"""
Managed LLM платформы: on-demand, PTU и выбор под SLA

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l01-managed-llm-platforms
Разбор:  /check-code p17-l01-managed-llm-platforms
"""

import math

HOURS_PER_DAY = 24
DAYS_PER_MONTH = 30
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
    pass


def ondemand_cost(tokens_in, tokens_out, price_in_per_mtok, price_out_per_mtok):
    """Стоимость on-demand: платим за фактические токены, вход и выход по разным ставкам.

    ondemand_cost(3_000_000, 1_000_000, 3.0, 15.0)  ->  24.0
    ondemand_cost(0, 0, 3.0, 15.0)                  ->  0.0

    Ставки на прайс-листах даны за МИЛЛИОН токенов ($/M tok), а объём приходит
    в штуках. Делить на 1e6 нужно объём, а не цену.

    Это базовая единица всех расчётов урока: любую другую модель оплаты мы
    сравниваем именно с ней.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def ptu_cost(total_tokens, tokens_per_hour_per_unit, hourly_price, hours):
    """Стоимость PTU-пути: число единиц * цена часа * часы.

    ptu_cost(45_000_000, 2_000_000, 10.0, 24)  ->  240.0
    ptu_cost(1_000, 2_000_000, 10.0, 24)       ->  240.0   (тот же счёт!)

    Обрати внимание на второй пример: объём упал в сорок пять тысяч раз, счёт
    не изменился. PTU — фиксированная плата за резерв, она не зависит от того,
    сколько трафика реально прошло.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
