"""
Экономика inference-платформ: per-token, per-minute и точка окупаемости — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math

MINUTES_PER_DAY = 1440
DAYS_PER_MONTH = 30

# Каталог провайдеров из урока. Ровно одна из трёх ставок не None — это и есть
# модель оплаты вендора. Цены учебные, из docs/en.md, а не с прайс-листа.
#   per_mtok                  — $ за миллион выходных токенов (Fireworks, Together)
#   per_minute                — $ за минуту выделенной GPU (Baseten, Modal, Anyscale)
#   per_prediction            — $ за один вызов модели (Replicate)
#   tokens_per_minute         — сколько токенов в минуту выдаёт насыщенная GPU
#   reserved_minutes_per_day  — минимум оплаченных минут в сутки (тёплый пул)
VENDORS = {
    "fireworks": {"per_mtok": 0.90, "per_minute": None, "per_prediction": None,
                  "tokens_per_minute": 900_000, "reserved_minutes_per_day": 0},
    "together": {"per_mtok": 0.88, "per_minute": None, "per_prediction": None,
                 "tokens_per_minute": 850_000, "reserved_minutes_per_day": 0},
    "baseten": {"per_mtok": None, "per_minute": 0.55, "per_prediction": None,
                "tokens_per_minute": 900_000, "reserved_minutes_per_day": 1440},
    "modal": {"per_mtok": None, "per_minute": 0.48, "per_prediction": None,
              "tokens_per_minute": 800_000, "reserved_minutes_per_day": 60},
    "replicate": {"per_mtok": None, "per_minute": None, "per_prediction": 0.006,
                  "tokens_per_minute": 750_000, "reserved_minutes_per_day": 0},
    "anyscale": {"per_mtok": None, "per_minute": 0.60, "per_prediction": None,
                 "tokens_per_minute": 850_000, "reserved_minutes_per_day": 1440},
}


class ZeroWorkload(Exception):
    """Нормировать стоимость не на что: в знаменателе ноль токенов.

    Свой класс, а не ZeroDivisionError и не RuntimeError. NotImplementedError —
    наследник RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы
    зелёным на пустой заготовке.
    """


class NeverBreakEven(Exception):
    """Точки окупаемости не существует: один вариант дешевле при любом объёме."""


def per_token_cost(tokens, price_per_mtok):
    """Счёт при поштучной оплате токенов.

    per_token_cost(2_000_000, 0.90)  ->  1.8
    per_token_cost(0, 0.90)          ->  0.0

    Ставка дана за МИЛЛИОН токенов. Никакого простоя ты не оплачиваешь — в
    этом весь смысл модели.
    """
    return tokens / 1e6 * price_per_mtok


def per_minute_cost(tokens, tokens_per_minute, price_per_minute, reserved_minutes_per_day):
    """Счёт при поминутной оплате выделенной GPU за сутки.

    per_minute_cost(2_000_000, 900_000, 0.55, 1440)  ->  792.0
    per_minute_cost(2_000_000, 800_000, 0.48, 60)    ->  28.8

    Оплачиваем МАКСИМУМ из двух величин: сколько минут GPU реально считала
    (tokens / tokens_per_minute) и сколько минут стоит тёплый пул, даже
    пустой. Именно этот пол и делает модель дорогой на малых объёмах — оба
    примера выше при одинаковом трафике различаются в 27 раз только полом.

    Ловушка: соблазн взять сумму вместо максимума. Резерв не прибавляется к
    работе, он её поглощает.
    """
    if tokens_per_minute <= 0:
        raise ValueError("tokens_per_minute must be positive")
    saturated_minutes = tokens / tokens_per_minute
    return max(saturated_minutes, reserved_minutes_per_day) * price_per_minute


def per_prediction_cost(predictions, price_per_prediction):
    """Счёт при оплате за вызов модели.

    per_prediction_cost(10_000, 0.006)  ->  60.0

    Длина ответа на цену не влияет — поэтому модель выгодна на коротких
    генерациях и разорительна на длинных.
    """
    return predictions * price_per_prediction


def effective_rate_per_mtok(cost, tokens):
    """Привести любой счёт к общей единице: $ за миллион токенов.

    effective_rate_per_mtok(792.0, 2_000_000)    ->  396.0
    effective_rate_per_mtok(792.0, 100_000_000)  ->  7.92

    Без этой нормировки прайс-листы несравнимы: у одного $/M токенов, у
    другого $/минута, у третьего $/вызов. Оба примера — один и тот же счёт
    Baseten; разница только в том, сколько работы он покрыл.

    На нулевом трафике делить не на что — ZeroWorkload.
    """
    if tokens <= 0:
        raise ZeroWorkload("cannot normalize cost over zero tokens")
    return cost / (tokens / 1e6)


def utilization_breakeven(price_per_mtok, tokens_per_minute, price_per_minute,
                          reserved_minutes_per_day):
    """Доля суточной загрузки GPU, выше которой per-minute дешевле per-token.

    utilization_breakeven(0.90, 900_000, 0.55, 1440)  ->  0.679...
    utilization_breakeven(0.90, 800_000, 0.48, 60)    ->  0.0277...

    Разбор первого примера: при 100% загрузке GPU выдаёт 900k*1440 = 1.296B
    токенов, что по $0.90/M стоит $1166.40. Резерв на сутки стоит 1440*$0.55
    = $792 и от загрузки не зависит. Пересечение: 792 / 1166.40 = 0.679.

    Второй пример — тот же расчёт, но пол всего 60 минут: $28.80 против тех
    же $1036.80 за полные сутки, отсюда 2.8%. Вывод урока «per-minute
    выигрывает выше ~30%» верен только для конкретного пола: пол и есть
    главный параметр, а не ставка.

    Если поминутный вариант и при 100% загрузке не дешевле — NeverBreakEven.
    """
    if tokens_per_minute <= 0 or price_per_mtok <= 0:
        raise ValueError("tokens_per_minute and price_per_mtok must be positive")
    token_cost_full_day = tokens_per_minute * MINUTES_PER_DAY / 1e6 * price_per_mtok
    minute_cost_full_day = MINUTES_PER_DAY * price_per_minute
    # при полной загрузке поминутный путь всё ещё не дешевле — пересечения нет
    if minute_cost_full_day >= token_cost_full_day:
        raise NeverBreakEven("per-minute never beats per-token at any utilization")
    utilization = reserved_minutes_per_day * price_per_minute / token_cost_full_day
    if utilization > 1.0:
        raise NeverBreakEven("break-even utilization exceeds a full day")
    return utilization


def blended_rate(base_rate_per_mtok, batch_share, batch_discount):
    """Средняя ставка, когда часть трафика уходит в дешёвую batch-очередь.

    blended_rate(0.90, 0.4, 0.5)  ->  0.72   (40% трафика со скидкой 50%)
    blended_rate(0.90, 0.0, 0.5)  ->  0.90   (никто не ушёл — скидки нет)
    blended_rate(0.90, 1.0, 0.5)  ->  0.45   (весь трафик в batch)

    Ловушка: скидка 50% на 40% трафика — это НЕ минус 50% и не минус 40%,
    а минус 20%. Считай средневзвешенное, а не вычитай проценты.

    Доли задаются числами 0..1, а не процентами; выход за диапазон —
    ValueError, потому что «120% трафика в batch» — это опечатка, а не
    экстремальный сценарий.
    """
    if not 0.0 <= batch_share <= 1.0:
        raise ValueError("batch_share must be within [0, 1]")
    if not 0.0 <= batch_discount <= 1.0:
        raise ValueError("batch_discount must be within [0, 1]")
    return base_rate_per_mtok * (1.0 - batch_share * batch_discount)


def selfhosted_breakeven_requests(managed_price_per_request, fixed_monthly,
                                  variable_per_request):
    """Минимум запросов в месяц, начиная с которого self-hosted СТРОГО дешевле.

    selfhosted_breakeven_requests(0.002, 2000.0, 0.0005)  ->  1_333_334
    selfhosted_breakeven_requests(0.004, 3000.0, 0.001)   ->  1_000_001

    Managed стоит managed_price за запрос и ноль в простое. Self-hosted стоит
    fixed_monthly (аренда GPU, дежурство, мониторинг) плюс variable за запрос.
    Порог: fixed / (managed - variable).

    Второй пример показывает, зачем слово «строго»: 3000/0.003 = ровно
    миллион, но на миллионе счета РАВНЫ. Дешевле становится только со
    следующего запроса, поэтому ответ 1_000_001, а не 1_000_000.

    Если своя переменная стоимость не ниже managed-ставки, порога нет вообще
    — NeverBreakEven. Это типичный случай для низкого трафика: своя GPU
    никогда не догонит API.
    """
    margin = managed_price_per_request - variable_per_request
    if margin <= 0:
        raise NeverBreakEven("self-hosted variable cost is not below the managed rate")
    exact = fixed_monthly / margin
    # floor + 1, а не ceil: в точке ровного равенства выгоды ещё нет
    return math.floor(exact) + 1


def cheapest_vendor(vendors, tokens_per_day, predictions_per_day):
    """Самый дешёвый вендор каталога на суточной нагрузке: вернуть (name, cost).

    cheapest_vendor(VENDORS, 2_000_000, 10_000)      ->  ("together", 1.76)
    cheapest_vendor(VENDORS, 100_000_000, 500_000)   ->  ("modal", 60.0)

    Один и тот же каталог, разный объём — разный победитель. В этом весь урок:
    вопрос «какая платформа дешевле» без объёма не имеет ответа.

    Модель оплаты определяется тем, какая ставка не None. При равной цене
    берём меньшее имя по алфавиту, чтобы результат не зависел от порядка
    ключей в словаре.
    """
    best = None
    for name in sorted(vendors):
        v = vendors[name]
        if v["per_mtok"] is not None:
            cost = per_token_cost(tokens_per_day, v["per_mtok"])
        elif v["per_minute"] is not None:
            cost = per_minute_cost(tokens_per_day, v["tokens_per_minute"],
                                   v["per_minute"], v["reserved_minutes_per_day"])
        elif v["per_prediction"] is not None:
            cost = per_prediction_cost(predictions_per_day, v["per_prediction"])
        else:
            raise ValueError(f"vendor {name} has no pricing model")
        if best is None or cost < best[1]:
            best = (name, cost)
    if best is None:
        raise ValueError("empty vendor catalog")
    return best
