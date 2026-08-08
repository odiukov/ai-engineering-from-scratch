"""
FinOps для LLM: атрибуция, аномалии, прогноз счёта — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

from datetime import date

# Четыре слоя токенов из урока. Три первых — вход, последний — выход,
# и стоят они по-разному. Свалить их в одну корзину значит ослепнуть.
INPUT_LAYERS = ("prompt", "tool", "memory")
OUTPUT_LAYER = "response"
LAYERS = INPUT_LAYERS + (OUTPUT_LAYER,)

# Цены в долларах за миллион токенов.
PRICES = {
    "haiku": {"input": 0.80, "output": 4.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
}

# Множители из «compounded-savings stack»: кэш L2 удешевляет вход в ~10 раз,
# batch API снимает половину со всего счёта.
CACHED_INPUT_MULTIPLIER = 0.1
BATCH_MULTIPLIER = 0.5


def call_cost(call, prices=PRICES):
    """Стоимость одного вызова в долларах.

    Вызов — словарь с ключами route, layers и необязательными
    cached_input / batch.

    call_cost({"route": "haiku",
               "layers": {"prompt": 1800, "tool": 600,
                          "memory": 400, "response": 150}})
        ->  0.00284
    То же самое с "cached_input": True   ->  0.000824
    То же самое с "batch": True          ->  0.00142

    Ловушка: response тарифицируется по выходной цене, она в разы дороже
    входной. Посчитать все четыре слоя по одной ставке — самая частая ошибка
    в самодельных калькуляторах, и она занижает счёт.

    Неизвестный route — ValueError: молча считать по нулю значит потерять
    целую строку счёта.
    """
    route = call["route"]
    if route not in prices:
        raise ValueError("unknown route: %r" % (route,))
    price = prices[route]
    layers = call["layers"]
    # .get(k, 0): отсутствующий слой — это ноль токенов, а не KeyError.
    # Агентские вызовы без tool-слоя — норма
    input_tokens = sum(layers.get(k, 0) for k in INPUT_LAYERS)
    output_tokens = layers.get(OUTPUT_LAYER, 0)
    input_rate = price["input"]
    if call.get("cached_input"):
        input_rate *= CACHED_INPUT_MULTIPLIER
    cost = input_tokens / 1e6 * input_rate + output_tokens / 1e6 * price["output"]
    if call.get("batch"):
        cost *= BATCH_MULTIPLIER
    return cost


def layer_shares(calls):
    """Доля каждого слоя в общем числе токенов. Словарь по всем четырём слоям.

    layer_shares([{"layers": {"prompt": 60, "tool": 20,
                              "memory": 10, "response": 10}}])
        ->  {'prompt': 0.6, 'tool': 0.2, 'memory': 0.1, 'response': 0.1}
    layer_shares([])  ->  {'prompt': 0.0, 'tool': 0.0, 'memory': 0.0,
                           'response': 0.0}

    Ключи есть всегда, даже если слой ни разу не встретился: дашборд с
    пропадающими колонками читать невозможно.

    Пустой список — нули, а не деление на ноль.
    """
    totals = {layer: 0 for layer in LAYERS}
    for call in calls:
        for layer in LAYERS:
            totals[layer] += call["layers"].get(layer, 0)
    grand = sum(totals.values())
    if not grand:
        return {layer: 0.0 for layer in LAYERS}
    return {layer: totals[layer] / grand for layer in LAYERS}


def attribute(calls, dimension, prices=PRICES):
    """Разложить счёт по измерению. Словарь: значение измерения -> доллары.

    Вызовы без нужного тега попадают в корзину "untagged" — так видно цену
    ретроактивного тегирования, а не «немного расходов куда-то делось».

    attribute([{"tenant_id": "t1", "route": "haiku",
                "layers": {"prompt": 1000, "response": 100}}], "tenant_id")
        ->  {'t1': 0.0012}
    attribute([{"route": "haiku", "layers": {"prompt": 1000}}], "tenant_id")
        ->  {'untagged': 0.0008}

    Измерения из урока: user_id, task_id, tenant_id.
    """
    out = {}
    for call in calls:
        key = call.get(dimension) or "untagged"
        out[key] = out.get(key, 0.0) + call_cost(call, prices)
    return out


def daily_totals(calls, prices=PRICES):
    """Расход по дням. Словарь: дата "YYYY-MM-DD" -> доллары.

    daily_totals([{"day": "2026-08-01", "route": "haiku",
                   "layers": {"prompt": 1000}}])
        ->  {'2026-08-01': 0.0008}

    Дни без вызовов в словарь не попадают — их отсутствие само по себе
    сигнал, и подставлять туда нули нельзя: ноль расхода и отсутствие
    телеметрии — разные вещи.
    """
    out = {}
    for call in calls:
        day = call["day"]
        out[day] = out.get(day, 0.0) + call_cost(call, prices)
    return out


def zscore(value, history):
    """На сколько стандартных отклонений value отстоит от истории.

    zscore(100.0, [50.0, 52.0, 48.0, 50.0, 50.0])  ->  примерно 35.36
    zscore(51.0, [50.0] * 5)                       ->  inf
    zscore(50.0, [50.0] * 5)                       ->  0.0
    zscore(51.0, [50.0])                           ->  0.0

    Две ловушки:

    1. Нулевое стандартное отклонение. Если value равно ровному baseline,
       расстояние равно 0. Любое ненулевое отклонение от идеально ровной
       истории имеет бесконечный z-score со знаком отклонения: замена sd на
       единицу произвольно вводит окно в один доллар и прячет малые сдвиги.
    2. История короче двух значений. Дисперсии ещё нет, аномалий тоже.

    Короткая история даёт 0.0: baseline ещё не определён.
    """
    n = len(history)
    if n < 2:
        return 0.0
    mean = sum(history) / n
    # выборочная дисперсия (n-1): базлайн — это выборка, а не вся генеральная
    variance = sum((x - mean) ** 2 for x in history) / (n - 1)
    sd = variance ** 0.5
    if sd == 0.0:
        if value == mean:
            return 0.0
        return float("inf") if value > mean else float("-inf")
    return (value - mean) / sd


def anomaly_days(daily, threshold=4.0, min_history=5, min_usd=0.0):
    """Дни, где расход аномально вырос. Отсортированный список дат.

    День сравнивается только с ПРЕДЫДУЩИМИ днями: сравнивать с будущим —
    значит получить детектор, который в проде не работает.

    anomaly_days({"2026-08-0%d" % d: 10.0 for d in range(1, 7)})  ->  []

    min_history — сколько дней должно накопиться, прежде чем судить (в уроке
    это 5). min_usd отсекает копеечные дни: относительный скачок с $0.60 до
    $0.97 формально огромен, но будить из-за него человека незачем.
    """
    days = sorted(daily)
    out = []
    for i, day in enumerate(days):
        history = [daily[d] for d in days[:i]]
        if len(history) < min_history:
            continue
        if daily[day] < min_usd:
            continue
        if zscore(daily[day], history) > threshold:
            out.append(day)
    return out


def forecast_month(daily, now, month_days):
    """Прогноз расхода до конца месяца по неполным данным.

    daily — расход по уже прожитым дням месяца, now — последний наблюдённый
    день ("YYYY-MM-DD"), month_days — сколько дней в этом месяце.

    Ловушка: линейная экстраполяция total / прожито * month_days. Трафик
    продукта в выходные проседает, и прогноз, снятый в пятницу, завышает
    счёт, а снятый в воскресенье — занижает. Считай отдельное среднее по
    будням и по выходным и достраивай оставшиеся дни по их типу.

    Август 2026 начинается с субботы. Наблюдали 14 дней: будни по $100,
    выходные по $10 — это 10 будней и 4 выходных, итого $1040.
    forecast_month(...)  ->  2200.0   (ещё 11 будней и 6 выходных)
    Линейная экстраполяция дала бы 1040 / 14 * 31 = 2302.86.

    Если один из типов дней ещё не наблюдался, брать для него общее среднее —
    выдумывать нечего.
    """
    last = date.fromisoformat(now[:10])
    values = list(daily.values())
    overall = sum(values) / len(values) if values else 0.0
    weekday = [v for d, v in daily.items() if date.fromisoformat(d[:10]).weekday() < 5]
    weekend = [v for d, v in daily.items() if date.fromisoformat(d[:10]).weekday() >= 5]
    mean_weekday = sum(weekday) / len(weekday) if weekday else overall
    mean_weekend = sum(weekend) / len(weekend) if weekend else overall

    total = sum(values)
    for day_number in range(last.day + 1, month_days + 1):
        future = date(last.year, last.month, day_number)
        total += mean_weekday if future.weekday() < 5 else mean_weekend
    return total


def enforcement_action(spend_today, history, policy):
    """Какую ступень лестницы включить. Одна из четырёх строк.

    'kill_switch'  — z-score расхода выше policy["kill_z"];
    'cap_alert'    — превышен дневной cap (контракт * spend_cap_multiplier);
    'rate_limit'   — превышен сам контракт;
    'ok'           — ничего не превышено.

    policy = {"contracted_daily_usd": 100.0, "spend_cap_multiplier": 2.0,
              "kill_z": 4.0, "min_history": 5}
    enforcement_action(50.0, [], policy)                  ->  'ok'
    enforcement_action(150.0, [], policy)                 ->  'rate_limit'
    enforcement_action(250.0, [], policy)                 ->  'cap_alert'
    enforcement_action(900.0, [50.0, 52.0, 48.0, 51.0, 49.0, 50.0], policy)
        ->  'kill_switch'

    Порядок проверок важен: kill switch старше cap, cap старше rate limit.
    Тенант, у которого рвануло в двадцать раз, должен получить паузу, а не
    вежливое письмо от customer success.
    """
    min_history = policy.get("min_history", 5)
    z = zscore(spend_today, history) if len(history) >= min_history else 0.0
    if z > policy["kill_z"]:
        return "kill_switch"
    if spend_today > policy["contracted_daily_usd"] * policy["spend_cap_multiplier"]:
        return "cap_alert"
    if spend_today > policy["contracted_daily_usd"]:
        return "rate_limit"
    return "ok"
