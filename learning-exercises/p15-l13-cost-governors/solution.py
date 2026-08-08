"""
Бюджеты действий, лимиты итераций и губернаторы стоимости — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Цена за тысячу токенов (input+output вперемешку) для модели класса Sonnet.
DOLLARS_PER_KTOK = 0.003

# Стек лимитов из Microsoft Agent Governance Toolkit и Claude Code Agent SDK.
# Каждый слой ловит свой класс отказа: max_tokens_per_request — раздутый
# ответ, max_turns — бесконечный цикл рассуждений, max_budget_usd — сессию
# целиком, velocity_* — runaway loop за минуты, monthly_cap_usd — медленную
# течь за дни.
DEFAULT_LIMITS = {
    "max_tokens_per_request": 10_000,
    "max_turns": 200,
    "max_budget_usd": 50.0,
    "velocity_usd_per_min": 5.0,
    "velocity_window_min": 10.0,
    "monthly_cap_usd": 500.0,
}

# Порядок проверки лимитов. Он же порядок, в котором first_breached_cap
# называет сработавший слой, если их пробило сразу несколько.
CAP_ORDER = (
    "max_turns",
    "max_budget_usd",
    "velocity_usd_per_min",
    "monthly_cap_usd",
)


def tokens_to_usd(tokens, dollars_per_ktok=DOLLARS_PER_KTOK):
    """Перевести токены в доллары по цене за тысячу токенов.

    tokens_to_usd(1000)         ->  0.003
    tokens_to_usd(8000)         ->  0.024
    tokens_to_usd(500, 0.015)   ->  0.0075

    Считать надо в тысячах: tokens / 1000 * цена. Частая ошибка — умножить
    на цену напрямую и получить счёт в тысячу раз больше.
    """
    return (tokens / 1000.0) * dollars_per_ktok


def cap_request_tokens(requested, max_tokens):
    """Обрезать запрос по потолку max_tokens ДО того, как он оплачен.

    cap_request_tokens(8000, 10000)   ->  8000
    cap_request_tokens(80000, 10000)  ->  10000
    cap_request_tokens(80000, None)   ->  80000   (потолок не задан)

    Смысл слоя в порядке: сначала режем, потом списываем. Если списать
    сначала, а обрезать потом, деньги за нелимитированный ответ уже ушли —
    ровно так и выглядит Denial of Wallet.

    Отрицательный запрос — это баг вызывающего кода, а не ноль токенов:
    бросай ValueError.
    """
    if requested < 0:
        raise ValueError("requested tokens must be >= 0")
    if max_tokens is None:
        return requested
    return min(requested, max_tokens)


def new_ledger():
    """Пустая книга учёта расходов сессии.

    new_ledger()["turns"]   ->  0
    new_ledger()["usd"]     ->  0.0

    Поля: turns (сколько ходов сделано), tokens (сумма токенов),
    usd (сумма денег), history (список пар (минута, накопленные доллары)),
    stopped_by (какой лимит остановил сессию, пока None).
    """
    return {"turns": 0, "tokens": 0, "usd": 0.0, "history": [], "stopped_by": None}


def record_turn(ledger, tokens, now_min, dollars_per_ktok=DOLLARS_PER_KTOK):
    """Списать один ход и вернуть НОВУЮ книгу учёта. Старую не трогать.

    record_turn(new_ledger(), 1000, 0.5)["usd"]      ->  0.003
    record_turn(new_ledger(), 1000, 0.5)["history"]  ->  [(0.5, 0.003)]

    Возвращать копию, а не править аргумент, — не педантизм. Тесты
    сценариев гоняют один и тот же ledger по разным веткам, и общая
    изменяемая история склеила бы их расходы.

    Время приходит параметром now_min, а не из time.time(): иначе
    velocity-лимит невозможно проверить воспроизводимо.
    """
    usd = ledger["usd"] + tokens_to_usd(tokens, dollars_per_ktok)
    return {
        "turns": ledger["turns"] + 1,
        "tokens": ledger["tokens"] + tokens,
        "usd": usd,
        # список пересобираем: общий list превратил бы «копию» в алиас
        "history": list(ledger["history"]) + [(now_min, usd)],
        "stopped_by": ledger["stopped_by"],
    }


def window_velocity(history, now_min, window_min):
    """Скорость трат в долларах за минуту внутри скользящего окна.

    window_velocity([(1.0, 0.5), (2.0, 1.0)], 2.0, 10.0)  ->  0.5
    window_velocity([], 5.0, 10.0)                        ->  0.0

    Как считать: базовая точка — последняя запись, которая старше окна
    (её накопленная сумма уже «за пределами»), либо 0.0, если такой нет.
    Потраченное в окне = накопленное сейчас минус база.

    Ловушка прогрева: делить на ширину окна нельзя. В первые минуты сессии
    прошло меньше времени, чем window_min, и деление на 10 занизит скорость
    в разы — runaway loop проскочит. Дели на min(window_min, now_min).
    """
    if not history or now_min <= 0:
        return 0.0
    cutoff = now_min - window_min
    baseline = 0.0
    for t, usd in history:
        if t <= cutoff:
            baseline = usd
        else:
            break
    spent = history[-1][1] - baseline
    elapsed = min(window_min, now_min)
    return spent / elapsed


def first_breached_cap(ledger, limits, now_min):
    """Имя первого пробитого лимита или None, если все в порядке.

    first_breached_cap(new_ledger(), DEFAULT_LIMITS, 0.0)  ->  None

    Порядок обхода — CAP_ORDER. Лимит, которого нет в словаре limits,
    просто пропускается: так собирается конфигурация «velocity выключен».

    Сравнение везде >=, а не >: лимит в 200 ходов означает, что двухсотый
    ход — последний, а не первый разрешённый сверх.
    """
    for name in CAP_ORDER:
        if name not in limits:
            continue
        if name == "max_turns" and ledger["turns"] >= limits[name]:
            return name
        if name == "max_budget_usd" and ledger["usd"] >= limits[name]:
            return name
        if name == "monthly_cap_usd" and ledger["usd"] >= limits[name]:
            return name
        if name == "velocity_usd_per_min":
            window = limits.get("velocity_window_min", 10.0)
            if window_velocity(ledger["history"], now_min, window) > limits[name]:
                return name
    return None


def run_session(turn_tokens, limits=None, dollars_per_ktok=DOLLARS_PER_KTOK,
                seconds_per_turn=30.0):
    """Прогнать сессию по списку запрошенных токенов и вернуть книгу учёта.

    run_session([1000, 1000], {"max_turns": 1})["turns"]       ->  1
    run_session([1000, 1000], {"max_turns": 1})["stopped_by"]  ->  'max_turns'
    run_session([80000], {"max_tokens_per_request": 10000})["tokens"]  ->  10000

    Один ход: обрезать запрос по max_tokens_per_request, зарезервировать его
    стоимость под жёсткими денежными лимитами, сдвинуть часы, списать и
    проверить остальные лимиты. Если следующий ход превысил бы dollar cap,
    отказываем ДО вызова и списания: kill switch после оплаты уже опоздал.

    Ровно до лимита тратить можно. Такой последний ход записывается, после
    чего first_breached_cap останавливает сессию на границе. Ход, который
    вывел бы сумму выше лимита, в turns, tokens и history не попадает.

    limits=None означает DEFAULT_LIMITS.
    """
    limits = DEFAULT_LIMITS if limits is None else limits
    ledger = new_ledger()
    now_min = 0.0
    for requested in turn_tokens:
        granted = cap_request_tokens(requested, limits.get("max_tokens_per_request"))
        projected_usd = ledger["usd"] + tokens_to_usd(granted, dollars_per_ktok)
        hard_stop = next(
            (
                name
                for name in ("max_budget_usd", "monthly_cap_usd")
                if name in limits and projected_usd > limits[name]
            ),
            None,
        )
        if hard_stop is not None:
            ledger = dict(ledger, stopped_by=hard_stop)
            break
        now_min += seconds_per_turn / 60.0
        ledger = record_turn(ledger, granted, now_min, dollars_per_ktok)
        breached = first_breached_cap(ledger, limits, now_min)
        if breached is not None:
            # dict(...) вместо присваивания: record_turn уже вернул свежий
            # словарь, но так видно, что функция никого не мутирует
            ledger = dict(ledger, stopped_by=breached)
            break
    return ledger


def budget_warnings(ledger, limits, warn_at=0.8):
    """Лимиты, съеденные на warn_at и больше, но ещё не пробитые.

    budget_warnings({"turns": 8, ...}, {"max_turns": 10})  ->  ('max_turns',)
    budget_warnings(new_ledger(), DEFAULT_LIMITS)          ->  ()

    Это тот самый «alert on week-over-week growth», которого не хватало в
    разобранном кейсе с ростом счёта $1200 -> $4800: лимит ещё не сработал,
    а предупредить уже пора.

    Уже пробитый лимит в предупреждения НЕ попадает — он не предупреждение,
    а срабатывание.
    """
    warnings = []
    for name in CAP_ORDER:
        if name not in limits:
            continue
        if name == "max_turns":
            used = ledger["turns"] / limits[name]
        elif name == "velocity_usd_per_min":
            window = limits.get("velocity_window_min", 10.0)
            now_min = ledger["history"][-1][0] if ledger["history"] else 0.0
            rate = window_velocity(ledger["history"], now_min, window)
            used = rate / limits[name]
        else:
            used = ledger["usd"] / limits[name]
        if warn_at <= used < 1.0:
            warnings.append(name)
    return tuple(warnings)
