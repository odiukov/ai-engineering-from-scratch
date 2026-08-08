"""
Бюджеты действий, лимиты итераций и губернаторы стоимости

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l13-cost-governors
Разбор:  /check-code p15-l13-cost-governors
"""

DOLLARS_PER_KTOK = 0.003
DEFAULT_LIMITS = {
    "max_tokens_per_request": 10_000,
    "max_turns": 200,
    "max_budget_usd": 50.0,
    "velocity_usd_per_min": 5.0,
    "velocity_window_min": 10.0,
    "monthly_cap_usd": 500.0,
}
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
    raise NotImplementedError


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
    raise NotImplementedError


def new_ledger():
    """Пустая книга учёта расходов сессии.

    new_ledger()["turns"]   ->  0
    new_ledger()["usd"]     ->  0.0

    Поля: turns (сколько ходов сделано), tokens (сумма токенов),
    usd (сумма денег), history (список пар (минута, накопленные доллары)),
    stopped_by (какой лимит остановил сессию, пока None).
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def first_breached_cap(ledger, limits, now_min):
    """Имя первого пробитого лимита или None, если все в порядке.

    first_breached_cap(new_ledger(), DEFAULT_LIMITS, 0.0)  ->  None

    Порядок обхода — CAP_ORDER. Лимит, которого нет в словаре limits,
    просто пропускается: так собирается конфигурация «velocity выключен».

    Сравнение везде >=, а не >: лимит в 200 ходов означает, что двухсотый
    ход — последний, а не первый разрешённый сверх.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
