"""
FinOps для LLM: атрибуция, аномалии, прогноз счёта

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l27-finops-llms
Разбор:  /check-code p17-l27-finops-llms
"""

from datetime import date

INPUT_LAYERS = ("prompt", "tool", "memory")
OUTPUT_LAYER = "response"
LAYERS = INPUT_LAYERS + (OUTPUT_LAYER,)
PRICES = {
    "haiku": {"input": 0.80, "output": 4.00},
    "sonnet": {"input": 3.00, "output": 15.00},
    "opus": {"input": 15.00, "output": 75.00},
}
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def daily_totals(calls, prices=PRICES):
    """Расход по дням. Словарь: дата "YYYY-MM-DD" -> доллары.

    daily_totals([{"day": "2026-08-01", "route": "haiku",
                   "layers": {"prompt": 1000}}])
        ->  {'2026-08-01': 0.0008}

    Дни без вызовов в словарь не попадают — их отсутствие само по себе
    сигнал, и подставлять туда нули нельзя: ноль расхода и отсутствие
    телеметрии — разные вещи.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def anomaly_days(daily, threshold=4.0, min_history=5, min_usd=0.0):
    """Дни, где расход аномально вырос. Отсортированный список дат.

    День сравнивается только с ПРЕДЫДУЩИМИ днями: сравнивать с будущим —
    значит получить детектор, который в проде не работает.

    anomaly_days({"2026-08-0%d" % d: 10.0 for d in range(1, 7)})  ->  []

    min_history — сколько дней должно накопиться, прежде чем судить (в уроке
    это 5). min_usd отсекает копеечные дни: относительный скачок с $0.60 до
    $0.97 формально огромен, но будить из-за него человека незачем.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
