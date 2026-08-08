"""
Batch API: очередь, скидка и SLA завершения

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l15-batch-apis
Разбор:  /check-code p17-l15-batch-apis
"""

import math

PRICE_INPUT = 3.00
PRICE_CACHED_READ = 0.30
PRICE_OUTPUT = 15.00
WRITE_PREMIUM = 1.25
BATCH_DISCOUNT = 0.50
BATCH_CACHE_POLICY = {
    "anthropic": "stack",
    "vertex-gemini": "cache_precedence",
}
INTERACTIVE_MAX_S = 60
SEMI_MAX_S = 3600
BATCH_SLA_H = 24


class BatchError(Exception):
    """Очередь или тариф спрошены о невозможном.

    Свой класс, а не ValueError и не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """
    pass


def sync_cost(n, prefix_tokens, unique_tokens, output_tokens):
    """Счёт за n синхронных вызовов без всякого кэша — базовая линия.

    sync_cost(1, 4000, 2000, 200)   ->  0.021
    sync_cost(50_000, 4000, 2000, 200)  ->  1050.0

    Разбор: (4000+2000)/1e6 * 3.00 = 0.018 за вход, 200/1e6 * 15.00 = 0.003
    за выход. Общий системный промпт здесь оплачивается заново каждый раз —
    в этом и смысл базовой линии.

    Ловушка: n < 0 — BatchError. Ноль допустим, это пустой прогон.
    """
    raise NotImplementedError


def cached_cost(n, prefix_tokens, unique_tokens, output_tokens):
    """То же, но общий префикс кэшируется: первый пишет, остальные читают.

    cached_cost(50_000, 4000, 2000, 200)  ->  примерно 510.01

    Первый вызов платит за префикс с премией 1.25x вместо базовой цены,
    остальные n-1 — цену чтения, в десять раз дешевле входа. Уникальная
    часть промпта и выход не кэшируются никогда.

    Ловушка: премия платится ВМЕСТО базовой цены, а не вдобавок к ней. Если
    сложить обе, первый вызов подорожает вдвое и счёт разойдётся с рейт-картой.
    """
    raise NotImplementedError


def batch_cost(n, prefix_tokens, unique_tokens, output_tokens, cached,
               provider="anthropic"):
    """Счёт по batch с явной политикой провайдера для prompt cache.

    batch_cost(50_000, 4000, 2000, 200, False)  ->  525.0
    batch_cost(50_000, 4000, 2000, 200, True)   ->  примерно 255.0

    У Anthropic скидки складываются. У Vertex Gemini cache price takes
    precedence: cached prefix оплачивается по cache-тарифу без дополнительной
    batch-скидки, а уникальный вход и выход всё ещё получают -50%.

    Неизвестный provider — BatchError: молча выбрать финансовую политику
    нельзя.
    """
    raise NotImplementedError


def submit(queue, job_id, n_requests, submitted_h):
    """Положить задание в очередь. Возвращает НОВЫЙ список, вход не портит.

    submit([], "night-1", 20_000, 1.0)  ->  [{'job_id': 'night-1', ...}]

    Ловушка: очередь-аргумент менять нельзя. Тест «вход не изменился» есть,
    и он не про чистоту стиля: очередь дня обычно строят один раз и потом
    гоняют по ней несколько сценариев планирования.

    BatchError на дубль job_id, на n_requests <= 0 и на отрицательное время.
    """
    raise NotImplementedError


def drain_window(queue, window_start_h, window_end_h, throughput_per_h):
    """Прогнать очередь через ежедневное окно недозагрузки.

    Окно повторяется каждые сутки: [window_start_h, window_end_h) часов от
    начала суток. Задания берутся по (submitted_h, job_id), одно за другим —
    провайдер не начнёт следующее, пока не закончил текущее.

    Возвращает список {"job_id", "finished_h", "wait_h"} в порядке запуска.

    Окно 0-6, пропускная 10000 запросов в час:
      задание на 20000 запросов, подано в 1.0  ->  finished_h 3.0,  wait_h 2.0
      задание на 10000 запросов, подано в 10.0 ->  finished_h 25.0, wait_h 15.0
      задание на 100000 запросов, подано в 1.0 ->  finished_h 29.0, wait_h 28.0

    Третий случай — про то, почему SLA это 24 часа, а не «2-6 часов как
    обычно»: работа не влезла в одно окно и доехала только следующей ночью.

    Ловушка: окно закрылось посреди задания — работу надо ПРОДОЛЖИТЬ в
    следующем окне, а не начать заново и не досчитать за границей окна.
    """
    raise NotImplementedError


def sla_report(completions, sla_h=BATCH_SLA_H):
    """Уложились ли задания в обещание провайдера.

    Возвращает dict:
      total, met, missed,
      met_fraction  — доля уложившихся, 0.0 на пустом входе,
      max_wait_h    — худшее ожидание, 0.0 на пустом входе,
      worst_job     — id худшего задания или None.

    sla_report([])["met_fraction"]  ->  0.0

    Ловушка на границе: ожидание ровно 24.0 часа — это УЛОЖИЛИСЬ. SLA
    формулируется как «в течение 24 часов», сравнение нестрогое.
    """
    raise NotImplementedError


def triage(latency_budget_s):
    """Полоса по бюджету задержки: 'interactive' | 'semi' | 'batch'.

    triage(5)       ->  'interactive'   (пользователь смотрит на спиннер)
    triage(600)     ->  'semi'          (вернётся через несколько минут)
    triage(86_400)  ->  'batch'         (нужно к утру)

    Границы нестрогие: ровно 60 с — ещё interactive, ровно 3600 с — ещё semi.

    Ловушка не в коде, а в голове: «продакшен» — это не спецификация
    задержки. Спецификация — это SLA. Половина синхронных пайплайнов
    считаются интерактивными только потому, что их так назвали.

    BatchError на бюджет <= 0.
    """
    raise NotImplementedError


def lane_decision(n, prefix_tokens, unique_tokens, output_tokens, latency_budget_s,
                  provider="anthropic"):
    """Выбрать полосу и посчитать, сколько это стоит и сколько потеряно.

    Возвращает dict:
      lane            — из triage,
      cost            — счёт в выбранной полосе,
      baseline_cost   — синхронно и без кэша,
      best_cost       — batch + кэш по политике provider, недостижимый минимум,
      saving_usd/pct  — экономия против baseline,
      forgone_usd     — сколько оставлено на столе из-за требования к задержке.

    Кэш доступен во всех полосах, скидка batch — только в 'batch'. У Vertex
    Gemini cache-тариф на общий префикс имеет приоритет над batch-скидкой.

    lane_decision(50_000, 4000, 200, 100, 5)["lane"]       ->  'interactive'
    lane_decision(50_000, 4000, 200, 100, 86_400)["lane"]  ->  'batch'

    Главное свойство: forgone_usd > 0 у interactive и semi и ровно 0 у batch.
    Batch выгоден ровно тогда, когда задержка допустима — не «почти всегда»,
    а именно тогда.
    """
    raise NotImplementedError
