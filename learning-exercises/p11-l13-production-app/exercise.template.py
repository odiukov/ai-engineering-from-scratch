"""
Продакшен-приложение на LLM

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l13-production-app
Разбор:  /check-code p11-l13-production-app
"""

import hashlib
import math

MODEL_PRICING = {
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
FALLBACK_CHAIN = ("claude-sonnet-5", "gpt-4o", "gpt-4o-mini")
DEGRADED_TEXT = "Service temporarily unavailable. Please try again in a moment."


class ProviderError(Exception):
    """Провайдер не ответил: 429, 500 или таймаут.

    Свой класс, а не RuntimeError, специально: NotImplementedError — тоже
    RuntimeError, и тест `pytest.raises(RuntimeError)` прошёл бы зелёным на
    пустой заготовке, ничего не проверив.
    """
    pass


def estimate_tokens(text):
    """Грубая оценка числа токенов в тексте: слов x 4/3, но не меньше единицы.

    estimate_tokens("hello world")        ->  2
    estimate_tokens("a b c d e f")        ->  8
    estimate_tokens("")                   ->  1

    Настоящий токенизатор считает по BPE-словарю и в стандартную библиотеку не
    входит. Для прикидки бюджета правило «3 слова ~ 4 токена» ошибается на
    10-20% и этого хватает, чтобы поймать запрос на 50 000 токенов до того,
    как он придёт в API.
    """
    raise NotImplementedError


def request_cost(model, input_tokens, output_tokens, pricing):
    """Цена одного запроса в долларах по прайс-листу pricing.

    request_cost("gpt-4o", 1500, 400, MODEL_PRICING)       ->  0.00775
    request_cost("gpt-4o-mini", 1500, 400, MODEL_PRICING)  ->  0.000465

    pricing — словарь {модель: (цена за 1M входных, цена за 1M выходных)}.

    Ловушка: неизвестную модель нельзя молча оценивать по цене какой-нибудь
    соседней. Так рождаются счета-сюрпризы. Неизвестная модель — ValueError.
    """
    raise NotImplementedError


def backoff_delay(attempt, base=1.0, cap=10.0, rng=None):
    """Пауза перед попыткой номер attempt: экспонента с потолком плюс jitter.

    backoff_delay(0)  ->  0.0   (первая попытка идёт сразу)
    backoff_delay(1)  ->  1.0
    backoff_delay(2)  ->  2.0
    backoff_delay(3)  ->  4.0
    backoff_delay(9)  ->  10.0  (упёрлось в cap)

    rng — объект random.Random. Если его передали, к паузе добавляется jitter
    из диапазона [0, задержка/2). Без rng функция строго детерминирована.

    Зачем jitter: без него тысяча клиентов, отвалившихся в одну секунду,
    ретраится тоже в одну секунду и добивает провайдера повторно. С jitter
    их ретраи размазываются по интервалу.

    Ловушка: jitter обязан браться из ПЕРЕДАННОГО rng, а не из глобального
    random. Иначе один и тот же seed даёт разные прогоны и тест не повторить.
    """
    raise NotImplementedError


def retry_with_backoff(call, max_retries=3, base=1.0, cap=10.0, rng=None):
    """Повторяет call(attempt), пока тот бросает ProviderError.

    Возвращает кортеж (результат, число попыток, суммарная пауза в секундах).
    Когда попытки кончились — бросает ProviderError.

    retry_with_backoff(lambda a: "ok")            ->  ("ok", 1, 0.0)
    retry_with_backoff(падает_один_раз)           ->  ("ok", 2, 1.0)

    call принимает номер попытки (0, 1, 2, ...) — это позволяет тесту
    сымитировать «первые две попытки 500, третья успех».

    Никакого time.sleep здесь нет: паузу мы СЧИТАЕМ и возвращаем. В бою на
    её месте стоял бы await asyncio.sleep(delay), но тест, который реально
    спит четыре секунды, никто не станет запускать.
    """
    raise NotImplementedError


def call_with_fallback(models, call, max_retries=3, base=1.0, cap=10.0, rng=None):
    """Перебирает цепочку моделей, каждую — с ретраями. Никогда не бросает.

    Возвращает словарь с ключами:
      model         — модель, которая ответила, либо None
      text          — ответ либо DEGRADED_TEXT
      degraded      — True, если упала вся цепочка
      models_tried  — список моделей в порядке перебора

    call принимает (model, attempt) и либо возвращает текст, либо бросает
    ProviderError.

    Это graceful degradation из урока: падение вторичной системы не имеет
    права уронить основной поток. Пользователь всегда получает хоть что-то,
    пусть и от модели подешевле.
    """
    raise NotImplementedError


def ab_bucket(user_id, experiment, traffic_pct):
    """Ветка A/B-эксперимента для пользователя: "variant" или "control".

    ab_bucket("user_001", "chat_v2", 10)     ->  "control"
    ab_bucket("user_001", "other_exp", 10)   ->  "variant"
    ab_bucket("bob", "chat_v2", 20)          ->  "variant"

    Бакет считается как md5("<user_id>:<experiment>") mod 100.

    Почему хеш, а не random: пользователь обязан видеть одну и ту же ветку на
    всех своих запросах, иначе метрики эксперимента бессмысленны, а интерфейс
    мигает. Имя эксперимента входит в хеш, чтобы человек, попавший в вариант
    одного теста, не оказывался в варианте всех остальных.
    """
    raise NotImplementedError


def percentiles(values, ps):
    """Перцентили по методу ближайшего ранга. Возвращает {p: значение}.

    percentiles([1, 2, 3, 4], (50, 100))     ->  {50: 2, 100: 4}
    percentiles(list(range(1, 101)), (99,))  ->  {99: 99}

    Пустой список — ValueError: среднее по нулю запросов не бывает.
    p вне интервала (0, 100] — тоже ValueError.

    Зачем в проде именно перцентили, а не среднее: одна восьмисекундная
    хвостовая задержка растворяется в среднем, но именно она гонит
    пользователей прочь. P99 её видит, среднее — нет.
    """
    raise NotImplementedError


def summarize_requests(logs, pricing):
    """Сводка по журналу запросов — то, что уходит на дашборд.

    logs — список словарей с ключами model, input_tokens, output_tokens,
    latency_ms, cache_hit (bool), error (строка или None).

    Возвращает словарь: requests, total_cost_usd, avg_cost_usd,
    cache_hit_rate_pct, error_rate_pct, p50_latency_ms, p99_latency_ms,
    cost_by_model.

    Попадание в кэш стоит ноль и в cost_by_model уходит нулём — строку из
    журнала при этом не выбрасываем, иначе hit rate посчитать будет не по чему.

    Пустой журнал — ValueError. «Ноль запросов, всё хорошо» — худший вид
    зелёного дашборда.
    """
    raise NotImplementedError
