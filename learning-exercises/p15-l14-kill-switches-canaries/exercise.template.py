"""
Kill switch, circuit breaker и canary token

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p15-l14-kill-switches-canaries
Разбор:  /check-code p15-l14-kill-switches-canaries
"""

KILL_SWITCH_OFF = {
    "engaged": False,
    "reason": None,
    "engaged_at": None,
    "released_by": None,
    "released_at": None,
}
BREAKER_CLOSED = {
    "state": "closed",
    "recent": (),        # хвост последних action_key
    "fails": 0,          # подряд идущих неудач
    "opened_at": None,
    "probes_left": 0,    # сколько пробных вызовов осталось в half_open
}
CANARY_PATHS = (
    "~/.env.canary",
    "~/notes/fake-credentials.txt",
)


def engage_kill_switch(switch, reason, now):
    """Оператор дёргает рубильник. Возвращает НОВОЕ состояние выключателя.

    engage_kill_switch(KILL_SWITCH_OFF, "runaway loop", 10.0)["engaged"]  ->  True
    engage_kill_switch(KILL_SWITCH_OFF, "", 10.0)  ->  ValueError

    Свойство, ради которого это отдельная функция: идемпотентность. Второй
    вызов по уже сработавшему выключателю НЕ перетирает причину и время
    первого срабатывания — иначе разбор инцидента потеряет момент, когда
    агента остановили на самом деле.

    Пустая причина запрещена: выключатель без записи в журнале нельзя
    аудировать (EU AI Act, статья 14).
    """
    raise NotImplementedError


def release_kill_switch(switch, operator, note, now):
    """Человек снимает блокировку. Возвращает НОВОЕ состояние выключателя.

    release_kill_switch(engaged, "alice", "patched the loop", 20.0)["engaged"]
        ->  False
    release_kill_switch(engaged, "", "note", 20.0)  ->  ValueError

    Правило урока: обратное включение — явное человеческое действие, а не
    автоматический таймаут. Поэтому обязательны и имя оператора, и запись
    о том, что изменилось. Без них — ValueError.

    Снятие уже снятого выключателя — не ошибка, а безобидный no-op.
    """
    raise NotImplementedError


def breaker_step(breaker, action_key, ok, now, threshold=5,
                 cooldown_min=10.0, probes=1):
    """Один шаг прерывателя. Вернуть (allowed, новый breaker).

    breaker_step(BREAKER_CLOSED, "read:a", True, 0.0)[0]  ->  True

    Пять одинаковых action_key подряд ИЛИ пять неудач подряд открывают
    прерыватель, и вызов, который его открыл, уже НЕ выполняется.

    Состояния (Nygard, 2007):
      closed    — пропускаем, копим хвост последних вызовов;
      open      — блокируем; когда с opened_at прошло cooldown_min,
                  переходим в half_open;
      half_open — probes пробных вызовов разрешены. Каждый удачный
                  уменьшает счётчик, на нуле прерыватель закрывается.
                  Любая неудача снова открывает его с новым отсчётом.

    Пробный вызов РАЗРЕШЁН даже когда он проваливается: иначе о провале
    было бы неоткуда узнать.

    Ловушка: пока прерыватель открыт, opened_at менять нельзя, иначе
    охлаждение будет продлеваться каждой попыткой и half_open не наступит
    никогда.
    """
    raise NotImplementedError


def canary_hits(actions, canaries=CANARY_PATHS):
    """Найти обращения к приманкам. Вернуть кортеж пар (номер хода, путь).

    canary_hits([{"kind": "read", "payload": "~/.env.canary"}])
        ->  ((1, '~/.env.canary'),)
    canary_hits([{"kind": "read", "payload": "README.md"}])  ->  ()

    Ходы нумеруются с единицы — так их видит журнал.

    Приманка срабатывает только на чтение (kind == "read"). Запись в файл,
    которого агент не касался, — другой класс события; здесь мы ловим
    именно попытку прочитать поддельный секрет.
    """
    raise NotImplementedError


def ewma(values, alpha):
    """Экспоненциально взвешенное среднее ряда. Вернуть список той же длины.

    ewma([1.0, 1.0, 1.0], 0.5)  ->  [1.0, 1.0, 1.0]
    ewma([0.0, 1.0], 0.5)       ->  [0.0, 0.5]

    Формула: s[0] = values[0], s[i] = alpha * values[i] + (1 - alpha) * s[i-1].

    Это и есть «адаптируется к дрейфу»: базовая линия сама уползает вслед за
    данными. Для честного роста нагрузки — фича, для медленной атаки — дыра.
    """
    raise NotImplementedError


def ewma_alarm(values, alpha, k, warmup=5):
    """Индекс первого значения, отскочившего от EWMA-базы больше чем на k сигм.

    ewma_alarm([1.0] * 10 + [50.0], 0.3, 4.0)  ->  10
    ewma_alarm([1.0] * 20, 0.3, 4.0)           ->  None

    База и сигма считаются по ряду ДО текущей точки (первые warmup точек
    пропускаем — на них статистики ещё нет). Сигма — среднее абсолютное
    отклонение от EWMA, чтобы не тянуть math.sqrt ради того же смысла.

    Если сигма нулевая (идеально ровный ряд), любое отличие от базы считаем
    тревогой: делить на ноль нельзя, а скачок на ровном ряду — это скачок.

    warmup обязан быть >= 1, иначе сигму не на что делить.
    """
    raise NotImplementedError


def hard_limit_breach(times, max_calls, window_min):
    """Индекс вызова, который пробил жёсткий лимит «не больше N за окно».

    hard_limit_breach([0.0, 1.0, 2.0], 5, 10.0)          ->  None
    hard_limit_breach([0.0, 0.1, 0.2, 0.3], 3, 10.0)     ->  3

    Для каждого вызова считаем, сколько вызовов попало в хвост окна
    (times[i] - window_min, times[i]] включая сам вызов. Как только их
    больше max_calls — возвращаем индекс.

    В отличие от EWMA этот детектор НЕ адаптируется: медленный дрейф,
    прошедший мимо статистики, всё равно упрётся в константу. Ровно это и
    есть «hard constitutional limit» из урока.

    times обязаны идти по неубыванию — это журнал, а не мешок чисел. Тогда
    хватает одного бегущего указателя вместо пересчёта окна на каждом шаге.
    """
    raise NotImplementedError


def run_trajectory(actions, switch=KILL_SWITCH_OFF, breaker=BREAKER_CLOSED,
                   canaries=CANARY_PATHS, threshold=5, cooldown_min=10.0):
    """Прогнать траекторию через три детектора и вернуть отчёт.

    run_trajectory([{"kind": "tool", "payload": "read:a"}])["executed"]  ->  1
    run_trajectory([...], switch=engaged)["stopped_by"]  ->  'kill_switch'

    Отчёт: executed (сколько действий выполнено), stopped_by (None,
    'kill_switch' или 'circuit_breaker'), canary (кортеж срабатываний
    приманок), breaker (итоговое состояние прерывателя).

    Порядок проверок принципиален: выключатель читается ПЕРЕД каждым
    действием, а не один раз на старте. Прерыватель — следующий. Приманка
    выполнение не останавливает: она поднимает тревогу, а не тормозит агента.

    Время хода берём как его номер в минутах — параметром, не из time.time().
    """
    raise NotImplementedError
