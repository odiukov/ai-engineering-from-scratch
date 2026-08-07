"""
Kill switch, circuit breaker и canary token — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Начальное состояние выключателя. Живёт ВНЕ агента: агент его читает, но не
# пишет. В продакшне это feature flag, ключ в Redis или подписанный конфиг.
KILL_SWITCH_OFF = {
    "engaged": False,
    "reason": None,
    "engaged_at": None,
    "released_by": None,
    "released_at": None,
}

# Начальное состояние прерывателя (Nygard, 2007): closed -> open -> half_open.
BREAKER_CLOSED = {
    "state": "closed",
    "recent": (),        # хвост последних action_key
    "fails": 0,          # подряд идущих неудач
    "opened_at": None,
    "probes_left": 0,    # сколько пробных вызовов осталось в half_open
}

# Приманки в рабочем каталоге. У агента нет ни одной законной причины их
# читать, поэтому сам факт чтения — сигнал тревоги.
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
    if not reason:
        raise ValueError("kill switch needs a non-empty reason")
    if switch["engaged"]:
        # идемпотентно: копия того же состояния, первопричина сохранена
        return dict(switch)
    return dict(switch, engaged=True, reason=reason, engaged_at=now,
                released_by=None, released_at=None)


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
    if not operator or not note:
        raise ValueError("release needs an operator and a note")
    if not switch["engaged"]:
        return dict(switch)
    return dict(switch, engaged=False, released_by=operator, released_at=now,
                reason=switch["reason"])


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
    state = breaker["state"]

    if state == "open":
        if now - breaker["opened_at"] < cooldown_min:
            return False, dict(breaker)
        # охлаждение прошло: пускаем пробные вызовы
        state = "half_open"
        breaker = dict(breaker, state="half_open", probes_left=probes)

    if state == "half_open":
        if not ok:
            return True, dict(breaker, state="open", opened_at=now, probes_left=0)
        left = breaker["probes_left"] - 1
        if left <= 0:
            return True, dict(BREAKER_CLOSED)
        return True, dict(breaker, probes_left=left)

    recent = (breaker["recent"] + (action_key,))[-threshold:]
    fails = 0 if ok else breaker["fails"] + 1
    tripped = (
        len(recent) >= threshold and all(a == recent[0] for a in recent)
    ) or fails >= threshold
    if tripped:
        return False, dict(breaker, state="open", recent=recent, fails=fails,
                           opened_at=now, probes_left=0)
    return True, dict(breaker, state="closed", recent=recent, fails=fails)


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
    hits = []
    for i, a in enumerate(actions, 1):
        if a.get("kind") == "read" and a.get("payload") in canaries:
            hits.append((i, a["payload"]))
    return tuple(hits)


def ewma(values, alpha):
    """Экспоненциально взвешенное среднее ряда. Вернуть список той же длины.

    ewma([1.0, 1.0, 1.0], 0.5)  ->  [1.0, 1.0, 1.0]
    ewma([0.0, 1.0], 0.5)       ->  [0.0, 0.5]

    Формула: s[0] = values[0], s[i] = alpha * values[i] + (1 - alpha) * s[i-1].

    Это и есть «адаптируется к дрейфу»: базовая линия сама уползает вслед за
    данными. Для честного роста нагрузки — фича, для медленной атаки — дыра.
    """
    out = []
    prev = None
    for v in values:
        prev = v if prev is None else alpha * v + (1 - alpha) * prev
        out.append(prev)
    return out


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
    if warmup < 1:
        raise ValueError("warmup must be >= 1")
    smooth = ewma(values, alpha)
    # сумму отклонений копим на ходу: пересчитывать её на каждом шаге —
    # это O(n^2), а детектор гоняют по длинным журналам
    dev_sum = sum(abs(values[j] - smooth[j]) for j in range(min(warmup, len(values))))
    for i in range(warmup, len(values)):
        base = smooth[i - 1]
        sigma = dev_sum / i
        if sigma == 0:
            if values[i] != base:
                return i
        elif abs(values[i] - base) > k * sigma:
            return i
        dev_sum += abs(values[i] - smooth[i])
    return None


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
    left = 0
    for i, t in enumerate(times):
        while times[left] <= t - window_min:
            left += 1
        if i - left + 1 > max_calls:
            return i
    return None


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
    report = {"executed": 0, "stopped_by": None, "canary": (), "breaker": dict(breaker)}
    hits = []
    for i, a in enumerate(actions, 1):
        if switch["engaged"]:
            report["stopped_by"] = "kill_switch"
            break
        key = f"{a.get('kind')}:{a.get('payload')}"
        allowed, report["breaker"] = breaker_step(
            report["breaker"], key, a.get("ok", True), float(i),
            threshold=threshold, cooldown_min=cooldown_min,
        )
        if not allowed:
            report["stopped_by"] = "circuit_breaker"
            break
        if a.get("kind") == "read" and a.get("payload") in canaries:
            hits.append((i, a["payload"]))
        report["executed"] += 1
    report["canary"] = tuple(hits)
    return report
