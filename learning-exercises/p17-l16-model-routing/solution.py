"""
Роутинг моделей: каскад, эскалация и цена качества — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

Чему это соответствует в настоящих системах:

    call_cost                <-  строка счёта провайдера: $/M на вход и выход
    pre_route                <-  классификатор перед вызовом: LiteLLM router,
                                 Portkey guards, RouteLLM (LMSYS)
    cascade                  <-  «cheap-first, escalate on low confidence»:
                                 дешёвая модель, порог уверенности, повтор
                                 на frontier
    cascade_break_even_rate  <-  ответ на вопрос «при какой доле эскалаций
                                 каскад перестаёт экономить»
    run_workload             <-  дашборд роутера: blended cost, escalation
                                 rate, quality loss
    drift_alarm              <-  online quality gate, который ловит cheap
                                 creep: доля эскалаций поползла вверх,
                                 качество вниз, и никто не заметил квартал

Ни сети, ни моделей: роутер — это арифметика над ценой и уверенностью, и
она моделируется честно. Всё случайное приходит параметрами, глобального
random внутри функций нет.

Важно про similarity: это ЧИСЛО, посчитанное где-то снаружи настоящим
эмбеддером. Здесь оно приходит готовым параметром. Хеш строки на это место
подставлять нельзя — «похожесть» от хеша не бывает, у хеша нет топологии.

Цены — снимок рейт-карты 2026, они дрейфуют.
"""

# $/M токенов. Дешёвая модель класса Haiku против frontier класса GPT-5.
MODELS = {
    "cheap": {"input": 0.40, "output": 2.00},
    "frontier": {"input": 3.00, "output": 15.00},
}

# Промпт длиннее — связность падает у дешёвой модели, отправляем на frontier.
LONG_PROMPT_TOKENS = 4000

# Косинус к размеченному «трудному» ведру, выше которого эскалируем сразу.
HARD_SIMILARITY = 0.88

# Классы задач, где дешёвая модель заведомо не тянет.
HARD_TASKS = ("code", "math", "planning")

# Порог уверенности первого прогона: ниже — второй заход на frontier.
CASCADE_THRESHOLD = 0.75

# Каскад, который эскалирует чаще этого, — сигнал, что дешёвую модель
# переиспользуют: маршрут стоит пересобрать.
MAX_ESCALATION_RATE = 0.30

# Допустимая просадка качества против frontier-only, в процентных пунктах.
MAX_QUALITY_LOSS_PCT = 2.0

# Стратегии, которые понимает route_request.
STRATEGIES = ("frontier_only", "cheap_only", "pre_route", "cascade")


class RoutingError(Exception):
    """Роутер спрошен о невозможном: чужая модель, чужая стратегия, кривой вход.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """


def make_request(req_id, task, prompt_tokens, output_tokens,
                 similarity=0.0, cheap_confidence=1.0, cheap_correct=True):
    """Один запрос со всеми сигналами роутинга.

    make_request("q1", "chat", 300, 80)["task"]                 ->  'chat'
    make_request("q2", "code", 300, 80, cheap_confidence=0.4)   ->  dict
    make_request("q3", "chat", 300, 80, similarity=1.7)         ->  RoutingError

    Поля:
      similarity       — косинус к размеченному трудному ведру, 0..1;
      cheap_confidence — уверенность дешёвой модели после первого прогона, 0..1;
      cheap_correct    — правда ли дешёвая модель ответила верно. Это разметка
                         для оценки, роутеру она в момент решения недоступна.

    Ловушка: cheap_confidence и cheap_correct — разные вещи. Модель бывает
    уверенно неправа, и именно из-за этого каскад с высоким порогом теряет
    качество, а не только деньги.

    RoutingError на неположительные prompt_tokens, отрицательные
    output_tokens и на similarity/cheap_confidence вне [0, 1].
    """
    if prompt_tokens <= 0:
        raise RoutingError(f"{req_id}: prompt_tokens must be positive, got {prompt_tokens}")
    if output_tokens < 0:
        raise RoutingError(f"{req_id}: output_tokens must be non-negative, got {output_tokens}")
    if not 0.0 <= similarity <= 1.0:
        raise RoutingError(f"{req_id}: similarity out of range: {similarity}")
    if not 0.0 <= cheap_confidence <= 1.0:
        raise RoutingError(f"{req_id}: cheap_confidence out of range: {cheap_confidence}")
    return {
        "req_id": req_id,
        "task": task,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "similarity": float(similarity),
        "cheap_confidence": float(cheap_confidence),
        "cheap_correct": bool(cheap_correct),
    }


def call_cost(model, prompt_tokens, output_tokens):
    """Цена одного вызова модели по рейт-карте MODELS, в долларах.

    call_cost("cheap", 1000, 200)     ->  0.0008
    call_cost("frontier", 1000, 200)  ->  0.006

    Разбор первого: 1000/1e6 * 0.40 = 0.0004 за вход, 200/1e6 * 2.00 = 0.0004
    за выход.

    Заметь: у обеих моделей выход ровно в 5 раз дороже входа, поэтому
    отношение цен cheap/frontier равно 2/15 на ЛЮБОЙ смеси токенов, а не
    только на этой. Это нам ещё пригодится в cascade_break_even_rate.

    RoutingError на неизвестную модель и на отрицательные токены.
    """
    if model not in MODELS:
        raise RoutingError(f"unknown model {model!r}")
    if prompt_tokens < 0 or output_tokens < 0:
        raise RoutingError(f"negative token count: {prompt_tokens}, {output_tokens}")
    rate = MODELS[model]
    return prompt_tokens / 1e6 * rate["input"] + output_tokens / 1e6 * rate["output"]


def pre_route(request):
    """Классификатор ПЕРЕД вызовом: 'cheap' или 'frontier'.

    Три сигнала, доступные до первого прогона:
      1. класс задачи из HARD_TASKS;
      2. промпт длиннее LONG_PROMPT_TOKENS;
      3. similarity >= HARD_SIMILARITY к трудному ведру.

    pre_route(make_request("a", "chat", 300, 80))                    ->  'cheap'
    pre_route(make_request("b", "math", 300, 80))                    ->  'frontier'
    pre_route(make_request("c", "chat", 9000, 80))                   ->  'frontier'
    pre_route(make_request("d", "chat", 300, 80, similarity=0.9))    ->  'frontier'

    Ловушка: сигналы НЕ голосуют большинством, они складываются дизъюнкцией.
    Один сигнал за frontier перевешивает два за cheap, потому что цена ошибки
    несимметрична: лишние полцента против испорченного ответа.

    Четвёртый сигнал урока — уверенность первого прогона — здесь недоступен
    принципиально: он появляется только ПОСЛЕ вызова. На нём построена
    cascade(), и это ровно та разница между pre-route и каскадом, из-за
    которой у каскада выше пол качества и выше задержка.
    """
    if request["task"] in HARD_TASKS:
        return "frontier"
    if request["prompt_tokens"] > LONG_PROMPT_TOKENS:
        return "frontier"
    if request["similarity"] >= HARD_SIMILARITY:
        return "frontier"
    return "cheap"


def cascade(request, threshold=CASCADE_THRESHOLD):
    """Каскад: сначала дешёвая модель, при низкой уверенности — повтор на frontier.

    Возвращает dict: models, escalated, cost_usd, correct.

    Уверенный дешёвый ответ:
        cascade(make_request("a", "chat", 1000, 200, cheap_confidence=0.9))
        ->  models ['cheap'], escalated False, cost_usd 0.0008
    Неуверенный:
        cascade(make_request("b", "chat", 1000, 200, cheap_confidence=0.4))
        ->  models ['cheap', 'frontier'], escalated True, cost_usd 0.0068

    Ловушка: эскалация НЕ отменяет счёт за первый прогон. Вход оплачивается
    дважды — 0.0008 + 0.006 = 0.0068, а не 0.006. Именно из-за этого каскад
    на трудном потоке дороже, чем просто frontier-only.

    Порог нестрогий сверху: уверенность ровно на пороге эскалацию НЕ вызывает.

    Опора качества: frontier принимается за правильный всегда. Это не
    утверждение о мире, это выбор базовой линии — иначе «просадку качества»
    не от чего отсчитывать.

    RoutingError на threshold вне [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise RoutingError(f"threshold out of range: {threshold}")
    p, o = request["prompt_tokens"], request["output_tokens"]
    cost = call_cost("cheap", p, o)
    if request["cheap_confidence"] < threshold:
        # второй заход платится сверх первого, а не вместо него
        return {
            "models": ["cheap", "frontier"],
            "escalated": True,
            "cost_usd": cost + call_cost("frontier", p, o),
            "correct": True,
        }
    return {
        "models": ["cheap"],
        "escalated": False,
        "cost_usd": cost,
        "correct": request["cheap_correct"],
    }


def cascade_break_even_rate(prompt_tokens, output_tokens):
    """Доля эскалаций, выше которой каскад дороже, чем сразу frontier.

    cascade_break_even_rate(4000, 300)  ->  примерно 0.8667
    cascade_break_even_rate(100, 5000)  ->  примерно 0.8667  (та же цифра!)

    Вывод: счёт каскада на долю эскалаций e равен cheap + e * frontier,
    потому что дешёвый прогон платится ВСЕГДА. Он дешевле frontier-only, пока
    cheap + e * frontier < frontier, то есть e < 1 - cheap/frontier.

    На нашей рейт-карте cheap/frontier = 2/15 при любой смеси токенов, отсюда
    порог 13/15 ≈ 0.867 и не зависит от длины промпта и ответа.

    Отсюда правило чтения дашборда: 30% эскалаций из MAX_ESCALATION_RATE —
    это НЕ порог по деньгам, по деньгам там ещё огромный запас. Это порог по
    качеству и по задержке: каждая эскалация — второй круг, то есть удвоенная
    латентность на этой доле трафика.

    RoutingError на нулевой запрос: у бесплатного запроса порога не бывает.
    """
    frontier = call_cost("frontier", prompt_tokens, output_tokens)
    if frontier <= 0:
        raise RoutingError("empty request has no break-even rate")
    return 1.0 - call_cost("cheap", prompt_tokens, output_tokens) / frontier


def route_request(request, strategy, threshold=CASCADE_THRESHOLD):
    """Обслужить запрос выбранной стратегией. Формат ответа общий для всех.

    Возвращает dict: models, escalated, cost_usd, correct.

    route_request(req, "frontier_only")["models"]  ->  ['frontier']
    route_request(req, "cheap_only")["models"]     ->  ['cheap']
    route_request(req, "pre_route")                ->  один вызов, escalated False
    route_request(req, "cascade")                  ->  как cascade()

    'pre_route' делает РОВНО один вызов — тем и отличается от каскада: нет
    второго круга, значит нет и удвоенной задержки, но и пола качества нет:
    ошибку классификатора исправлять нечем.

    RoutingError на стратегию не из STRATEGIES.
    """
    if strategy not in STRATEGIES:
        raise RoutingError(f"unknown strategy {strategy!r}")
    p, o = request["prompt_tokens"], request["output_tokens"]
    if strategy == "cascade":
        return cascade(request, threshold)
    if strategy == "frontier_only":
        model = "frontier"
    elif strategy == "cheap_only":
        model = "cheap"
    else:
        model = pre_route(request)
    return {
        "models": [model],
        "escalated": False,
        "cost_usd": call_cost(model, p, o),
        # frontier — базовая линия качества, дешёвая модель права по разметке
        "correct": True if model == "frontier" else request["cheap_correct"],
    }


def run_workload(requests, strategy, threshold=CASCADE_THRESHOLD):
    """Прогнать поток запросов стратегией и собрать отчёт роутера.

    Возвращает dict:
      requests           — сколько запросов,
      total_cost_usd     — счёт стратегии,
      baseline_cost_usd  — счёт frontier-only,
      saving_usd/pct     — экономия против базовой линии, может быть
                           ОТРИЦАТЕЛЬНОЙ,
      escalation_rate    — доля вторых заходов (0 у всех, кроме каскада),
      cheap_share        — доля запросов, обслуженных без frontier,
      accuracy           — доля верных ответов,
      quality_loss_pct   — просадка против frontier-only в п.п.

    run_workload([], "cascade")["saving_pct"]  ->  0.0

    Главное свойство: экономия каскада — не константа, а функция состава
    потока. На лёгком потоке она большая, на трудном становится
    отрицательной, потому что дешёвый прогон оплачен и выброшен. Обе стороны
    проверены тестами.

    Пустой вход не должен делить на ноль: нули и accuracy 1.0 (ничего не
    испортили).
    """
    total = baseline = 0.0
    escalated = correct = cheap_served = 0
    for request in requests:
        out = route_request(request, strategy, threshold)
        total += out["cost_usd"]
        baseline += call_cost("frontier", request["prompt_tokens"], request["output_tokens"])
        escalated += int(out["escalated"])
        correct += int(out["correct"])
        cheap_served += int("frontier" not in out["models"])
    n = len(requests)
    if n == 0:
        return {"requests": 0, "total_cost_usd": 0.0, "baseline_cost_usd": 0.0,
                "saving_usd": 0.0, "saving_pct": 0.0, "escalation_rate": 0.0,
                "cheap_share": 0.0, "accuracy": 1.0, "quality_loss_pct": 0.0}
    accuracy = correct / n
    return {
        "requests": n,
        "total_cost_usd": total,
        "baseline_cost_usd": baseline,
        "saving_usd": baseline - total,
        "saving_pct": (baseline - total) / baseline * 100 if baseline else 0.0,
        "escalation_rate": escalated / n,
        "cheap_share": cheap_served / n,
        "accuracy": accuracy,
        "quality_loss_pct": (1.0 - accuracy) * 100,
    }


def drift_alarm(report, max_escalation_rate=MAX_ESCALATION_RATE,
                max_quality_loss_pct=MAX_QUALITY_LOSS_PCT):
    """Онлайновый гейт качества: что в отчёте роутера уже нехорошо.

    Возвращает СПИСОК причин, отсортированный, пустой список = всё в норме.
    Возможные причины: 'escalation_rate', 'quality_loss'.

    drift_alarm(run_workload(easy, "cascade"))   ->  []
    drift_alarm(run_workload(hard, "cascade"))   ->  ['escalation_rate', 'quality_loss']

    Сравнение строгое: ровно на пороге тревоги нет, тревога начинается за ним.

    Зачем список, а не bool: «дрейф» — это не одно событие. Поехавшая доля
    эскалаций и просевшее качество приходят из разных причин и чинятся
    по-разному: первое — пересборкой маршрута, второе — порогом каскада.
    Отчёт, который схлопывает их в один флаг, не даёт понять, что чинить.
    """
    reasons = []
    if report["escalation_rate"] > max_escalation_rate:
        reasons.append("escalation_rate")
    if report["quality_loss_pct"] > max_quality_loss_pct:
        reasons.append("quality_loss")
    return sorted(reasons)
