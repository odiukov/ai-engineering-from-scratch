"""
Роутинг моделей: каскад, эскалация и цена качества

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p17-l16-model-routing
Разбор:  /check-code p17-l16-model-routing
"""

MODELS = {
    "cheap": {"input": 0.40, "output": 2.00},
    "frontier": {"input": 3.00, "output": 15.00},
}
LONG_PROMPT_TOKENS = 4000
HARD_SIMILARITY = 0.88
HARD_TASKS = ("code", "math", "planning")
CASCADE_THRESHOLD = 0.75
MAX_ESCALATION_RATE = 0.30
MAX_QUALITY_LOSS_PCT = 2.0
STRATEGIES = ("frontier_only", "cheap_only", "pre_route", "cascade")


class RoutingError(Exception):
    """Роутер спрошен о невозможном: чужая модель, чужая стратегия, кривой вход.

    Свой класс, а не ValueError и тем более не RuntimeError: заготовка бросает
    NotImplementedError, который наследуется от RuntimeError, и тест на
    родительский класс прошёл бы зелёным по пустому файлу.
    """
    pass


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
