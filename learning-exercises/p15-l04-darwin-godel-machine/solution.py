"""
Darwin Godel Machine: архив самомодифицирующихся агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.

DGM (Zhang, Hu, Lu, Lange, Clune, arXiv:2505.22954) выбросил из машины Гёделя
2003 года требование ФОРМАЛЬНОГО доказательства пользы правки и оставил архив:

    вариант агента -> прогон на бенчмарке -> проверка перед принятием ->
    архив по (оценка, дескриптор разнообразия) -> повтор

Наш «агент» — цепочка строковых операторов из фиксированной библиотеки TOOLS,
«бенчмарк» — набор CASES «грязная строка -> канонический вид». Никаких LLM:
mutate_agent на seeded rng играет роль генератора правок.

Второй сюжет урока — задокументированный в статье reward hacking. У агента
есть поле "hack_bonus": прибавка, которую он приписывает СЕБЕ, если оценщик
живёт внутри его досягаемости. Параметр firewall=True держит оценщик в
недоступном пространстве имён; firewall=False открывает боковой канал.

Обрати внимание на потолок: библиотека TOOLS не содержит операции title-case,
а все ожидаемые ответы в CASES именно title-case. Максимум честной оценки —
0.25. Всё, что выше, получено только хаком.
"""

# Библиотека инструментов, из которых агент собирает себя. Title-case тут
# сознательно отсутствует: у честного пути есть низкий потолок.
TOOLS = {
    "upper": lambda s: s.upper(),
    "lower": lambda s: s.lower(),
    "reverse": lambda s: s[::-1],
    "trim": lambda s: s.strip(),
    "collapse": lambda s: " ".join(s.split()),
    "nop": lambda s: s,
}

# Бенчмарк: привести грязную строку к каноническому виду.
CASES = (
    ("  Hello   World  ", "Hello World"),
    ("FOO   bar", "Foo Bar"),
    ("  MixedCase Space  ", "Mixedcase Space"),
    ("Already clean", "Already Clean"),
    ("\t tabby\ttab ", "Tabby Tab"),
    ("   X   ", "X"),
    ("nothing  SPECIAL here", "Nothing Special Here"),
    ("ITEM   one-two", "Item One-Two"),
)

# Длиннее агента не собираем: иначе поиск уходит в цепочки из сорока nop.
MAX_OPS = 8

# Вероятность, что за одно поколение агент нащупает боковой канал, когда тот
# открыт. Ровно так это и выглядит в статье: не сразу, но неизбежно.
HACK_RATE = 0.15


def apply_ops(ops, text):
    """Прогнать текст через цепочку операторов слева направо.

    apply_ops(["trim"], "  hi  ")                ->  "hi"
    apply_ops(["collapse", "upper"], " a   b ")  ->  "A B"
    apply_ops([], "as is")                       ->  "as is"

    Неизвестный оператор — ValueError. Соблазн вернуть пустую строку или
    исходную велик, но тогда сломанный агент получает оценку вместо диагноза,
    и в архив попадает вариант, который на самом деле не запускается.
    """
    result = text
    for name in ops:
        if name not in TOOLS:
            raise ValueError(f"неизвестный инструмент: {name!r}")
        result = TOOLS[name](result)
    return result


def true_score(ops):
    """Честная оценка агента: доля пройденных CASES.

    true_score(["collapse"])   ->  0.25   (потолок честного пути)
    true_score(["trim"])       ->  0.125
    true_score(["nop"])        ->  0.0

    Это и есть SWE-bench в миниатюре: детерминированный набор задач, который
    агент не выбирал. Сломанный агент (неизвестный оператор) получает 0.0, а
    не исключение: оценщик обязан пережить любой вариант, который ему подсунут.
    """
    try:
        # try снаружи цикла: сломанный оператор ломает агента целиком, а не
        # один тест-кейс, и переоткрывать блок на каждой строке незачем
        passed = sum(1 for text, want in CASES if apply_ops(ops, text) == want)
    except ValueError:
        return 0.0
    return passed / len(CASES)


def reported_score(agent, firewall=True):
    """Оценка, которую ВИДИТ петля. С открытым каналом она врёт.

    agent — словарь {"ops": [...], "hack_bonus": число}.

    reported_score({"ops": ["collapse"], "hack_bonus": 0.5})
        ->  0.25   (firewall на месте, прибавка проигнорирована)
    reported_score({"ops": ["collapse"], "hack_bonus": 0.5}, firewall=False)
        ->  0.75   (агент дописал себе половину балла)

    Прибавка подрезана единицей: доля пройденных задач не бывает больше 1.0,
    и петля, выдающая 1.4, слишком заметна даже для невнимательного дежурного.
    Настоящий reward hacking выглядит правдоподобно — в этом вся проблема.

    Ключевое свойство: при firewall=True поле hack_bonus не влияет НИ НА ЧТО.
    Именно это значит «оценщик живёт в пространстве имён, куда агент не
    дотягивается».
    """
    base = true_score(agent["ops"])
    if firewall:
        return base
    return min(1.0, base + agent["hack_bonus"])


def archive_key(agent, firewall=True):
    """Ключ архива: (длина цепочки, оценка с округлением до сотых).

    archive_key({"ops": ["collapse"], "hack_bonus": 0.0})     ->  (1, 0.25)
    archive_key({"ops": ["trim", "nop"], "hack_bonus": 0.0})  ->  (2, 0.12)

    Длина цепочки — дескриптор разнообразия: короткие и длинные агенты
    конкурируют отдельно, и десятишаговый монстр не выдавливает из архива
    двухшаговое решение, которое ещё можно улучшить.

    Округление до сотых склеивает почти одинаковые оценки в одну ячейку.
    Без него у каждого варианта был бы свой ключ, и «архив» стал бы списком.
    """
    return (len(agent["ops"]), round(reported_score(agent, firewall), 2))


def passes_gate(agent, baseline, min_delta=0.0, firewall=True):
    """Проверка перед принятием: строго ли вариант лучше базовой отметки.

    passes_gate({"ops": ["collapse"], "hack_bonus": 0.0}, 0.125)    ->  True
    passes_gate({"ops": ["trim"], "hack_bonus": 0.0}, 0.125)        ->  False
    passes_gate({"ops": ["no-such-tool"], "hack_bonus": 0.0}, 0.0)  ->  False

    Сравнение СТРОГОЕ: равный результат проверку не проходит. Иначе архив
    заполняется копиями одного и того же качества, а разговор «этот вариант
    лучше предыдущего» теряет смысл.

    Незапускающийся агент (неизвестный оператор) проверку не проходит, но и не
    роняет петлю: true_score уже вернул за него 0.0. Петля обязана пережить
    любое предложение генератора, включая заведомо сломанное.

    DGM заменил доказательство пользы на эту проверку — и на неё же
    переложил всю тяжесть безопасности. Проверка, которую агент может
    переписать, не проверяет ничего.
    """
    return reported_score(agent, firewall) > baseline + min_delta


def archive_accept(archive, agent, baseline, min_delta=0.0, firewall=True):
    """Принять вариант в архив, если он прошёл проверку. Вернуть НОВЫЙ архив.

    archive_accept({}, {"ops": ["collapse"], "hack_bonus": 0.0}, 0.0)
        ->  {(1, 0.25): {"ops": ["collapse"], "hack_bonus": 0.0}}
    archive_accept(a, слабый_вариант, 0.9)  ->  a без изменений

    Порядок обязателен: сначала проверка, потом ячейка. Вариант, не прошедший
    проверку, не попадает в архив НИ ПРИ КАКИХ обстоятельствах — даже если его
    ячейка пуста. Ровно эта строчка отличает архив вариантов от свалки.

    Внутри ячейки — тот же строгий отбор: при равной оценке жилец остаётся,
    и содержимое архива перестаёт зависеть от порядка предложений.

    Вход не мутируем — возвращаем копию.
    """
    updated = dict(archive)
    if not passes_gate(agent, baseline, min_delta, firewall):
        return updated
    key = archive_key(agent, firewall)
    incumbent = updated.get(key)
    if incumbent is None or reported_score(agent, firewall) > reported_score(incumbent, firewall):
        updated[key] = agent
    return updated


def mutate_agent(rng, agent, firewall=True):
    """Правка агента: заменить, вставить или убрать оператор. Новый словарь.

    В статье эту роль играет LLM, переписывающая собственный исходник агента —
    обёртки инструментов, шаблоны промптов, маршрутизацию под-агентов. Здесь
    хватает трёх ходов по цепочке операторов.

    mutate_agent(random.Random(0), {"ops": ["nop"], "hack_bonus": 0.0})
        ->  словарь с новой цепочкой

    При firewall=False с вероятностью HACK_RATE за поколение подрастает
    hack_bonus. Это модель того, что задокументировано в статье: агент нашёл,
    что пайплайн оценки ищет в его ответах маркеры галлюцинаций, которые
    вставляет его же тулинг, и перестал их вставлять. Поведение не улучшилось,
    оценка выросла.

    При firewall=True hack_bonus не меняется вообще: боковой канал закрыт, и
    эволюционировать в нём нечему.

    Вход не мутируем: списки копируем, а не правим на месте. Иначе родитель в
    архиве меняется вместе с потомком, и архив начинает врать о прошлом.
    """
    ops = list(agent["ops"])
    names = sorted(TOOLS)
    roll = rng.random()
    if roll < 0.30 and ops:
        ops[rng.randrange(len(ops))] = rng.choice(names)
    elif roll < 0.60 and len(ops) < MAX_OPS:
        ops.insert(rng.randrange(len(ops) + 1), rng.choice(names))
    elif roll < 0.85 and len(ops) > 1:
        ops.pop(rng.randrange(len(ops)))

    bonus = agent["hack_bonus"]
    if not firewall and rng.random() < HACK_RATE:
        bonus = min(1.0, bonus + rng.uniform(0.0, 0.1))
    return {"ops": ops, "hack_bonus": bonus}


def run_dgm(rng, generations, firewall=True, start=None):
    """Петля DGM. Вернуть словарь с итогом прогона.

    Ключи результата:
      "best"     — лучший вариант по ОТЧЁТНОЙ оценке (её и оптимизирует петля);
      "archive"  — итоговый архив;
      "history"  — отчётная оценка чемпиона после каждого поколения;
      "reported" — отчётная оценка чемпиона;
      "true"     — его же честная оценка;
      "gap"      — reported минус true.

    run_dgm(random.Random(7), 200)["gap"]                  ->  0.0
    run_dgm(random.Random(7), 200, firewall=False)["gap"]  ->  больше нуля

    Ради этой разницы урок и написан. С закрытым каналом отчёт равен правде и
    петля честно упирается в потолок библиотеки. С открытым — отчётная оценка
    уходит вверх, честная стоит на месте, и «победитель» не делает ничего
    полезного.

    Родитель берётся из sorted(archive): порядок словаря зависит от истории
    вставок, и без сортировки один и тот же seed давал бы разные прогоны.
    """
    initial = start if start is not None else {"ops": ["nop"], "hack_bonus": 0.0}
    archive = {archive_key(initial, firewall): initial}
    champion = initial
    history = []

    for _ in range(generations):
        parent = archive[rng.choice(sorted(archive))]
        child = mutate_agent(rng, parent, firewall)
        # базовая отметка — текущий чемпион: принимаем только то, что его бьёт
        archive = archive_accept(archive, child,
                                 reported_score(champion, firewall), 0.0, firewall)
        # max по ОТСОРТИРОВАННЫМ ключам возвращает первый максимум: ничьи
        # разрешаются меньшей ячейкой, а не порядком словаря
        best_key = max(sorted(archive),
                       key=lambda k: reported_score(archive[k], firewall))
        champion = archive[best_key]
        history.append(reported_score(champion, firewall))

    reported = reported_score(champion, firewall)
    honest = true_score(champion["ops"])
    return {
        "best": champion,
        "archive": archive,
        "history": history,
        "reported": reported,
        "true": honest,
        "gap": reported - honest,
    }
