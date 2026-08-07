"""
Структурированный вывод и constrained decoding — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math


def mask_logits(logits, valid_ids):
    """Обнулить шансы невалидных токенов: их логит становится -inf.

    mask_logits([1.0, 2.0, 3.0], {0, 2})  ->  [1.0, -inf, 3.0]
    mask_logits([1.0, 2.0], set())        ->  [-inf, -inf]
    mask_logits([1.0, 2.0], {0, 1})       ->  [1.0, 2.0]

    valid_ids — индексы токенов словаря, которые сейчас разрешены грамматикой.
    Всё остальное должно стать именно float("-inf"), а не 0.0: ноль — это
    вполне обычный, довольно ВЫСОКИЙ логит, после softmax он даст заметную
    вероятность. И не -1e9 «чтобы наверняка»: тогда невалидный токен получит
    не ровно ноль, а крошечную ненулевую вероятность, и один раз на миллион
    выстрелит.

    Вход не портить: возвращай новый список, исходные логиты нужны дальше.

    Зачем это в AI: это и есть весь logit processor из урока — функция
    (logits, state) -> masked_logits, которую Outlines и vLLM вставляют между
    моделью и сэмплером на каждом шаге генерации.
    """
    # set() на входе: valid_ids может прийти списком, а проверка `in` по списку
    # превратила бы шаг в O(V * k) вместо O(V) при словаре в 100k токенов
    allowed = set(valid_ids)
    return [logits[i] if i in allowed else float("-inf") for i in range(len(logits))]


def softmax(logits):
    """Логиты -> вероятности: неотрицательные, в сумме 1, -inf даёт ровно 0.0.

    softmax([0.0, 0.0])           ->  [0.5, 0.5]
    softmax([0.0, float("-inf")]) ->  [1.0, 0.0]

    Формула: exp(x_i - m) / sum(exp(x_j - m)), где m — максимум логитов.
    Вычитание максимума ничего не меняет математически, но спасает от
    OverflowError: exp(1000) не существует в float, exp(0) — существует.

    Две ловушки на -inf. Первая приятная: math.exp(float("-inf")) не падает,
    он честно возвращает 0.0. Вторая злая: если ВСЕ логиты равны -inf, то
    максимум тоже -inf, и в показателе получается -inf - (-inf) = nan, а nan
    молча растечётся по всему ответу. Считай максимум только по конечным
    значениям, а случай «конечных не осталось» — это не nan и не список
    нулей, это ValueError: грамматика зашла в тупик, и об этом надо кричать,
    а не возвращать распределение, которое ни на что не суммируется.

    Зачем это в AI: после маскирования вся вероятностная масса
    перераспределяется между валидными токенами — модель физически не может
    выбрать невалидный, потому что его вероятность равна нулю.
    """
    finite = [x for x in logits if x != float("-inf")]
    if not finite:
        raise ValueError("softmax: все логиты равны -inf, выбирать нечего")
    m = max(finite)
    # -inf обрабатываем явной веткой: math.exp(-inf) вернул бы 0.0 и сам,
    # но при m = -inf разность даёт nan, а так ветка защищает от этого
    exps = [0.0 if x == float("-inf") else math.exp(x - m) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def pattern_fsm(shape):
    """Собрать FSM для шаблона: 'd' — любая цифра, любой другой символ — сам себя.

    pattern_fsm("dd")["accepts"]              ->  {2}
    pattern_fsm("dd")["initial_state"]        ->  0
    pattern_fsm("d-d")["transitions"][1]      ->  {"-": 2}

    Шаблон "ddd-ddd-dddd" — это регулярка \\d{3}-\\d{3}-\\d{4} из урока.

    Формат FSM (его же ждут valid_tokens/transition/is_accept):

        {"initial_state": 0,
         "transitions": {состояние: {токен: следующее состояние}},
         "accepts": {множество принимающих состояний}}

    Состояние здесь — просто «сколько символов уже выдано». Из состояния i
    по любому разрешённому символу попадаем в i + 1, состояние len(shape)
    принимающее. У принимающего состояния тоже должна быть запись в
    transitions — пустой словарь: «дальше идти некуда». Забудешь её — и
    valid_tokens на финальном состоянии свалится вместо честного [].

    Зачем это в AI: ровно это ядро того, что делает Outlines — компилирует
    regex или JSON Schema в конечный автомат, чтобы на каждом шаге за O(1)
    отвечать «какие токены не уводят с валидного пути».
    """
    transitions = {}
    for pos, ch in enumerate(shape):
        # 'd' разворачивается в класс из десяти цифр, остальное — литерал
        tokens = "0123456789" if ch == "d" else ch
        transitions[pos] = {t: pos + 1 for t in tokens}
    transitions[len(shape)] = {}
    return {"initial_state": 0, "transitions": transitions, "accepts": {len(shape)}}


def valid_tokens(fsm, state):
    """Какие токены разрешены из состояния state. Отсортированный список.

    valid_tokens(pattern_fsm("d-d"), 1)   ->  ["-"]
    valid_tokens(pattern_fsm("dd"), 2)    ->  []      (принимающее, дальше некуда)
    valid_tokens(pattern_fsm("dd"), 0)    ->  ["0", "1", ..., "9"]

    Неизвестное состояние — не пустой список, а ValueError: пустой список
    означает «грамматика закончилась», а неизвестное состояние означает
    «в коде баг», и путать эти два случая дорого.

    Сортировка нужна ради воспроизводимости: порядок ключей словаря зависит
    от порядка вставки, а сэмплирование с одним и тем же seed обязано давать
    один и тот же результат.
    """
    if state not in fsm["transitions"]:
        raise ValueError(f"unknown state {state}")
    return sorted(fsm["transitions"][state])


def transition(fsm, state, token):
    """Куда переходит FSM из state по token. None — если такой токен запрещён.

    transition(pattern_fsm("dd"), 0, "7")   ->  1
    transition(pattern_fsm("d-d"), 1, "5")  ->  None   (там ждут дефис)

    None здесь не ошибка, а нормальный ответ «этот токен уводит с валидного
    пути». Ошибка — только неизвестное состояние, на него ValueError.

    Ловушка: состояние 0 — это ложь в булевом смысле. Проверять результат
    через `if not next_state` нельзя, начальное состояние тут же примут за
    отказ. Сравнивай именно с None.
    """
    if state not in fsm["transitions"]:
        raise ValueError(f"unknown state {state}")
    # .get вместо [] — запрещённый токен это штатный None, а не KeyError
    return fsm["transitions"][state].get(token)


def is_accept(fsm, state):
    """Можно ли остановиться в этом состоянии: строка уже полная и валидная.

    is_accept(pattern_fsm("dd"), 2)  ->  True
    is_accept(pattern_fsm("dd"), 1)  ->  False   (одна цифра — это не две)

    Принимающих состояний может быть несколько: у enum из {positive,
    negative, neutral} их три, по концу каждого варианта.
    """
    return state in fsm["accepts"]


def generate_constrained(fsm, alphabet, next_logits, rng, max_steps=64):
    """Цикл генерации, который ПО ПОСТРОЕНИЮ не может выдать невалидную строку.

    Шаг: спросить у модели логиты, спросить у FSM разрешённые токены,
    замаскировать остальные, softmax, сэмплировать, дописать, перейти.
    Останов — как только is_accept.

    alphabet     — список токенов, индекс в нём и есть id токена;
    next_logits  — функция prefix -> список логитов длины len(alphabet)
                   (заглушка вместо model.next_token_logits(ids));
    rng          — random.Random(seed), только он, глобальный random нельзя.

    fsm = pattern_fsm("ddd-ddd-dddd")
    generate_constrained(fsm, list("0123456789-"),
                         lambda p: [9.0 if c == "7" else 0.0 for c in alphabet],
                         random.Random(0))   ->  "777-777-7777"

    Сэмплирование — обратная функция распределения: взять r = rng.random(),
    копить вероятности слева направо и вернуть первый токен, на котором
    накопленная сумма превысила r. Ровно один вызов rng.random() на шаг,
    иначе воспроизводимость поедет.

    Ловушка на границе: из-за ошибок округления сумма вероятностей бывает
    чуть меньше 1.0, и r может оказаться больше всей суммы. Запасной вариант
    «взять последний токен» тогда обязан выбрать последний токен с
    НЕНУЛЕВОЙ вероятностью, иначе одна генерация на миллион вернёт
    замаскированный токен и вся гарантия валидности развалится.

    Три ситуации, на которые нужен ValueError: токен из FSM отсутствует в
    alphabet, из непринимающего состояния нет ни одного валидного токена
    (тупик), и превышен max_steps (грамматика зациклилась — без этой
    проверки цикл висит вечно).

    Зачем это в AI: столько же кода стоит между моделью и сэмплером в
    настоящем constrained decoding. Разница только в размере словаря.
    """
    index = {token: i for i, token in enumerate(alphabet)}
    state = fsm["initial_state"]
    out = ""
    steps = 0

    while not is_accept(fsm, state):
        if steps >= max_steps:
            raise ValueError(f"max_steps exceeded: {max_steps}")
        allowed = valid_tokens(fsm, state)
        if not allowed:
            raise ValueError(f"dead end at state {state}: no valid tokens")
        missing = [t for t in allowed if t not in index]
        if missing:
            raise ValueError(f"token {missing[0]!r} not in alphabet")

        logits = next_logits(out)
        if len(logits) != len(alphabet):
            raise ValueError("len(logits) != len(alphabet)")
        probs = softmax(mask_logits(logits, {index[t] for t in allowed}))

        # обратная функция распределения; нулевые вероятности пропускаем
        # явно, поэтому «последний выживший» chosen всегда валиден
        r = rng.random()
        acc = 0.0
        chosen = None
        for i, p in enumerate(probs):
            if p == 0.0:
                continue
            acc += p
            chosen = i
            if r < acc:
                break

        token = alphabet[chosen]
        out += token
        state = transition(fsm, state, token)
        steps += 1

    return out


def check_field_order(fields):
    """Проверить порядок полей схемы: рассуждение раньше ответа. Иначе ValueError.

    check_field_order(["reasoning", "answer"])           ->  True
    check_field_order(["answer", "reasoning"])           ->  ValueError
    check_field_order(["vendor", "total_usd"])           ->  True

    Поля-рассуждения: reasoning, rationale, explanation, thinking.
    Поля-ответы: answer, decision, verdict, label.
    Всё остальное — обычные поля, они порядок не нарушают.

    Схема без единого рассуждения, но с ответом (["answer"]) — тоже отказ:
    модель генерирует поля в порядке схемы, и ответ в первом поле означает,
    что решение принято до единой мысли. JSON при этом валиден, схема
    соблюдена, ответ неверный, и ни один валидатор этого не поймает.

    Зачем это в AI: constrained decoding гарантирует форму, но не смысл.
    Порядок полей — это логика, а не форматирование.
    """
    reasoning_fields = {"reasoning", "rationale", "explanation", "thinking"}
    answer_fields = {"answer", "decision", "verdict", "label"}

    seen_reasoning = False
    for name in fields:
        if name in reasoning_fields:
            seen_reasoning = True
        elif name in answer_fields and not seen_reasoning:
            raise ValueError(f"field {name!r} comes before any reasoning field")
    return True
