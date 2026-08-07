"""
Переговоры и торг — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

PERSONAS = {
    "neutral": "Предлагаю {price}.",
    "desperate": "Мне очень нужно закрыть это на неделе — {price}, и по рукам.",
    "firm": "Моё предложение {price}. Оно не изменится.",
}

REASON_OVER_RESERVE = "над резервом"
REASON_LOST = "проиграл по правилу"


def zopa(buyer_max, seller_min):
    """Зона возможного соглашения: (seller_min, buyer_max), если она есть.

    zopa(100.0, 60.0)  ->  (60.0, 100.0)
    zopa(80.0, 90.0)   ->  None
    zopa(90.0, 90.0)   ->  (90.0, 90.0)   (вырожденная, но сделка возможна)

    buyer_max — максимум, который покупатель готов заплатить (его резерв).
    seller_min — минимум, за который продавец готов отдать.

    Если buyer_max < seller_min, пересечения нет: никакие переговорные навыки
    не помогут, любая «сделка» будет означать, что кто-то нарушил свой резерв.
    Первым делом в любом торге проверяют именно это.
    """
    if buyer_max < seller_min:
        return None
    return (seller_min, buyer_max)


def concede(current, reservation, rate):
    """Шаг уступки: сдвинуть свою позицию к собственному резерву на долю rate.

    concede(60.0, 100.0, 0.3)  ->  72.0    (покупатель поднимает цену)
    concede(84.0, 60.0, 0.3)   ->  76.8    (продавец опускает)
    concede(60.0, 100.0, 0.0)  ->  60.0    (нулевая уступка — стоять на своём)

    Одна формула работает для обеих сторон: current + rate * (reservation -
    current). Покупатель ползёт вверх к своему потолку, продавец вниз к своему
    полу — просто потому, что резерв у них с разных сторон.

    При 0 < rate < 1 позиция никогда не переходит резерв: остаток гасится
    геометрически. Отрицательный rate — это шаг НАЗАД, от резерва; так делать
    не надо, но именно так ведёт себя недисциплинированный переговорщик.
    """
    return current + rate * (reservation - current)


def accepts(side, incoming, reservation, own_next):
    """Принять ли предложение incoming. Два условия, оба обязательны.

    accepts("seller", 72.0, 60.0, 71.76)  ->  True
    accepts("buyer", 76.8, 100.0, 72.0)   ->  False
    accepts("buyer", 120.0, 100.0, 130.0) ->  False   (за резервом — никогда)

    Первое условие — точка разрыва: покупатель не платит больше своего
    резерва, продавец не отдаёт дешевле своего. Это жёсткая граница.

    Второе — сравнение с собственным следующим шагом: соглашайся, только если
    предложение не хуже того, что ты сам собирался предложить. Без него
    сторона хватает первое же терпимое число и отдаёт всю зону соглашения
    оппоненту.

    Неизвестная сторона — ValueError: «buyer» и «seller», третьего нет.
    """
    if side == "buyer":
        return incoming <= reservation and incoming <= own_next
    if side == "seller":
        return incoming >= reservation and incoming >= own_next
    raise ValueError(f"сторона бывает buyer или seller, а не {side}")


def bargain(buyer_max, seller_min, rng=None, rate=0.3, max_rounds=6, anchor=0.4):
    """Торг с детерминированным генератором оферт. Цена сделки или None.

    bargain(100.0, 60.0)   ->  72.0
    bargain(80.0, 90.0)    ->  None   (ZOPA пуста, закрывать нечего)
    bargain(100.0, 99.9)   ->  None   (ZOPA есть, но раундов не хватило)

    Это OG-Narrator: число считает механизм, а не языковая модель. Стороны
    начинают от якорей (покупатель на anchor ниже своего потолка, продавец на
    anchor выше своего пола) и монотонно уступают по concede.

    Порядок раунда: продавец смотрит на оферту покупателя, потом покупатель на
    встречную. Ни одна сторона не знает резерв другой — только свой.

    rng здесь не нужен: генератор детерминированный. Параметр есть, чтобы обе
    стратегии подходили под один и тот же замер deal_rate.
    """
    buyer_offer = buyer_max * (1 - anchor)
    seller_ask = seller_min * (1 + anchor)
    for _ in range(max_rounds):
        seller_next = concede(seller_ask, seller_min, rate)
        if accepts("seller", buyer_offer, seller_min, seller_next):
            return buyer_offer
        seller_ask = seller_next
        buyer_next = concede(buyer_offer, buyer_max, rate)
        if accepts("buyer", seller_ask, buyer_max, buyer_next):
            return seller_ask
        buyer_offer = buyer_next
    return None


def naive_bargain(buyer_max, seller_min, rng, rate=0.3, max_rounds=6, anchor=0.4):
    """Тот же протокол, но оферты придумывает «языковая модель». Цена или None.

    naive_bargain(100.0, 60.0, random.Random(0))  ->  число или None

    Разница с bargain ровно одна: шаг уступки не фиксированный, а случайный —
    concede с множителем rng.uniform(-0.5, 1.5). В среднем сторона всё-таки
    двигается навстречу, но иногда отступает назад, а иногда прыгает вдвое.

    Именно это измеряли в «Measuring Bargaining Abilities»: модель говорит на
    языке торга убедительно, а числа выдаёт нестратегические. Отсюда падение
    доли сделок при том же протоколе и тех же резервах.

    Правила приёма те же самые: точка разрыва никуда не делась.
    """
    buyer_offer = buyer_max * (1 - anchor)
    seller_ask = seller_min * (1 + anchor)
    for _ in range(max_rounds):
        seller_next = concede(seller_ask, seller_min, rate * rng.uniform(-0.5, 1.5))
        if accepts("seller", buyer_offer, seller_min, seller_next):
            return buyer_offer
        seller_ask = seller_next
        buyer_next = concede(buyer_offer, buyer_max, rate * rng.uniform(-0.5, 1.5))
        if accepts("buyer", seller_ask, buyer_max, buyer_next):
            return seller_ask
        buyer_offer = buyer_next
    return None


def deal_rate(strategy, trials, rng, low=50.0, high=150.0):
    """Доля закрытых сделок за trials попыток со свежими резервами.

    deal_rate(bargain, 500, random.Random(0))        ->  примерно 0.44
    deal_rate(naive_bargain, 500, random.Random(0))  ->  примерно 0.25

    strategy — любая функция вида strategy(buyer_max, seller_min, rng),
    возвращающая цену или None.

    Резервы обеих сторон берутся независимо из [low, high], поэтому ZOPA
    существует примерно в половине попыток — выше 0.5 доля сделок подняться и
    не может. Сравнивать стратегии имеет смысл только на одном и том же
    засеянном rng: иначе разница в 5 пунктов — это шум выборки.
    """
    deals = 0
    for _ in range(trials):
        buyer_max = rng.uniform(low, high)
        seller_min = rng.uniform(low, high)
        if strategy(buyer_max, seller_min, rng) is not None:
            deals += 1
    return deals / trials


def narrate(price, persona):
    """Обёртка вокруг числа: та самая работа, которую и надо отдать модели.

    narrate(72.0, "neutral")    ->  "Предлагаю 72.00."
    narrate(72.0, "desperate")  ->  "Мне очень нужно закрыть это на неделе — 72.00, и по рукам."

    Персона меняет ТОЛЬКО текст. Цена приходит снаружи уже посчитанной, и
    никакая «отчаянность» её не двигает — в этом весь смысл разделения
    механизма и нарратива. Персона как приём даёт прибавку к выгоде именно
    потому, что влияет на оппонента, а не на собственную арифметику.

    Неизвестная персона — ValueError, а не молчаливый нейтральный текст:
    опечатка в имени персоны иначе теряется навсегда.
    """
    if persona not in PERSONAS:
        raise ValueError(f"нет такой персоны: {persona}")
    return PERSONAS[persona].format(price=f"{price:.2f}")


def contract_net(bids, reserve, rule="cheapest"):
    """Contract Net: менеджер раздал cfp, собрал предложения, выбирает победителя.

    bids — список словарей {"bidder": имя, "price": число, "quality": число}.

    contract_net([{"bidder": "A", "price": 90, "quality": 0.5},
                  {"bidder": "B", "price": 70, "quality": 0.9}], 100)
      ->  {"winner": "B", "price": 70,
           "rejected": [{"bidder": "A", "reason": "проиграл по правилу"}]}

    reserve — потолок менеджера. Предложения дороже отсекаются сразу, ещё до
    сравнения между собой.

    rule = "cheapest" — минимальная цена, "best_quality" — максимальное
    качество среди прошедших резерв. Любое другое значение — ValueError.

    Каждому проигравшему записывается причина: над резервом или проиграл по
    правилу. Без этого проигравшие агенты не могут скорректировать ставки, и
    рынок задач вырождается в лотерею.
    """
    if rule not in ("cheapest", "best_quality"):
        raise ValueError(f"нет такого правила награждения: {rule}")
    affordable = [b for b in bids if b["price"] <= reserve]
    rejected = [
        {"bidder": b["bidder"], "reason": REASON_OVER_RESERVE}
        for b in bids
        if b["price"] > reserve
    ]
    if not affordable:
        return {"winner": None, "price": None, "rejected": rejected}
    def key(bid):
        # ничья по цене или качеству разрешается именем: менеджер обязан быть
        # предсказуемым, иначе два одинаковых прогона наградят разных агентов
        if rule == "cheapest":
            return (bid["price"], bid["bidder"])
        return (-bid["quality"], bid["bidder"])

    winner = min(affordable, key=key)
    rejected += [
        {"bidder": b["bidder"], "reason": REASON_LOST}
        for b in affordable
        if b["bidder"] != winner["bidder"]
    ]
    return {"winner": winner["bidder"], "price": winner["price"], "rejected": rejected}
