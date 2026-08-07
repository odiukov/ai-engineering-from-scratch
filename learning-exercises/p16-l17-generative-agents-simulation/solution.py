"""
Генеративные агенты и эмерджентная симуляция — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

KINDS = ("observation", "reflection", "plan")

STOPWORDS = {"в", "и", "на", "о", "с", "что", "это", "к", "у", "не"}

# Три слагаемых оценки воспоминания из Park 2023: свежесть, важность,
# относимость. Веса подобраны так, чтобы свежесть НЕ забивала остальные два:
# иначе агент помнит только последнюю секунду и ведёт себя как рефлекс.
WEIGHTS = {"recency": 0.4, "importance": 0.35, "relevance": 0.25}

HALF_LIFE = 2.0

INVITATION = "вечеринка в кафе в пять"
PARTY_QUERY = "вечеринка кафе"


def keywords(text):
    """Значимые слова текста: нижний регистр, без стоп-слов, множеством.

    keywords("Вечеринка в кафе в пять")  ->  {"вечеринка", "кафе", "пять"}
    keywords("в и на")                   ->  set()

    Множество, а не список: повтор слова не должен усиливать совпадение.
    Настоящий Smallville считает косинус эмбеддингов, но идея та же — свести
    текст к набору признаков, по которым его можно сравнивать с запросом.
    """
    return {w for w in text.lower().split() if w not in STOPWORDS}


def make_memory(text, kind, ts, importance):
    """Одна запись потока памяти.

    make_memory("вечеринка в кафе в пять", "observation", 0, 9)
      ->  {"text": ..., "kind": "observation", "ts": 0, "importance": 9,
           "reflected": False}

    kind — один из KINDS: наблюдение, рефлексия или план.
    importance — самооценка агента от 1 до 10, проставляется один раз при
    записи и дальше только читается (в реальной системе это отдельный вызов
    модели, и кэшировать его — вопрос счёта за токены).

    Флаг reflected говорит, учтено ли воспоминание в какой-нибудь рефлексии.
    Ставится ровно один раз: рефлексия не должна пережёвывать одно и то же.

    Значения вне 1..10 и незнакомый kind — ValueError.
    """
    if kind not in KINDS:
        raise ValueError(f"неизвестный тип воспоминания: {kind}")
    if not 1 <= importance <= 10:
        raise ValueError("важность оценивается по шкале от 1 до 10")
    return {
        "text": text,
        "kind": kind,
        "ts": ts,
        "importance": importance,
        "reflected": False,
    }


def relevance(memory, query):
    """Относимость воспоминания к запросу: доля Жаккара по значимым словам.

    relevance(make_memory("вечеринка кафе", "observation", 0, 5),
              "вечеринка кафе")                             ->  1.0
    relevance(то же, "кот подоконник")                      ->  0.0

    Жаккар — это |пересечение| / |объединение|. Длинное воспоминание,
    случайно задевшее одно слово запроса, получит низкую оценку: знаменатель
    растёт вместе с текстом.

    Пустой текст или пустой запрос — 0.0, а не деление на ноль.
    """
    memory_words = keywords(memory["text"])
    query_words = keywords(query)
    if not memory_words or not query_words:
        return 0.0
    return len(memory_words & query_words) / len(memory_words | query_words)


def retrieval_score(memory, query, now, half_life=HALF_LIFE):
    """Итоговая оценка воспоминания: свежесть + важность + относимость.

    Свежесть — 2 ** (-возраст / half_life): за один half_life она падает
    ровно вдвое. Важность приводится к [0, 1] делением на 10, относимость уже
    в [0, 1]. Слагаемые взвешиваются по WEIGHTS.

    retrieval_score(свежий пустяк, "вечеринка кафе", now)  ->  примерно 0.44
    retrieval_score(старая важная запись, тот же запрос)   ->  примерно 0.61

    Второе больше первого — и это главное свойство схемы: важное и по делу
    всплывает раньше свежего пустяка. Если сделать вес свежести подавляющим,
    агент превратится в рефлекс без биографии.
    """
    age = now - memory["ts"]
    recency = 2.0 ** (-age / half_life)
    return (
        WEIGHTS["recency"] * recency
        + WEIGHTS["importance"] * memory["importance"] / 10.0
        + WEIGHTS["relevance"] * relevance(memory, query)
    )


def retrieve(stream, query, now, k=3, half_life=HALF_LIFE):
    """Top-k воспоминаний по итоговой оценке, от большего к меньшему.

    retrieve(stream, "вечеринка кафе", now=5, k=1)  ->  [самое уместное]
    retrieve([], "что угодно", 0)                   ->  []

    Извлечение РАНЖИРУЮЩЕЕ, а не фильтрующее: жёсткий порог выбросил бы
    контекст, который агенту всё равно нужен, просто он слабее остальных.
    Если в потоке всего одна запись — она и вернётся, какой бы слабой ни была.

    Ничьи разрешаются в пользу более поздней записи, потом по тексту: без
    этого один и тот же прогон даст разные ответы.
    """
    ranked = sorted(
        stream,
        key=lambda m: (-retrieval_score(m, query, now, half_life), -m["ts"], m["text"]),
    )
    return ranked[:k]


def reflect(stream, now, threshold=5, topics=2):
    """Свернуть накопленные воспоминания в одну запись повыше уровнем.

    Возвращает новую запись рефлексии или None, если накопилось мало.

    Триггер — сумма важности ЕЩЁ НЕ учтённых записей. Пока она ниже
    threshold, рефлексии не будет: синтезировать не из чего.

    reflect([наблюдение важности 9], now=1)   ->  "важно: вечеринка кафе"
    reflect([наблюдение важности 3], now=1)   ->  None

    Тема рефлексии — topics самых частых значимых слов, ничьи по алфавиту.
    Одного слова мало: свёрнутая до одного слова рефлексия теряет
    относимость к запросу и на извлечении проигрывает сырому наблюдению, из
    которого она же и сделана.

    Важность рефлексии на 2 выше максимальной из учтённых (потолок 10): вывод
    весит больше сырых наблюдений и потому дольше переживает затухание.
    Именно этим рефлексия и держит цель в голове агента.

    Учтённые записи помечаются reflected. Сама рефлексия тоже — иначе на
    следующем такте она попадёт себе же на вход, и агент начнёт бесконечно
    рефлексировать над словом «важно».
    """
    fresh = [m for m in stream if not m["reflected"]]
    if sum(m["importance"] for m in fresh) < threshold:
        return None
    counts = {}
    for m in fresh:
        for word in keywords(m["text"]):
            counts[word] = counts.get(word, 0) + 1
    top = sorted(counts, key=lambda w: (-counts[w], w))[:topics]
    importance = min(10, max(m["importance"] for m in fresh) + 2)
    for m in fresh:
        m["reflected"] = True
    entry = make_memory("важно: " + " ".join(top), "reflection", now, importance)
    entry["reflected"] = True
    stream.append(entry)
    return entry


def make_plan(stream, goal, now, k=3, threshold=0.5, half_life=HALF_LIFE):
    """План агента по цели goal, если память его поддерживает. Иначе None.

    make_plan([свежее приглашение], "вечеринка кафе", now=0)
      ->  "план: вечеринка кафе"
    make_plan([то же приглашение], "вечеринка кафе", now=20)
      ->  None

    Планирование сверху вниз начинается с вопроса «а что я вообще про это
    знаю»: извлекаем top-k по цели и смотрим на лучшую запись. Не дотянула до
    threshold — цели в голове агента больше нет, и план не строится.

    Именно так в ablation-эксперименте Park 2023 «рассыпается» поведение: без
    подкрепления память выцветает, и намерение исчезает само собой.
    """
    top = retrieve(stream, goal, now, k, half_life)
    if not top:
        return None
    if retrieval_score(top[0], goal, now, half_life) < threshold:
        return None
    return f"план: {goal}"


def simulate(n_agents, ticks, rng, use_reflection=True, threshold=0.5):
    """Мини-Смолвиль: одно зерно, никакого оркестратора, вечеринка сама собой.

    Возвращает список агентов вида {"name", "stream", "plan"}.

    simulate(5, 24, random.Random(0))                      -> все пятеро с планом
    simulate(5, 24, random.Random(0), use_reflection=False) -> почти никто

    Устройство такта:
      1. rng сводит ровно одну случайную пару агентов;
      2. если у встреченного есть план, он приглашает — в поток второго
         падает наблюдение важности 5 (слухи весят меньше, чем свой опыт);
      3. каждый агент при желании рефлексирует и заново строит план.

    Агент 0 стартует с наблюдением важности 9 — это и есть «Изабелла хочет
    устроить вечеринку». Больше никто ничего не знает.

    Выключи рефлексию — и цель рассеется: наблюдения затухают, план пропадает,
    приглашать становится некому. Это ablation из Park 2023 в миниатюре.
    """
    agents = [{"name": f"a{i}", "stream": [], "plan": None} for i in range(n_agents)]
    agents[0]["stream"].append(make_memory(INVITATION, "observation", 0, 9))
    agents[0]["plan"] = make_plan(agents[0]["stream"], PARTY_QUERY, 0, threshold=threshold)
    for t in range(1, ticks + 1):
        first, second = rng.sample(range(n_agents), 2)
        # приглашение идёт в обе стороны: любой из пары может оказаться знающим
        for src, dst in ((first, second), (second, first)):
            if agents[src]["plan"] is not None:
                agents[dst]["stream"].append(
                    make_memory(INVITATION, "observation", t, 5)
                )
        for agent in agents:
            if use_reflection:
                reflect(agent["stream"], t)
            agent["plan"] = make_plan(agent["stream"], PARTY_QUERY, t, threshold=threshold)
    return agents
