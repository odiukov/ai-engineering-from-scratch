<!-- i18n:manual -->
# Entity linking и disambiguation

> NER нашёл «Paris». Entity linking решает: Париж во Франции? Пэрис Хилтон? Пэрис в Техасе? Парис, троянский царевич? Без linking ваш knowledge graph остаётся неоднозначным.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## The Problem

В тексте написано: "Jordan beat the press." Ваш NER пометил «Jordan» как PERSON. Хорошо. Но *какой* Jordan?

- Michael Jordan (баскетбол)?
- Michael B. Jordan (актёр)?
- Michael I. Jordan (профессор ML из Беркли — да, в статьях по ML эта путаница реальна)?
- Jordan (страна Иордания)?
- Jordan (имя на иврите)?

Entity linking (EL) привязывает каждое упоминание к единственной записи в knowledge base: Wikidata, Wikipedia, DBpedia или ваша доменная KB. Две подзадачи:

1. **Candidate generation.** Дано «Jordan» — какие записи KB вообще правдоподобны?
2. **Disambiguation.** Дан контекст — какой из кандидатов верный?

Оба шага обучаемы. Оба измеряются на benchmark. Сама связка стабильна уже десять лет — меняется только качество disambiguation.

> 🎒 **На пальцах.** Представьте телефонную книгу, где пятеро записаны как «Иванов». NER говорит «это фамилия», а вам нужен конкретный человек. Здесь пять кандидатов на слово «Jordan»: угадать наугад — 1 шанс из 5, то есть 20%. Entity linking должен довести это до 90+%, читая контекст вокруг слова.

## The Concept

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation.** По поверхностной форме упоминания («Jordan») ищем кандидатов в alias-индексе. Словари алиасов из Wikipedia покрывают большинство именованных сущностей: "JFK" → John F. Kennedy, Jacqueline Kennedy, аэропорт JFK, фильм "JFK". Обычный индекс возвращает 10-30 кандидатов на упоминание.

> 🎒 **На пальцах.** Alias-индекс — это оглавление книги наоборот: не «глава → страница», а «слово → все записи, где оно может значиться». На «JFK» оглавление выдаёт 4 варианта, на «Jordan» — десятки. Ваша задача дальше — выбрать один из 10-30, а не из всех 100 миллионов записей Wikidata. Поиск сузился примерно в 5 миллионов раз.

**Disambiguation: three approaches.**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Работает хорошо, быстро, обучение не нужно.
2. **Embedding-based (ESS / REL / Blink).** Кодируем упоминание вместе с контекстом. Кодируем описание каждого кандидата. Берём максимум косинуса. Стандарт 2020-2024.
3. **Generative (GENRE, 2021; LLM-based, 2023+).** Декодируем каноническое имя сущности токен за токеном. Декодирование ограничено префиксным деревом допустимых имён, поэтому на выходе гарантированно валидный id из KB.

> 🎒 **На пальцах.** Три способа выбрать ресторан. Prior + context — «беру самый популярный поблизости»: дёшево и часто верно. Embedding — «сравниваю описание каждого с тем, чего хочу»: точнее, но надо всё описать заранее. Generative — «называю имя по буквам, а автодополнение не даёт написать несуществующее». Первый способ не требует ни одной секунды обучения, третий не может выдать битый id в принципе.

**End-to-end vs pipeline.** Современные модели (ELQ, BLINK, ExtEnD, GENRE) делают NER, candidate generation и disambiguation за один проход. В продакшене всё равно чаще живут pipeline-системы: в них можно менять компоненты по отдельности.

> 🎒 **На пальцах.** End-to-end — как моноблок: красиво, но сгорел блок питания — меняешь весь компьютер. Pipeline — системный блок из трёх частей: не устраивает disambiguation, поменяли только его, а NER и candidate generation остались как были. Именно поэтому продакшен консервативен.

### The two measurements

- **Mention recall (candidate gen).** Доля эталонных упоминаний, для которых правильная запись KB вообще попала в список кандидатов. Это потолок для всего pipeline.
- **Disambiguation accuracy / F1.** При правильных кандидатах — как часто верхний ответ верный.

Всегда сообщайте обе цифры. Система с 99% disambiguation при 80% candidate recall — это система на 80%.

> 🎒 **На пальцах.** Перемножьте сами: 0.99 × 0.80 = 0.792, то есть 79%. Если нужного кандидата не принесли на шаг disambiguation, никакая модель его уже не выберет — его физически нет в списке. Это как отличный судья на конкурсе, куда забыли пригласить победителя.

```figure
gx-entity-linking
```

## Build It

### Step 1: build an alias index from Wikipedia redirects

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Данные алиасов из Wikipedia: около 18 млн пар (alias, entity). Скачиваются из дампов Wikidata. Хранятся как инвертированный индекс.

> 🎒 **На пальцах.** 18 млн пар — это словарь, где у одного слова бывает сотня значений. В примере выше на «paris» три записи, на «apple» две. В реальном индексе на «paris» их сотни. Зато поиск по такому словарю — одна операция по хешу, микросекунды, независимо от размера.

### Step 2: context-based disambiguation

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Пересечение по Жаккару здесь игрушечное. Замените его на косинусную близость эмбеддингов (версия с трансформером — в `code/main.py`, шаг 2).

> 🎒 **На пальцах.** Жаккар считает, сколько общих слов у контекста и у описания сущности. Пусть в контексте 15 слов, в описании 10, общих 3. Тогда объединение 15 + 10 − 3 = 22, а score = 3 / 22 ≈ 0.14. У кандидата-баскетболиста общими окажутся «Chicago» и «Bulls», у страны — ничего. Побеждает тот, у кого дробь больше.

### Step 3: embedding-based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

На этапе индексации каждую сущность KB кодируем один раз. На запросе один раз кодируем упоминание с контекстом, считаем скалярные произведения с пулом кандидатов и берём максимум.

> 🎒 **На пальцах.** Модель `all-MiniLM-L6-v2` выдаёт вектор из 384 чисел. Сравнить упоминание с 30 кандидатами — это 30 скалярных произведений по 384 умножения, около 11 500 операций. Процессор делает это за доли миллисекунды. Тяжёлая часть — закодировать миллионы сущностей заранее, но её делают один раз.

### Step 4: generative entity linking (concept)

GENRE декодирует заголовок статьи Wikipedia символ за символом. Ограниченное декодирование (см. урок 20) гарантирует, что на выход попадёт только существующий заголовок. Всё это тесно связано с префиксным деревом, построенным по KB. Современные потомки этого подхода — REL-GEN и EL по промпту к LLM со структурированным выводом.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

В связке с белым списком (`choice` из Outlines) это самый простой EL-pipeline, который можно выкатить в 2026 году.

> 🎒 **На пальцах.** Обычная LLM может выдать заголовок "Michael Jordan (basketball player)", которого в Wikipedia нет, — и вы получите битую ссылку. Белый список из 3 разрешённых вариантов делает такой ответ невозможным: модель выбирает из 3 строк, а не сочиняет свою. Вероятность невалидного id падает с «несколько процентов» до нуля.

### Step 5: evaluate on AIDA-CoNLL

AIDA-CoNLL — стандартный benchmark для EL: 1393 статьи Reuters, 34 тысячи упоминаний, сущности из Wikipedia. Сообщайте точность по записям, которые есть в KB (`P@1`), и долю верно найденных NIL — тех, которых в KB нет.

> 🎒 **На пальцах.** 34 000 упоминаний на 1393 статьи — это примерно 24 упоминания на статью, то есть по одному на каждые пару предложений. Каждое из них ваша система должна привязать к правильной записи. Ошибка в 1% — это 340 неверных фактов в knowledge graph.

## Pitfalls

- **NIL handling.** Некоторых упоминаний в KB просто нет (новые сущности, малоизвестные люди). Система должна предсказать NIL, а не угадывать неправильную сущность. Это измеряется отдельно.
- **Mention boundary errors.** NER выше по конвейеру теряет часть спана («Bank of America» размечен как просто «Bank»). Recall у EL падает.
- **Popularity bias.** Обученные системы слишком часто выбирают частые сущности. Упоминание «Michael I. Jordan» в статье по ML регулярно уезжает к баскетболисту.
- **Cross-lingual EL.** Привязка упоминаний из китайского текста к англоязычным сущностям Wikipedia. Нужен многоязычный энкодер или шаг перевода.
- **KB staleness.** Новых компаний, событий и людей нет в прошлогоднем дампе Wikipedia. В продакшене нужен цикл обновления.

> 🎒 **На пальцах.** Popularity bias — это как поиск, который на запрос «Jordan» всегда показывает баскетболиста, даже если вы читаете научную статью. Баскетболист упоминается в Wikipedia на порядки чаще профессора, поэтому prior для него огромный. Лечится только тем, что контексту дают больший вес, чем частоте.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| General-purpose English + Wikipedia | BLINK или REL |
| Cross-lingual, KB = Wikipedia | mGENRE |
| LLM-friendly, few mentions/day | Промпт к Claude/GPT-4 со списком кандидатов и ограниченным JSON |
| Domain-specific KB (medical, legal) | Свой BERT с KB-aware ретривером и дообучением на доменном наборе в стиле AIDA |
| Extremely low-latency | Только точное совпадение по prior (базовый вариант Milne-Witten) |
| Research SOTA | GENRE / ExtEnD / генеративный LLM-EL |

Схема, которая работает в продакшене в 2026 году: NER → coref → EL по каждому упоминанию → схлопывание кластеров в одну каноническую сущность на кластер. Результат: один id из KB на сущность в документе, а не на каждое упоминание.

> 🎒 **На пальцах.** Шесть строк таблицы — шесть разных ответов на один вопрос «чем линковать». Разброс огромный: базовый Milne-Witten отвечает за микросекунды и не требует GPU, а GENRE генерирует ответ и требует видеокарту. Если у вас 50 упоминаний в день, берите промпт к LLM и не стройте инфраструктуру.

## Ship It

Сохраните как `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

> 🎒 **На пальцах.** Обратите внимание на слово «Refuse» в конце: скилл обязан отказываться от pipeline без базовой цифры mention recall. Причина простая — без неё вы не знаете, что оцениваете. Если candidate generation даёт 60%, то ваши «95% точности» на самом деле 0.95 × 0.60 = 57%.

## Exercises

1. **Easy.** Реализуйте disambiguation по схеме prior+context в `code/main.py` на 10 неоднозначных упоминаниях (Paris, Jordan, Apple). Разметьте правильные сущности руками. Измерьте точность.
2. **Medium.** Закодируйте 50 неоднозначных упоминаний sentence-трансформером. Постройте эмбеддинги описаний кандидатов. Сравните disambiguation на эмбеддингах с пересечением контекста по Жаккару.
3. **Hard.** Соберите доменную KB на 1000 сущностей (например, сотрудники и продукты вашей компании). Сделайте NER + EL целиком. Измерьте precision и recall на 100 отложенных предложениях.

> 🎒 **На пальцах.** Подсказка к первому заданию: 10 упоминаний — это шкала с шагом 10%. Восемь верных = 80%, и одна ошибка стоит целых 10 процентных пунктов. Для честного сравнения двух методов десяти примеров мало, поэтому во втором задании их уже 50. Начните с руками размеченного ответа: без него измерять нечего.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Entity linking (EL) | Ссылка на Wikipedia | Привязать упоминание к единственной записи в KB. |
| Candidate generation | Кто это может быть? | Вернуть короткий список правдоподобных записей KB для упоминания. |
| Disambiguation | Выбрать правильного | Оценить кандидатов по контексту и выбрать победителя. |
| Alias index | Табличка для поиска | Отображение поверхностной формы → сущности-кандидаты. |
| NIL | Нет в KB | Явное предсказание, что подходящей записи в KB не существует. |
| KB | Knowledge base | Wikidata, Wikipedia, DBpedia или ваша доменная KB. |
| AIDA-CoNLL | Тот самый benchmark | 1393 статьи Reuters с эталонными ссылками на сущности. |

## Further Reading

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — базовый подход prior+context.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — рабочая лошадка на эмбеддингах.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — генеративный EL с ограниченным декодированием.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — статья про benchmark.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — открытый продакшен-стек.
