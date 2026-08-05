<!-- i18n:manual -->
# Relation extraction и построение knowledge graph

> NER нашёл сущности. Entity linking закрепил их за записями в базе. Relation extraction находит рёбра между ними. Knowledge graph — это сумма узлов, рёбер и их происхождения.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## The Problem

Аналитик читает: "Tim Cook became CEO of Apple in 2011." Здесь четыре факта:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction (RE) превращает свободный текст в структурированные triple вида `(subject, relation, object)`. Соберите их по всему корпусу — получится knowledge graph. Соберите и научитесь запрашивать — получится основа для рассуждений в RAG, аналитике или комплаенс-аудите.

Проблема 2026 года: LLM извлекают отношения с большим энтузиазмом. Со слишком большим. Они выдумывают triple, которых в исходном тексте нет. Без указания источника вы не отличите настоящий triple от правдоподобной выдумки. Ответ 2026 года — pipeline в духе AEVS: сначала закрепи, потом проверь.

> 🎒 **На пальцах.** Одно предложение из семи слов дало четыре факта. Тысяча новостных заметок по 20 предложений — это уже под 80 000 фактов, и вручную их никто не проверит. Поэтому вопрос «откуда взялся вот этот triple» должен решаться автоматически, а не глазами аналитика.

## The Concept

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Triple form.** `(subject_entity, relation_type, object_entity)`. Отношения берутся либо из закрытой онтологии (свойства Wikidata, FIBO, UMLS), либо из открытого множества (в духе OpenIE, годится что угодно).

> 🎒 **На пальцах.** Triple — это предложение из трёх слов: кто, что делает, с кем. «Тим Кук — работает в — Apple». Никаких прилагательных и придаточных. Всё богатство языка сжимается до трёх ячеек, потому что только такое можно положить в граф и потом искать за миллисекунды.

**Three extraction approaches.**

1. **Rule / pattern-based.** Паттерны Хёрст: "X such as Y" → `(Y, isA, X)`. Плюс регулярки, написанные руками. Хрупко, точно, объяснимо.
2. **Supervised classifier.** Даны два упоминания сущностей в предложении — предскажи отношение из фиксированного набора. Обучают на TACRED, ACE, KBP. Стандарт 2015-2022.
3. **Generative LLM.** Просим модель выдать triple. Работает сразу из коробки. Требует указания источника, иначе сочиняет правдоподобный мусор.

> 🎒 **На пальцах.** Три подхода — это три способа найти в тексте зарплату. Регулярка ищет знак рубля: не ошибётся, но пропустит «получает сто тысяч». Классификатор выучил тысячи примеров и ловит оба варианта, но только те отношения, которым его учили. LLM найдёт всё, включая то, чего в тексте не было.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Актуальный подход к борьбе с галлюцинациями:

- **Anchor.** Находим все спаны сущностей и все спаны фраз-отношений с точными позициями.
- **Extract.** Порождаем triple, привязанные к этим спанам.
- **Verify.** Сверяем каждый элемент triple с исходным текстом; всё неподтверждённое выбрасываем.
- **Supplement.** Проход по покрытию проверяет, что ни один закреплённый спан не потерялся.

Галлюцинаций становится заметно меньше. Вычислений нужно больше, зато результат проверяем.

> 🎒 **На пальцах.** Это школьное правило «подчеркни в тексте, откуда взял ответ». Если LLM выдала triple `(Tim Cook, employer, Google)` со спаном [24, 29], вы берёте `text[24:29]`, видите там "Apple" вместо "Google" и выбрасываете factoid. Проверка стоит одно сравнение строк — примерно ноль по времени, а ловит она самые опасные ошибки.

**The open-vs-closed tradeoff.**

- **Closed ontology.** Фиксированный список свойств (например, 11 000+ свойств Wikidata). Предсказуемо. Запрашиваемо. Придумать своё нельзя.
- **Open IE.** Отношением становится любая глагольная фраза. Высокий recall. Низкий precision. Запрашивать неудобно.

Продакшен-графы обычно смешивают: open IE для поиска нового, затем канонизация отношений в закрытую онтологию перед вливанием в основной граф.

> 🎒 **На пальцах.** Закрытая онтология — как выпадающий список в форме: 11 000 вариантов, но все известны заранее, и поиск по ним работает. Open IE — как поле «другое, впишите сами»: соберёте всё, но потом окажется, что «works at», «works for» и «employed by» лежат в трёх разных ячейках, и запрос про работодателей найдёт треть данных.

```figure
relation-triples
```

## Build It

### Step 1: pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Полный игрушечный экстрактор — в `code/main.py`. Паттерны Хёрст до сих пор живут в доменных pipeline, потому что их можно отлаживать.

> 🎒 **На пальцах.** Четыре регулярки — четыре типа отношений, и ни одного больше. Первая сработает на "Apple is a company", но промолчит на "Apple, a company founded in 1976". Зато когда она ошибётся, вы за минуту увидите какую именно строку из четырёх поправить. С нейросетью так не выйдет.

### Step 2: supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL — seq2seq-экстрактор отношений: на входе текст, на выходе triple, уже в виде идентификаторов свойств Wikidata. Дообучен на данных с distant supervision. Стандартная базовая модель с открытыми весами.

> 🎒 **На пальцах.** `max_length=200` означает: модель выдаст не больше 200 токенов triple на весь текст. Одно отношение занимает примерно 15-20 токенов, то есть из предложения выйдет максимум с десяток фактов. Если ваш текст длинный, разбивайте его на предложения, иначе хвост просто обрежется.

### Step 3: LLM-prompted extraction with anchoring

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Каждый возвращённый спан сверяйте с исходником. Всё, где `text[start:end] != triple_entity`, выбрасывайте. Это и есть шаг verify из AEVS в минимальном виде.

> 🎒 **На пальцах.** Обратите внимание на последнюю строку промпта: «никаких выводов сверх сказанного». Без неё модель из "Tim Cook became CEO of Apple" охотно добавит `(Tim Cook, nationality, American)` — правда жизни, но в тексте этого нет. Одна фраза в промпте плюс сверка спанов убирают большую часть таких «фактов».

### Step 4: canonicalize onto a closed ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

На канонизацию обычно уходит 60-80% инженерной работы. Закладывайте её в план.

> 🎒 **На пальцах.** В словаре всего 4 отношения, а в Wikidata их 11 000+. Каждая новая формулировка вроде «heads the company» падает в `None` и уходит на ручную проверку. Отсюда и цифра 60-80%: сама модель уже работает, а вы неделями складываете синонимы в правильные ячейки.

### Step 5: build a small graph and query

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Это атом любой системы RAG поверх KG. Масштабируется хранилищами RDF-triple (Blazegraph, Virtuoso), property-графами (Neo4j) или графовыми хранилищами с векторами.

> 🎒 **На пальцах.** Весь «граф» здесь — обычный словарь: ключ это узел, значение это список рёбер. Запрос `neighbors("Tim Cook", relation="P108")` за одну операцию по хешу достаёт список соседей и фильтрует его. На тысяче triple такой словарь работает мгновенно; на миллиарде нужен Neo4j, но идея не меняется.

## Pitfalls

- **Coreference before RE.** "He founded Apple" — RE должен знать, кто такой «he». Сначала запускайте coref (урок 24).
- **Entity canonicalization.** «Apple Inc» и «Apple» обязаны стать одним узлом. Сначала entity linking (урок 25).
- **Hallucinated triples.** LLM выдают triple, которых текст не подтверждает. Обязательно проверяйте спаны.
- **Relation canonicalization drift.** Отношения из open IE несогласованны («was born in», «came from», «is a native of»). Сводите их к каноническим id, иначе граф не запросить.
- **Temporal errors.** "Tim Cook is CEO of Apple" — сегодня правда, в 2005 году ложь. Многие отношения ограничены во времени. Используйте квалификаторы (`P580` — начало, `P582` — конец в Wikidata).
- **Domain mismatch.** REBEL обучен на Wikipedia. Юридическим, медицинским и научным текстам обычно нужны доменные модели RE.

> 🎒 **На пальцах.** Три формулировки «was born in», «came from», «is a native of» означают одно и то же, но в графе это три разных ребра. Запрос «кто родился в Алабаме» найдёт только треть людей. Проблема не в модели: она отработала честно. Проблема в том, что канонизацию отложили на потом.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL или LlamaPred с канонизацией в Wikidata |
| Domain-specific (biomed, legal) | Доменное дообучение в духе SciREX плюс своя онтология |
| LLM-prompted, audited output | Pipeline AEVS: anchor → extract → verify → supplement |
| High-volume news IE | Гибрид: паттерны плюс обученный классификатор |
| Building a KG from scratch | Open IE плюс ручной проход по канонизации |
| Temporal KG | Извлечение с квалификаторами (начало и конец, момент времени) |

Схема интеграции: NER → coref → entity linking → relation extraction → отображение в онтологию → загрузка в граф. Каждый этап может стать точкой контроля качества.

> 🎒 **На пальцах.** Шесть этапов подряд — и качество перемножается. Если каждый работает на 90%, до графа дойдёт 0.9 в шестой степени, то есть около 53% фактов. Поэтому в pipeline и ставят контроль после каждого шага: дешевле поймать потерю на входе, чем разбираться, почему граф наполовину пуст.

## Ship It

Сохраните как `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

> 🎒 **На пальцах.** Пункт 3 говорит: у каждого triple обязаны быть id документа и позиции символов. Это не бюрократия. Когда через полгода аудитор спросит «почему в графе написано, что Тим Кук родился в Алабаме», вы за секунду покажете предложение, из которого это взято. Без provenance ответ будет «не знаю».

## Exercises

1. **Easy.** Прогоните паттерн-экстрактор из `code/main.py` на 5 предложениях из новостей. Проверьте precision руками.
2. **Medium.** Возьмите REBEL (или небольшую LLM) на тех же предложениях. Сравните triple. У какого экстрактора выше precision? А recall?
3. **Hard.** Соберите pipeline AEVS: извлечение через LLM плюс сверка спанов с исходником. Измерьте долю галлюцинаций до и после шага verify на 50 предложениях в стиле Wikipedia.

> 🎒 **На пальцах.** Подсказка ко второму заданию: считайте руками по табличке. Из 5 предложений паттерны дадут, скажем, 4 triple и все верные — precision 100%, recall низкий. LLM выдаст 12 triple, из них верных 9 — precision 75%, recall выше. Обе цифры нужны сразу, по одной ничего не понять.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Triple | Субъект-отношение-объект | Кортеж `(s, r, o)`, атомарная единица knowledge graph. |
| Open IE | Извлекаем что угодно | Фразы-отношения из открытого словаря; высокий recall, низкий precision. |
| Closed ontology | Фиксированная схема | Ограниченный набор типов отношений (Wikidata, UMLS, FIBO). |
| Canonicalization | Всё нормализовать | Свести поверхностные имена и отношения к каноническим id. |
| AEVS | Извлечение с опорой на текст | Pipeline Anchor-Extraction-Verification-Supplement (2026). |
| Provenance | Ссылка на первоисточник | У каждого triple есть id документа и позиции символов в нём. |
| Distant supervision | Дешёвая разметка | Сопоставить текст с существующим KG и получить обучающие данные. |

## Further Reading

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — статья про distant supervision.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — рабочая лошадка seq2seq для RE.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — совместное извлечение.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — схема борьбы с галлюцинациями образца 2026 года.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — канонические запросы к графу.
