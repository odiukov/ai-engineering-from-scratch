<!-- i18n:manual -->
# Разрешение coreference

> «She called him. He did not answer. The doctor was at lunch.» Три отсылки к двум людям, и ни одного имени. Coreference resolution разбирается, кто есть кто.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## The Problem

Достаньте из статьи на 300 слов все упоминания компании Apple Inc. Легко, когда в тексте написано «Apple». Трудно, когда написано «the company», «they», «Cupertino's technology giant» или «Jobs's firm». Если не свести эти mention к одной сущности, ваш NER-конвейер потеряет 60-80% упоминаний.

> 🎒 **На пальцах.** Потерять 60-80% — значит из 20 упоминаний Apple в статье увидеть 4-8. Представьте, что вы конспектируете лекцию и записываете только те предложения, где лектор произнёс полное название компании, а все «она», «эта фирма» и «они» выбрасываете. Конспект получится в пять раз беднее лекции.

Coreference resolution объединяет в один cluster все выражения, которые указывают на одну и ту же сущность реального мира. Это клей между поверхностным NLP (NER, разбор) и смысловыми задачами дальше по конвейеру (извлечение информации, QA, суммаризация, графы знаний).

Почему это важно в 2026-м:

- Суммаризация: «The CEO announced...» против «Tim Cook announced...» — в резюме гендиректора надо назвать по имени.
- Вопросно-ответные системы: «Who did she call?» требует разрешить «she».
- Извлечение информации: граф знаний, где «PER1 founded Apple» и «Jobs founded Apple» — две отдельные записи, просто неверен.
- Извлечение по нескольким документам: сведение mention из разных статей об одном событии — это cross-document coreference.

> 🎒 **На пальцах.** Пример с графом знаний самый наглядный. Без coreference у вас в базе две сущности-основателя вместо одной, и на вопрос «сколько компаний основал Джобс» система ответит вдвое больше правды. Одна нерешённая ссылка — и все цифры вниз по конвейеру поехали.

## The Concept

![Coreference clustering: mentions → entities](../assets/coref.svg)

**The task.** Вход: документ. Выход: разбиение mention (отрезков текста) на группы, где каждый cluster указывает на одну сущность.

> 🎒 **На пальцах.** Формально это задача кластеризации, только объекты — не точки, а куски текста. В предложении из эпиграфа три mention — «She», «him», «The doctor» — и два cluster: {She} и {him, The doctor}. Заметьте: правильный ответ требует догадаться, что доктор — мужчина, о чём в тексте прямо не сказано.

**Mention types.**

- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

> 🎒 **На пальцах.** Четыре типа идут по возрастанию сложности. Имя собственное найдёт даже поиск по строке. «The company» уже требует понять, о какой компании речь в этом абзаце. Местоимение «it» само по себе не несёт вообще никакой информации — весь смысл берётся из окружения. Именно на местоимениях модели и ошибаются чаще всего.

**Architectures.**

1. **Rule-based (Hobbs, 1978).** Разрешение местоимений по синтаксическому дереву с помощью грамматических правил. Хорошая база. Удивительно трудно обойти на местоимениях.
2. **Mention-pair classifier.** Для каждой пары mention (m_i, m_j) предсказываем, указывают ли они на одно и то же. Кластеризуем транзитивным замыканием. Стандарт до 2016 года.
3. **Mention-ranking.** Для каждого mention ранжируем кандидатов в antecedent (включая вариант «antecedent нет»). Берём верхнего.
4. **Span-based end-to-end (Lee et al., 2017).** Трансформерный энкодер. Перебираем все отрезки текста до ограничения по длине. Предсказываем оценку «это mention». Предсказываем вероятность antecedent для каждого отрезка. Жадно собираем cluster. Современный вариант по умолчанию.
5. **Generative (2024+).** Просим LLM: «Перечисли все местоимения в тексте и их antecedent». На лёгких случаях работает хорошо, на длинных документах и редких отсылках буксует.

> 🎒 **На пальцах.** Разница между вторым и четвёртым подходом — это цена перебора. Mention-pair сравнивает каждую пару: на 100 mention это 100 × 99 ÷ 2 = 4 950 сравнений. Span-based идёт дальше и перебирает вообще все возможные отрезки текста — поэтому в нём и стоит жёсткое ограничение на длину отрезка, обычно 10 слов, иначе перебор взорвётся.

**The evaluation metrics.** Пять стандартных метрик (MUC, B³, CEAF, BLANC, LEA), потому что ни одна метрика в одиночку не отражает качество кластеризации. Среднее первых трёх принято называть CoNLL F1. Уровень 2026 года на CoNLL-2012: около 83 F1.

> 🎒 **На пальцах.** Пять метрик на одну задачу — признак того, что «правильность» тут неоднозначна. Разбили один верный cluster из 5 mention на два по 2 и 3 — сколько ошибок вы сделали, одну или пять? MUC скажет одну, B³ насчитает больше. Поэтому и смотрят среднее, а не одно число. И 83 F1 значит, что примерно каждое шестое решение всё ещё неверное.

**Known hard cases.**

- Определённые описания, отсылающие к сущности, введённой страницами раньше.
- Bridging anaphora («the wheels» → упомянутая ранее машина).
- Нулевая анафора в языках вроде китайского и японского.
- Cataphora (местоимение перед тем, к чему оно отсылает): "When **she** walked in, Mary smiled."

> 🎒 **На пальцах.** Cataphora ломает главную эвристику всех простых систем — «antecedent всегда левее». В примере «she» стоит раньше «Mary», и правило «искать назад» не найдёт ничего. Такие случаи редки, порядка 1-2% местоимений, но именно они дают самые нелепые ошибки в готовом продукте.

```figure
coref-links
```

## Build It

### Step 1: pretrained neural coreference (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

На документе подлиннее вы получите примерно такое:
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

> 🎒 **На пальцах.** Посмотрите на пример из кода: «Apple announced new products. The company said they would ship soon.» Слово «they» здесь отсылает не к людям, а к продуктам — и модель обязана это понять из глагола «would ship». Именно поэтому Cluster 1 объединяет Apple с «The company», а «they» уходит по смыслу к продуктам. Три слова, два разных cluster.

### Step 2: rule-based pronoun resolver (teaching)

Реализация только на стандартной библиотеке лежит в `code/main.py`:

1. Извлечь mention: именованные сущности (отрезки с заглавной буквы), местоимения (поиск по словарю), определённые описания («the X»).
2. Для каждого местоимения посмотреть на предыдущие K mention и оценить их по:
   - согласованию по роду и числу (эвристика)
   - близости (кто ближе, тот и выигрывает)
   - синтаксической роли (подлежащие предпочтительнее)
3. Связать с самым высоко оценённым antecedent.

Это не конкурент нейросетям. Но так видно пространство поиска и те решения, которые end-to-end модель принимает внутри себя.

> 🎒 **На пальцах.** Три сигнала — это буквально три числа, которые складываются. Если K = 5, то на каждое местоимение вы проверяете пять предыдущих mention и выбираете одно из пяти. Даже случайный выбор дал бы 20% правильных ответов, а такие эвристики вытягивают до 60-70% на простых текстах — за тридцать строк кода без всякого обучения.

### Step 3: using LLMs for coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Два режима отказа, за которыми надо следить. Первый: LLM склеивает лишнее («him» и «her», указывающие на двух разных людей). Второй: LLM молча теряет mention в длинных документах. Всегда проверяйте по смещениям отрезков в тексте.

> 🎒 **На пальцах.** Проверка по смещениям — простая и обязательная: возьмите каждую строку из выданного JSON и найдите её в исходном тексте по индексу. Если модель вернула «the company», а в тексте такой подстроки нет, она её выдумала. На документе в 5 000 слов таких выдумок и пропаж набирается прилично, а без проверки вы их вообще не увидите.

### Step 4: evaluation

Стандартный скрипт conll-2012 считает MUC, B³, CEAF-φ4 и выводит их среднее. Для внутренней оценки начните с precision и recall на уровне отрезков по своему размеченному тестовому набору, а потом добавьте F1 по связыванию mention.

> 🎒 **На пальцах.** Начинать с precision и recall по отрезкам правильно потому, что это самая дешёвая проверка: нашли ли вы вообще нужные куски текста. Если модель не заметила половину mention, то никакая кластеризация уже не спасёт — верхняя граница качества сразу 50%. Сначала находим, потом группируем.

## Pitfalls

- **Singleton explosion.** Некоторые системы объявляют каждый mention отдельным cluster. B³ к этому снисходителен. MUC за это наказывает. Всегда смотрите все три метрики.
- **Pronouns in long context.** На документах длиннее 2 000 токенов качество падает примерно на 15 F1. Режьте аккуратно.
- **Gender assumptions.** Жёстко прописанные правила по роду ломаются на небинарных людях, организациях и животных. Берите обученные модели или нейтральный скоринг.
- **LLM drift on long docs.** Один вызов API не способен надёжно кластеризовать mention через 50+ абзацев. Используйте скользящее окно и слияние.

> 🎒 **На пальцах.** Падение на 15 F1 — это с 83 до 68, то есть почти каждое третье решение неверно. Практический вывод: если ваши документы длиннее 2 000 токенов, режьте их на окна с запасом внахлёст, разрешайте coreference внутри окна, а потом сливайте cluster по общим mention на стыках.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| English, single document | `en_coreference_web_trf` (spaCy-experimental) or AllenNLP neural coref |
| Multilingual | SpanBERT / XLM-R trained on OntoNotes or Multilingual CoNLL |
| Cross-document event coref | Specialized end-to-end models (2025–26 SOTA) |
| Quick LLM baseline | GPT-4o / Claude with structured-output coref prompt |
| Production dialog systems | Rule-based fallback + neural primary + manual review for critical slots |

Схема интеграции, которая доезжает до продакшена в 2026-м: сначала NER, потом coreference, потом слияние coreference-cluster в NER-сущности. Задачи дальше по конвейеру видят одну сущность на cluster, а не одну сущность на mention.

> 🎒 **На пальцах.** Порядок здесь принципиален. NER находит «Apple» и «Tim Cook». Coreference добавляет к «Apple» ещё «the company» и «they». После слияния в графе знаний остаётся одна вершина Apple с 20 упоминаниями, а не 20 разрозненных строк. Тот же текст, в пять раз больше извлечённых фактов.

## Ship It

Сохраните как `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Exercises

1. **Easy.** Прогоните rule-based резолвер из `code/main.py` на 5 самостоятельно составленных абзацах. Померьте точность связывания mention относительно эталона.
2. **Medium.** Примените предобученную нейросетевую coreference-модель к новостной статье. Сравните её cluster со своей ручной разметкой. Где она ошиблась?
3. **Hard.** Соберите NER-конвейер с coreference: сначала NER, потом слияние через coreference-cluster. Померьте прирост покрытия сущностей относительно чистого NER на 100 статьях.

> 🎒 **На пальцах.** Подсказка к первому заданию: составьте абзацы так, чтобы каждый проверял ровно одну эвристику. Один — на близость («John saw Bill. He waved.» — кто именно махнул?). Один — на согласование по роду. Один — на cataphora, где резолвер обязан провалиться. Пять абзацев с известным ответом дают вам понятную шкалу из пяти делений и сразу показывают, какое правило слабое.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Mention | «Отсылка» | Отрезок текста, указывающий на сущность (имя, местоимение, именная группа). |
| Antecedent | «То, к чему относится "it"» | Более раннее упоминание, с которым связано более позднее. |
| Cluster | «Все упоминания сущности» | Множество mention, указывающих на одну и ту же сущность реального мира. |
| Anaphora | «Отсылка назад» | Позднее упоминание отсылает к раннему («he» → «John»). |
| Cataphora | «Отсылка вперёд» | Раннее упоминание отсылает к позднему («When he arrived, John...»). |
| Bridging | «Неявная отсылка» | «I bought a car. The wheels were bad.» (колёса ТОЙ САМОЙ машины.) |
| CoNLL F1 | «Число из лидербордов» | Среднее F1-оценок MUC, B³ и CEAF-φ4. |

## Further Reading

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — каноническая глава из учебника.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end подход.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — предобучение, улучшающее coreference.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — тот самый бенчмарк.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — классика на правилах.
