<!-- i18n:manual -->
# Natural Language Inference — вывод следствия из текста

> «t влечёт h» означает: человек, прочитав t, заключит, что h истинно. NLI — задача предсказать entailment / contradiction / neutral. На вид скучно, а в продакшене держит на себе половину пайплайна.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## The Problem

Вы сделали суммаризатор. Он выдал краткое изложение. Как понять, что в нём нет галлюцинации?

Вы сделали чат-бота. Он ответил «да». Как понять, что ответ подкреплён найденным фрагментом текста?

Вам нужно разложить 10 000 новостных статей по темам. Размеченных данных нет. Можно ли переиспользовать готовую модель?

Все три задачи сводятся к NLI. NLI спрашивает: дана premise `t` и hypothesis `h` — следует ли `h` из `t` (entailment), противоречит ли (contradiction) или не связано (neutral)?

> 🎒 **На пальцах.** Три разные задачи, один вопрос: «следует ли одно из другого?». Как отвёртка, которой откручивают и розетку, и ножку стула, и крышку ноутбука. Инструмент один, применений три.

- **Hallucination check:** `t` = исходный документ, `h` = утверждение из краткого изложения. Нет entailment — есть галлюцинация.
- **Grounded QA:** `t` = найденный фрагмент, `h` = сгенерированный ответ. Нет entailment — ответ выдуман.
- **Zero-shot classification:** `t` = документ, `h` = метка, переписанная фразой («This is about sports»). Есть entailment — это и есть предсказанная метка.

> 🎒 **На пальцах.** Третий пункт — самый неожиданный. Классификатор на 4 темы вы получаете, просто написав 4 фразы вида «This is about sports». Ни одного размеченного примера, ни одной эпохи обучения. Модель уже умеет отвечать на вопрос «следует ли?», а вы подсовываете ей нужный вопрос.

Одна задача, три применения в продакшене. Именно поэтому под капотом любого фреймворка для оценки RAG стоит NLI-модель.

## The Concept

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**The three labels.**

- **Entailment.** `t` → `h`. «The cat is on the mat» влечёт «There is a cat».
- **Contradiction.** `t` → ¬`h`. «The cat is on the mat» противоречит «There is no cat».
- **Neutral.** Вывода нет ни в какую сторону. «The cat is on the mat» нейтрально к «The cat is hungry».

> 🎒 **На пальцах.** Разберите третий пример. Кот лежит на коврике — голоден он или нет, из этого не следует. Может быть голоден, может быть нет. Это и есть neutral: не «не знаю, что за текст», а «текст понятен, но ответа на этот вопрос в нём нет».

**Not logical entailment.** NLI — это вывод на *естественном* языке: то, что вывел бы обычный читатель, а не строгая логика. «John walked his dog» в NLI влечёт «John has a dog», хотя строгая логика первого порядка признала бы это, только если аксиоматизировать понятие владения.

> 🎒 **На пальцах.** Формально «выгуливал собаку» не значит «владеет собакой» — может, соседскую выгуливал. Но 95 читателей из 100 сделают именно этот вывод, и NLI размечается по ним, а не по учебнику логики. Отсюда и слово natural в названии.

**Datasets.**

- **SNLI** (2015). 570 тысяч пар с ручной разметкой, в роли premise — подписи к картинкам. Узкий домен.
- **MultiNLI** (2017). 433 тысячи пар из 10 жанров. Стандартный обучающий корпус в 2026 году.
- **ANLI** (2019). Adversarial NLI. Люди специально писали примеры, чтобы ломать существующие модели. Сложнее.
- **DocNLI, ConTRoL** (2020–21). Premise длиной с целый документ. Проверяют многошаговый вывод и вывод на большом расстоянии.

> 🎒 **На пальцах.** Обратите внимание на масштаб: 570 тысяч пар в SNLI размечали руками. Если один человек разбирает пару за 15 секунд, это примерно 2 400 часов чистой работы — год full-time на одного человека. Поэтому такие датасеты делают краудсорсингом и поэтому их так мало.

**The architecture.** Трансформер-энкодер (BERT, RoBERTa, DeBERTa) читает `[CLS] premise [SEP] hypothesis [SEP]`. Представление `[CLS]` подаётся на softmax из трёх классов. Обучаете на MNLI, замеряете на отложенных бенчмарках, получаете 90 % и выше на парах из того же распределения.

> 🎒 **На пальцах.** Заметьте: оба текста идут в модель одной строкой, разделённые `[SEP]`. Это принципиально — модель сравнивает premise и hypothesis словом к слову внутри себя, а не считает два отдельных вектора и потом косинус. Отсюда и качество 90 % против примерно 60 % у простых эвристик.

**Zero-shot via NLI.** Дан документ и список возможных меток — превращаем каждую метку в hypothesis («This text is about sports»). Считаем вероятность entailment для каждой. Берём максимум. Именно этот механизм стоит за пайплайном `zero-shot-classification` в Hugging Face.

```figure
nli-router
```

## Build It

### Step 1: run a pretrained NLI model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Для NLI в продакшене открытые варианты по умолчанию — `facebook/bart-large-mnli` и `microsoft/deberta-v3-large-mnli`. DeBERTa-v3 держит первые места в рейтингах.

> 🎒 **На пальцах.** Посмотрите на три числа: 0,97 + 0,02 + 0,01 = 1,00. Это softmax из трёх классов, вероятности всегда складываются в единицу. Модель почти уверена, что «спящий на диване кот» влечёт «в комнате есть кот» — и это ровно тот вывод, который сделали бы вы.

### Step 2: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

По умолчанию шаблон такой: «This example is about {label}.» Меняется через `hypothesis_template`. Обучающие данные не нужны. Дообучение не нужно. Работает из коробки.

> 🎒 **На пальцах.** Модель ни разу не видела ваших четырёх меток. Она просто четыре раза ответила на вопрос «следует ли из этого текста фраза "This example is about finance"?» — и получила 0,92 против 0,05, 0,02 и 0,01. Победитель определяется максимумом, а не порогом.

### Step 3: faithfulness check for RAG

```python
def is_supported(claim, context, threshold=0.5):
    result = nli({"text": context, "text_pair": claim})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold


def faithfulness(answer, context, split_claims, threshold=0.5):
    claims = [c for c in split_claims(answer) if c.strip()]
    if not claims:
        return 0.0
    supported = sum(is_supported(c, context, threshold) for c in claims)
    return supported / len(claims)
```

Это и есть ядро метрики faithfulness в RAGAS. Разбейте сгенерированный ответ на атомарные утверждения, проверьте каждое против найденного контекста, верните долю тех, что дают entailment. `split_claims` — это и есть шаг разбиения: дешёвый вариант — разделитель предложений, точный — вызов LLM. Не запускайте NLI по всему ответу одним вызовом: ответ из четырёх предложений, где выдумано одно, всё равно прочитается как «в основном следует», и единственная оценка эту одну фразу спрячет.

> 🎒 **На пальцах.** Здесь две функции, и разделены они не для красоты. `is_supported` — примитив: одно утверждение против контекста, ответ «да/нет». `faithfulness` — метрика: разбивает ответ на утверждения и считает долю поддержанных. Порог 0,5 внутри примитива означает просто «модель считает entailment более вероятным, чем нет». Если ответ разбился на 6 утверждений и 5 прошли проверку, faithfulness = 5/6 ≈ 0,83. Одно непрошедшее — и есть кандидат в галлюцинации, который стоит показать человеку.

> 🎒 **На пальцах.** Почему нельзя одним вызовом. Ответ: «Париж — столица Франции. Там живёт 2 миллиона человек. Эйфелева башня открыта в 1889 году. Её высота 450 метров». Три факта из контекста, четвёртый выдуман (330 метров, а не 450). Отдайте эти четыре предложения в NLI разом — модель увидит, что почти всё в тексте подтверждается, и выдаст entailment с оценкой вроде 0,7. Проверка пройдена, галлюцинация уехала в продакшен. Разбейте на четыре утверждения — получите 3/4 = 0,75, и, что важнее, у вас на руках будет *то самое* непрошедшее предложение. Разница не в точности числа, а в разрешающей способности: 0,7 на весь ответ ничего не показывает пальцем, а 3/4 с отмеченным четвёртым утверждением — это готовый тикет для разбора.

> 🎒 **На пальцах.** Мелочь про `sum(...)`: `is_supported` возвращает `True`/`False`, а в Python `True` — это 1, `False` — это 0. Поэтому `sum` по булевым значениям просто считает, сколько утверждений прошли. Делим на `len(claims)` — получаем долю. И отдельная строка `if not claims: return 0.0`: пустой ответ нечем подтверждать, а делить на ноль нельзя, поэтому такой случай честно оценивается в 0,0, а не роняет пайплайн.

### Step 4: hand-rolled NLI classifier (conceptual)

Смотрите `code/main.py` — там игрушечная реализация на одной стандартной библиотеке: premise и hypothesis сравниваются по лексическому пересечению плюс поиск отрицаний. С трансформерами она не соперник, зато показывает форму задачи: на входе два текста, на выходе метка из трёх, функция потерь — кросс-энтропия по `{entail, contradict, neutral}`.

> 🎒 **На пальцах.** Три класса — значит случайное угадывание даёт 33 %. Игрушка на пересечении слов и отрицаниях вытянет процентов 55-60. Трансформер — за 90. Эта лесенка и показывает, сколько именно даёт настоящая модель поверх простых правил.

## Pitfalls

- **Hypothesis-only shortcuts.** Модель может предсказать метку по одной только hypothesis с точностью около 60 % на SNLI, потому что «not», «nobody», «never» коррелируют с contradiction. Хороший базовый тест на утечку метки.
- **Lexical overlap heuristic.** Эвристика подпоследовательностей («любая подпоследовательность влечётся») проходит SNLI, но проваливает HANS/ANLI. Пользуйтесь adversarial-бенчмарками.
- **Document-length degradation.** Модели, обученные на отдельных предложениях, теряют 20 и более пунктов F1 на premise длиной с документ. Для длинного контекста берите модели, обученные на DocNLI.
- **Zero-shot template sensitivity.** «This example is about {label}» против «{label}» против «The topic is {label}» способны развести точность на 10 и более пунктов. Подбирайте шаблон.
- **Domain mismatch.** MNLI обучен на общем английском. Юридическим, медицинским и научным текстам нужны доменные NLI-модели (например, SciNLI, MedNLI).

> 🎒 **На пальцах.** Первый пункт стоит осознать полностью: модель выдаёт 60 % правильных ответов, вообще не читая premise. То есть половину качества можно получить, угадывая по слову «never». Всегда прогоняйте этот базовый тест — если ваша модель даёт 65 %, она почти ничего не выучила.

## Use It

Стек 2026 года:

| Use case | Model |
|---------|-------|
| NLI общего назначения | `microsoft/deberta-v3-large-mnli` |
| Быстро / на устройстве | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot классификация (лёгкая) | `facebook/bart-large-mnli` |
| NLI на уровне документа | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Мультиязычность | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Поиск галлюцинаций в RAG | Слой NLI внутри RAGAS / DeepEval |

Мета-паттерн 2026 года: NLI — это скотч для понимания текста. Как только вам нужно «подтверждает ли A утверждение B?» или «противоречит ли A утверждению B?» — тянитесь к NLI раньше, чем к очередному вызову LLM.

> 🎒 **На пальцах.** Разница в цене огромна. Прогнать `deberta-v3-base` по 10 000 пар на своей видеокарте — минуты и ноль долларов. Те же 10 000 вопросов через API большой LLM — это десятки долларов и часы ожидания. Для вопроса «следует ли?» большая модель почти всегда избыточна.

## Ship It

Сохраните как `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

> 🎒 **На пальцах.** Ключевая строка — последняя: NLI уменьшает галлюцинации, но не убирает их. Если модель-проверяльщик сама права в 90 % случаев, то из 100 пропущенных ею утверждений примерно 10 окажутся выдумкой. Это фильтр, а не гарантия.

## Exercises

1. **Easy.** Прогоните `facebook/bart-large-mnli` на 20 придуманных вручную тройках (premise, hypothesis, метка), покрывающих все три класса. Измерьте точность. Добавьте adversarial-ловушки на «эвристику подпоследовательностей» («I did not eat the cake» против «I ate the cake») и посмотрите, сломается ли модель.
2. **Medium.** Сравните zero-shot шаблоны `"This text is about {label}"`, `"The topic is {label}"` и `"{label}"` на 100 заголовках AG News. Приведите разброс точности.
3. **Hard.** Соберите проверку faithfulness для RAG: разбиение на атомарные утверждения плюс NLI по каждому. Оцените на 50 сгенерированных RAG-ответах с эталонным контекстом. Измерьте долю ложноположительных и ложноотрицательных срабатываний против ручной разметки.

> 🎒 **На пальцах.** Подсказка к первому заданию: ловушка с тортом работает потому, что «I did not eat the cake» содержит все слова из «I ate the cake». Модель, которая смотрит на пересечение слов, скажет entailment. Правильный ответ — contradiction. На 20 примерах вы увидите это сразу, ошибка будет одна и очень заметная.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | Классификация отношения premise и hypothesis на 3 класса. |
| RTE | Recognizing Textual Entailment | Старое название NLI; задача та же. |
| Entailment | "t implies h" | Обычный читатель заключил бы, что h истинно при данном t. |
| Contradiction | "t rules out h" | Обычный читатель заключил бы, что h ложно при данном t. |
| Neutral | "undecided" | Из t к h не следует ничего ни в ту, ни в другую сторону. |
| Zero-shot classification | NLI as classifier | Переписать метки как hypothesis и взять максимум entailment. |
| Faithfulness | Is the answer supported? | NLI по паре (найденный контекст, сгенерированный ответ). |

## Further Reading

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — бенчмарк ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI в роли классификатора.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — рабочая лошадка NLI в 2026 году.
