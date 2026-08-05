<!-- i18n:manual -->
# POS tagging и синтаксический разбор

> Грамматика какое-то время была немодной. Потом каждому LLM-пайплайну понадобилось проверять структурированное извлечение, и она вернулась.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## The Problem

Урок 01 обещал, что лемматизации нужен тег части речи. Не зная, что `running` — глагол, лемматизатор не сведёт его к `run`. Не зная, что `better` — прилагательное, не сведёт его к `good`.

За этим обещанием пряталась целая подобласть. POS tagging приписывает словам грамматические категории. Синтаксический разбор восстанавливает древесную структуру предложения: какое слово к какому относится, какой глагол управляет какими аргументами. Классический NLP оттачивал и то и другое двадцать лет. Потом deep learning свёл обе задачи к классификации токенов поверх предобученного трансформера, и исследователи ушли дальше.

Прикладники не ушли. Любой пайплайн структурированного извлечения до сих пор использует POS и деревья зависимостей под капотом. Сгенерированный LLM JSON проверяют на грамматические ограничения. Системы вопросов и ответов разбирают запрос через dependency parsing. Оценщики качества машинного перевода сверяют выравнивание деревьев разбора.

Знать это полезно. Урок даёт наборы тегов, базовые методы и ту точку, где вы перестаёте писать с нуля и вызываете spaCy.

> 🎒 **На пальцах.** Представьте склад, где все коробки свалены в кучу. POS tagging — это наклеить на каждую коробку ярлык «посуда», «книги», «инструменты». Синтаксический разбор — нарисовать схему, какая коробка на какой стоит. Без ярлыков лемматизатор не поймёт, что `running` надо свести к `run`, а не оставить как есть.

## The Concept

**POS tagging** приписывает каждому токену грамматическую категорию. Набор тегов **Penn Treebank (PTB)** — стандарт для английского. 36 тегов с различиями, которые обычному читателю кажутся занудством: `NN` — существительное в единственном числе, `NNS` — во множественном, `NNP` — имя собственное в единственном числе, `VBD` — глагол в прошедшем времени, `VBZ` — глагол в 3-м лице единственного числа настоящего времени, и так далее. Набор **Universal Dependencies (UD)** грубее (17 тегов) и не привязан к языку; он стал стандартом для межъязыковых задач.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

> 🎒 **На пальцах.** 36 против 17 — это как размер магазина одежды. В PTB отдельные полки для «глагол в прошедшем времени» и «глагол в 3-м лице». В UD всё это одна полка `VERB`. Мелкие полки точнее, но такой набор придётся заводить заново для каждого языка. В строке выше семь токенов и семь тегов — по одному на каждый, включая точку.

**Syntactic parsing** строит дерево. Два основных стиля:

- **Constituency parsing.** Именные, глагольные и предложные группы вложены друг в друга. На выходе — дерево нетерминальных категорий (NP, VP, PP), а слова висят листьями.
- **Dependency parsing.** У каждого слова есть ровно одно главное слово, от которого оно зависит, и связь помечена грамматическим отношением. На выходе — дерево, где каждое ребро — тройка (head, dependent, relation).

Dependency parsing победил в 2010-е, потому что чисто переносится между языками, особенно с языками со свободным порядком слов.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

> 🎒 **На пальцах.** Разбор зависимостей — это игра «кто чей начальник». В списке выше `running` — ROOT, то есть главный: у него начальника нет. У `cats`, `were` и `at` начальник — `running`. У `3pm` начальник — `at`. Шесть слов, пять стрелок: у дерева из N узлов всегда ровно N − 1 ребро.

```figure
pos-tagger
```

```figure
dependency-arcs
```

## Build It

### Step 1: most-frequent-tag baseline

Самый тупой POS tagger, который работает. Для каждого слова предсказываем тег, который чаще всего встречался у него в обучении.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

На корпусе Brown этот baseline даёт около 85% точности. Плохо, но это пол, ниже которого ни одна серьёзная модель падать не должна.

> 🎒 **На пальцах.** Это как отвечать на любой вопрос «а обычно как?». Слово `the` в обучении почти всегда было артиклем — значит, и здесь артикль. Работает в 85 случаях из 100. Каждое шестое слово размечено неверно: в предложении из 12 слов вы получите примерно две ошибки.

### Step 2: bigram HMM tagger

Моделируем совместную вероятность последовательности:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

Две таблицы: вероятности переходов (тег при известном предыдущем теге) и вероятности эмиссий (слово при известном теге). Обе оцениваем по счётчикам со сглаживанием Лапласа. Декодируем алгоритмом Витерби (динамическое программирование по решётке тегов).

> 🎒 **На пальцах.** Таблица переходов — это статистика «что за чем идёт», как в предсказании клавиатуры. Таблица эмиссий — «какое слово какой тег обычно порождает». Сглаживание Лапласа добавляет 0.01 к каждому счётчику, чтобы ни одна невиданная пара не получила вероятность 0: один ноль в произведении обнуляет весь путь.

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

> 🎒 **На пальцах.** Витерби перебирает не все варианты, а хранит для каждой позиции лучший путь до каждого тега. Если тегов 40, а предложение из 20 слов, полный перебор — это 40²⁰ вариантов, число с 32 нулями. Витерби делает 20 × 40 × 40 = 32 000 шагов. Разница между «никогда» и «мгновенно».

Bigram HMM на Brown даёт около 93% точности. Скачок с 85% до 93% — в основном заслуга вероятностей переходов: модель выучивает, что `DET NOUN` встречается часто, а `NOUN DET` редко.

> 🎒 **На пальцах.** 85% → 93% значит, что ошибок стало вдвое меньше: было 15 на сотню слов, стало 7. И всё потому, что модель смотрит на соседа слева. После артикля `the` почти всегда идёт существительное или прилагательное — этого хватило, чтобы убрать половину ошибок.

### Step 3: why modern taggers beat this

Вероятности переходов и эмиссий локальны. Они не улавливают, что `saw` — существительное в «I bought a saw» и глагол в «I saw the movie». CRF с произвольными признаками (суффикс, форма слова, слово слева и справа, само слово) даёт около 97%. BiLSTM-CRF или трансформер — 98% и выше.

Потолок задачи задаёт разногласие разметчиков. Люди-аннотаторы на Penn Treebank сходятся примерно в 97% случаев. Модели выше 98% скорее всего просто переобучились на тестовый набор.

> 🎒 **На пальцах.** Потолок 97% — это не про модели, это про людей. Если два лингвиста спорят о теге трёх слов из ста, «правильного ответа» на эти три слова просто нет. Модель, показавшая 99%, не стала умнее людей — она выучила привычки конкретного разметчика.

### Step 4: dependency parsing sketch

Полноценный dependency parsing с нуля выходит за рамки урока; канонический разбор есть в учебнике Джурафски и Мартина. Знать стоит два классических семейства:

- **Transition-based** парсеры (arc-eager, arc-standard) работают как shift-reduce парсер: читают токены, кладут их на стек и применяют reduce-действия, создающие дуги. Жадное декодирование быстрое. Классическая реализация — MaltParser. Современная нейросетевая версия — парсер Чена и Мэннинга.
- **Graph-based** парсеры (алгоритм Айснера, biaffine Дозата и Мэннинга) оценивают каждое возможное ребро «главное — зависимое» и выбирают максимальное остовное дерево. Медленнее, но точнее.

Для большинства прикладных задач вызывайте spaCy:

> 🎒 **На пальцах.** Transition-based — это как разбирать чемодан по одной вещи, решая на месте, куда её положить: быстро, но передумать нельзя. Graph-based — сначала оценить все пары вещей, потом собрать лучшую раскладку целиком. Для предложения из 10 слов graph-based считает около 10 × 10 = 100 оценок рёбер, transition-based — примерно 2 × 10 = 20 действий.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

Читайте колонку `dep` снизу вверх — и грамматическая структура предложения выпадает сама.

> 🎒 **На пальцах.** Посмотрите на строку `3pm`: её `head` — `at`, а у `at` head — `running`. Получается цепочка `3pm → at → running`. Так предлог связывает время с действием. И заметьте: у `running` head — он сам, это признак корня дерева.

## Use It

Каждая продакшен-библиотека NLP поставляет POS-теггер и парсер зависимостей как часть стандартного пайплайна.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). Быстро, точно, интегрировано с токенизацией, NER и лемматизацией. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (отношение зависимости).
- **Stanford NLP (stanza)**. Наследник CoreNLP от Стэнфорда. Уровень state of the art на 60+ языках.
- **trankit**. На трансформерах, хорошая точность по UD.
- **NLTK**. `pos_tag`. Пригоден, медленный, старый. Нормально для обучения.

> 🎒 **На пальцах.** Суффиксы моделей spaCy — это размер: `sm` (small) весит около 12 МБ, `trf` (transformer) — сотни мегабайт. Начинайте с `sm`: он ставится за секунды и на обычном тексте отстаёт от `trf` на пару процентов точности. Переходить на большой стоит, только когда эти проценты вам действительно мешают.

### Where this still matters in 2026

- **Lemmatization.** Уроку 01 нужен POS, чтобы лемматизировать правильно. Всегда.
- **Structured extraction from LLM outputs.** Проверяйте, что сгенерированное предложение соблюдает грамматические ограничения (например, согласование подлежащего со сказуемым, обязательные определения).
- **Aspect-based sentiment.** Дерево зависимостей говорит, какое прилагательное относится к какому существительному.
- **Query understanding.** Запрос «movies directed by Wes Anderson starring Bill Murray» разбирается на структурированные ограничения через разбор.
- **Cross-lingual transfer.** Теги UD и отношения зависимостей не привязаны к языку, что даёт zero-shot структурный анализ новых языков.
- **Low-compute pipelines.** Если трансформер выкатить нельзя, POS + разбор зависимостей + словарь-газеттир увезут вас на удивление далеко.

> 🎒 **На пальцах.** Возьмите отзыв «еда отличная, но обслуживание медленное». Мешок слов увидит одно хорошее и одно плохое прилагательное и решит «нейтрально». Разбор зависимостей скажет: «отличная» относится к «еде», «медленное» — к «обслуживанию». Две оценки вместо одной размытой — вот зачем нужны деревья.

## Ship It

Сохраните как `outputs/skill-grammar-pipeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

> 🎒 **На пальцах.** Обратите внимание на последний абзац скилла: писать свой парсер запрещено. Это не лень, а арифметика. Хороший парсер — это годы работы исследовательской группы, а `spacy.load()` — одна строка. Ваша задача — выбрать набор тегов и библиотеку, а не переизобретать Витерби.

## Exercises

1. **Easy.** Возьмите most-frequent-tag baseline на небольшом размеченном корпусе (например, подмножестве Brown из NLTK) и измерьте точность на отложенных предложениях. Проверьте результат ~85%.
2. **Medium.** Обучите bigram HMM выше и посчитайте precision/recall по каждому тегу. Какие теги HMM путает чаще всего?
3. **Hard.** С помощью разбора зависимостей spaCy извлеките тройки «подлежащее — сказуемое — дополнение» из выборки в 1000 предложений. Оцените на 50 размеченных вручную тройках. Опишите, где извлечение ломается (обычно на пассиве, однородных членах и опущенных подлежащих).

> 🎒 **На пальцах.** Подсказка ко второму заданию: путаницу удобно смотреть матрицей ошибок. Заранее предскажу два главных виновника — пара `NN`/`VB` (слова вроде `saw`, `book`, `can`) и пара `JJ`/`NN` (существительное в роли определения, как в `stone wall`). Если ваши цифры показывают то же самое, HMM обучен верно.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| POS tag | Тип слова | Грамматическая категория. В PTB их 36, в UD — 17. |
| Penn Treebank | Стандартный набор тегов | Только для английского. Подробные времена глаголов и число существительных. |
| Universal Dependencies | Многоязычный набор тегов | Грубее PTB, не привязан к языку, стандарт для межъязыковых задач. |
| Dependency parse | Дерево предложения | У каждого слова одно главное слово, у каждого ребра — грамматическое отношение. |
| Viterbi | Динамическое программирование | Находит самую вероятную последовательность тегов по эмиссиям и переходам. |

## Further Reading

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — канонический учебник по POS и синтаксическому разбору.
- [Universal Dependencies project](https://universaldependencies.org/) — межъязыковой набор тегов и коллекция трибанков, которыми пользуется любой многоязычный парсер.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — практический справочник по каждому атрибуту объекта `Token`.
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — статья, которая вывела нейросетевые парсеры в мейнстрим.
