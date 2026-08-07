<!-- i18n:manual -->
# Embedding-модели: глубокое погружение, 2026

> Word2Vec давал вектор на слово. Современные embedding-модели дают вектор на целый абзац, работают между языками, выдают sparse-, dense- и multi-vector-представления и подгоняются под размер вашего индекса. Ошибётесь с выбором — и RAG достанет не то.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## The Problem

Ваша RAG-система достаёт не тот фрагмент в 40% случаев. Виновата тут редко векторная база и редко промпт. Виновата embedding-модель.

Выбрать embedding в 2026 году — значит выбрать сразу по пяти осям:

1. **Dense vs sparse vs multi-vector.** Один вектор на весь фрагмент, или по вектору на токен, или разреженный взвешенный мешок слов.
2. **Language coverage.** Одноязычные английские модели всё ещё выигрывают на чисто английских задачах. Многоязычные выигрывают, когда корпус смешанный.
3. **Context length.** 512 токенов против 8 192 против 32 768 — причём реальная полезная ёмкость обычно составляет 60-70% от заявленного максимума.
4. **Dimension budget.** 3 072 float в полной точности = 12 КБ на вектор. На 100 млн векторов хранение стоит $1 300 в месяц. Matryoshka-усечение режет эту цифру в 4 раза.
5. **Open vs hosted.** Open-weight означает, что стек и данные под вашим контролем. Hosted означает, что вы меняете контроль на «всегда самая свежая версия».

Этот урок проговаривает компромиссы вслух, чтобы вы выбирали по фактам, а не по тому, что было модно в прошлом квартале.

> 🎒 **На пальцах.** Откуда 12 КБ: 3 072 числа, каждое float32 занимает 4 байта, 3072 × 4 = 12 288 байт. Умножьте на 100 млн векторов — получится больше терабайта, отсюда и $1 300 в месяц. Урезали вектор до 768 измерений — платите $325. Одна цифра в конфиге, четырёхкратная разница в счёте.

## The Concept

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings.** Один вектор на фрагмент текста (обычно 384-3 072 измерения). Косинусная близость ранжирует фрагменты по смысловой близости. Это OpenAI `text-embedding-3-large`, dense-режим BGE-M3, Voyage-3. Выбор по умолчанию.

**Sparse embeddings.** Подход в духе SPLADE. Трансформер предсказывает вес для каждого токена словаря, а потом почти все веса обнуляет. На выходе — разреженный вектор длиной со словарь. Ловит буквальные совпадения слов (как BM25), но веса выучены, а не посчитаны формулой. Силён на запросах, где важны конкретные ключевые слова.

**Multi-vector (late interaction).** ColBERTv2, Jina-ColBERT. По вектору на каждый токен. Скоринг через MaxSim: для каждого токена запроса ищем самый похожий токен документа и складываем оценки. Дороже и хранить, и считать, зато выигрывает на длинных запросах и узкоспециальных корпусах.

> 🎒 **На пальцах.** Три режима — три способа описать книгу. Dense: одна фраза «детектив про Лондон». Sparse: список слов с весами — «Холмс: 0.9, трубка: 0.4, Лондон: 0.6». Multi-vector: описание каждой страницы отдельно. Первое компактно, третье точно: у BGE-M3 dense-вектор занимает 1 024 числа, а multi-vector на фрагмент из 200 токенов — 200 × 1 024 = 204 800 чисел, то есть в 200 раз больше места.

**BGE-M3: all three at once.** Одна модель выдаёт dense-, sparse- и multi-vector-представления одновременно. Каждое можно спрашивать отдельно, а оценки потом складывать с весами. В 2026-м это выбор по умолчанию, если хочется гибкости из одного чекпоинта.

> 🎒 **На пальцах.** Это как один прогон сканера, который сразу отдаёт и фото страницы, и распознанный текст, и оглавление. Три индекса, но модель вы грузите в память один раз и считаете один forward pass — а не три разные модели по 500 МБ каждая.

**Matryoshka Representation Learning.** Модель обучена так, что первые N измерений вектора сами по себе являются полноценным embedding. Обрежьте вектор с 1 536 измерений до 256 — заплатите примерно 1% точности и получите шестикратную экономию памяти. Поддерживается в OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

> 🎒 **На пальцах.** Название честное: как матрёшка, где каждая внутренняя фигурка — тоже целая матрёшка, а не обломок. Считаем экономию: 1536 ÷ 256 = 6 раз меньше места. Точность падает на 1% — то есть из 100 запросов один найдёт не тот ответ. Обычно шестикратная экономия того стоит.

### The MTEB leaderboard tells a partial story

Massive Text Embedding Benchmark — 56 задач восьми типов на старте (2022), больше 100 задач в MTEB v2. В начале 2026-го Gemini Embedding 2 лидирует в retrieval (67.71 MTEB-R). Cohere embed-v4 ведёт в общем зачёте (65.2 MTEB). BGE-M3 — лучшая многоязычная open-weight (63.0). Лидерборд необходим, но недостаточен: всегда проверяйте модели на своём домене.

> 🎒 **На пальцах.** Лидерборд — это как средний балл аттестата у абитуриента. 67.71 против 63.0 звучит убедительно, но разрыв в 4.7 балла посчитан на чужих задачах. Если у вас корпус медицинских выписок, специализированная модель с MTEB 58 может обойти лидера. Проверяйте на своих данных, а не на чужой таблице.

### The three-tier pattern

| Use case | Pattern |
|----------|---------|
| Fast first-pass | Dense bi-encoder (BGE-M3, text-3-small) |
| Recall boost | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Precision on top-50 | Multi-vector (ColBERTv2) or cross-encoder reranker |

Большинство продакшен-стеков используют все три уровня.

> 🎒 **На пальцах.** Это воронка, как при найме. Bi-encoder быстро просматривает миллион резюме и оставляет тысячу. Sparse добавляет тех, кого dense пропустил из-за редкого термина. Cross-encoder читает оставшиеся 50 внимательно и расставляет по местам. Cross-encoder на миллионе документов работал бы часами — поэтому его и пускают только на последние 50.

```figure
gx-matryoshka
```

## Build It

### Step 1: baseline — dense embeddings with Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` делает скалярное произведение равным косинусной близости. Ставьте всегда.

> 🎒 **На пальцах.** Нормализация — это как привести все векторы к длине ровно 1. Тогда формула косинуса `(a·b)/(|a||b|)` превращается просто в `a·b`, потому что знаменатель равен 1. Строка `emb @ q_emb` даёт три числа — по одному на каждое предложение корпуса. У запроса про iPhone самое большое число будет у первого предложения.

### Step 2: Matryoshka truncation

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-12)  # guard: a head slice can be all zeros

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Нормализуйте заново после усечения — и подстрахуйте норму. Ничто не обязывает первые `dim` координат вектора быть отличными от нуля; чем короче обрезок, тем вероятнее, что так и случится. Голое деление тогда даёт `nan` на всю строку, а NaN молча расползается через скалярное произведение, так что пострадавший фрагмент не выдаёт ошибку, а просто выпадает из ранжира. Nomic v1.5, OpenAI text-3 и Voyage-4 обучены так, что первые несколько уровней проходят без потерь. Модели без Matryoshka (изначальный Sentence-BERT) от усечения деградируют резко.

> 🎒 **На пальцах.** Почему нужна повторная нормализация: у исходного вектора длина 1, но если выбросить хвост из 128 чисел, оставшаяся часть короче — например, 0.87. Косинусные оценки поедут. Деление на `norms` растягивает обрезок обратно до длины 1. Забудете эту строку — получите тихо испорченный ранжир.

> 🎒 **На пальцах.** Что делает `np.maximum(norms, 1e-12)`. Представьте, что у одного документа первые 128 координат оказались нулями (весь его смысл лежит в хвосте). Длина обрезка — ровно 0, и `0 / 0` в NumPy даёт не ошибку, а `nan` — «не число». Дальше `nan` заражает всё, к чему прикоснётся: `nan · что угодно = nan`, а любое сравнение с `nan` ложно, поэтому при сортировке такой документ просто уедет в конец списка. Ни исключения, ни предупреждения — фрагмент тихо перестаёт находиться. `np.maximum` заменяет нулевую норму на 1e-12 (0.000000000001), делить на неё можно, и вектор остаётся нулевым вместо того, чтобы стать NaN. Нулевой вектор даёт косинус 0 со всем на свете — это честное «не похоже ни на что», и такое поведение хотя бы видно в метриках.

### Step 3: BGE-M3 multi-functionality

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Три индекса за один вызов модели. Сложение оценок:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Подбирайте веса на своём домене.

> 🎒 **На пальцах.** Веса 0.4 + 0.2 + 0.4 в сумме дают 1.0 — это как голосование трёх экспертов, где sparse имеет половину голоса остальных. Если у вас юридические тексты с точными номерами статей, поднимите sparse до 0.4, а colbert опустите до 0.2. Веса не священны: это ровно три числа, которые нужно подобрать замерами.

### Step 4: MTEB eval on a custom task

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Прогоняйте кандидатов на *репрезентативном* подмножестве. Не доверяйте одному лишь месту в лидерборде — ваш домен решает.

> 🎒 **На пальцах.** Три задачи в списке выбраны не случайно: ArguAna — про аргументы, SciFact — про научные факты, NFCorpus — про медицину. Если ваш продукт про медицину, NFCorpus скажет о модели больше, чем весь остальной лидерборд. Три задачи вместо ста — прогон занимает минуты, а не сутки.

### Step 5: hand-rolled cosine from scratch

Смотрите `code/main.py`. Там усреднённые embedding на hashing trick, только стандартная библиотека. С трансформерными embedding это не конкурирует, зато показывает форму процесса: токенизация → вектор → нормализация → скалярное произведение.

> 🎒 **На пальцах.** Hashing trick — это как раскладывать письма по 1 000 ящиков по остатку от деления номера. Слово «iPhone» всегда падает в один и тот же ящик. Столкновения бывают, качество страдает, но весь механизм умещается в 30 строк без единой зависимости — и вы своими глазами видите, что embedding не магия, а массив чисел.

## Pitfalls

- **Same model for query and doc.** Некоторые модели (Voyage, Jina-ColBERT) используют асимметричное кодирование: запрос и документ проходят по разным путям. Всегда читайте карточку модели.
- **Missing prefix.** Моделям семейства `bge-*` нужно приписывать к запросу `"Represent this sentence for searching relevant passages: "`. Забудете — потеряете 3-5 пунктов recall.
- **Over-trimming Matryoshka.** 1 536 → 256 обычно безопасно. 1 536 → 64 уже нет. Проверяйте на своём валидационном наборе и проверяйте после усечения строки с нулевой нормой — неподстрахованная повторная нормализация превращает их в NaN.
- **Context truncation.** Большинство моделей молча обрезают вход, который длиннее их максимума. Длинные документы нужно резать на чанки (см. урок 23).
- **Ignoring latency tail.** Оценки MTEB прячут p99-задержку. Модель на 600M параметров может обойти модель на 335M на два пункта, но стоить втрое дороже за запрос.

> 🎒 **На пальцах.** Самая обидная ошибка тут — забытый префикс. Вы ничего не сломали, ошибок нет, код работает, просто recall упал с 85% до 81%. Четыре пункта — это 4 неверных ответа на каждые 100 запросов, и найти причину без чтения карточки модели почти невозможно.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| English-only, fast, API | `text-embedding-3-large` or `voyage-3-large` |
| Open-weight, English | `BAAI/bge-large-en-v1.5` |
| Open-weight, multilingual | `BAAI/bge-m3` or `Qwen3-Embedding-8B` |
| Long context (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2 (137M params, MoE) |
| Storage-constrained | Matryoshka-truncated + int8 quantization |
| Keyword-heavy queries | Add SPLADE sparse, RRF-fuse with dense |

Паттерн 2026 года: начните с BGE-M3 или text-3-large, оцените на своём домене через MTEB, меняйте модель, только если доменная выигрывает больше чем на 3 пункта.

> 🎒 **На пальцах.** Порог «3 пункта» задан не из вредности. Замена модели — это переиндексация всего корпуса: на 10 млн документов это часы GPU и новый индекс рядом со старым. Выигрыш в 1 пункт такой цены не стоит, а вот 3 пункта на нужном домене — уже стоит.

## Ship It

Сохраните как `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Exercises

1. **Easy.** Закодируйте 100 предложений моделью `bge-small-en-v1.5` в полной размерности (384), затем в Matryoshka-размерности 128. Измерьте падение MRR на 10 запросах.
2. **Medium.** Сравните dense, sparse и colbert у BGE-M3 на 500 фрагментах из вашего домена. Что выигрывает по recall@10? Обгоняет ли RRF-слияние лучший одиночный режим?
3. **Hard.** Прогоните MTEB на трёх моделях-кандидатах по двум важнейшим для вас доменным задачам. Отчитайтесь по MTEB-оценке, p99-задержке на батче из 100 запросов и цене за 1 млн запросов. Выберите парето-оптимальную.

> 🎒 **На пальцах.** Подсказка к первому заданию: 384 → 128 это трёхкратное усечение, а bge-small обучалась без Matryoshka. Ждите заметного падения MRR — возможно, 0.72 → 0.61. Это и есть смысл упражнения: увидеть своими глазами, что усечение работает только на моделях, которых этому специально учили.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Dense embedding | «Тот самый вектор» | Один вектор фиксированного размера на текст. Ранжирование по косинусной близости. |
| Sparse embedding | «Выученный BM25» | По одному весу на токен словаря, почти все нули, обучается end-to-end. |
| Multi-vector | «Как в ColBERT» | По вектору на токен, скоринг через MaxSim; индекс больше, recall лучше. |
| Matryoshka | «Фокус с матрёшкой» | Первые N измерений сами по себе являются рабочим меньшим embedding. |
| MTEB | «Тот самый бенчмарк» | Massive Text Embedding Benchmark — 56 задач на старте, больше 100 в v2. |
| BEIR | «Бенчмарк по retrieval» | 18 zero-shot задач на поиск; часто цитируется как проверка устойчивости между доменами. |
| Asymmetric encoding | «Запрос ≠ документ» | Модель применяет разные проекции к запросам и к документам. |

## Further Reading

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — статья про bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — статья про лидерборд.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — единая модель с тремя режимами.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — обучающая цель «лесенка размерностей».
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction в продакшене.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — живой рейтинг.
