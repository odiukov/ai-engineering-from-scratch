<!-- i18n:manual -->
# Системы вопросов и ответов

> Современный QA сформировали три системы. Extractive находил span. Retrieval-augmented привязывал ответ к документам. Генеративная сочиняла ответ. Любой сегодняшний AI-ассистент — смесь этих трёх.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## The Problem

Пользователь набирает «When did the first iPhone launch?» и ждёт «June 29, 2007». Не «история Apple длинна и разнообразна». Не голое «2007» без предложения вокруг. Прямой, обоснованный, правильный ответ.

За последнее десятилетие в QA доминировали три архитектуры.

- **Extractive QA.** Даны вопрос и пассаж, про который известно, что ответ внутри. Нужно найти индексы начала и конца span с ответом. SQuAD — канонический бенчмарк.
- **Open-domain QA.** Пассаж не дан. Сначала найти нужный пассаж, потом извлечь или сгенерировать ответ. На этом стоит любой сегодняшний RAG-пайплайн.
- **Generative / Closed-book QA.** Большая языковая модель отвечает из своей параметрической памяти. Поиска нет вообще. Самая быстрая на инференсе, самая ненадёжная в фактах.

Тренд 2026 года — гибрид: найти несколько лучших пассажей, потом попросить генеративную модель ответить, опираясь только на них. Это и есть RAG. Урок 14 подробно разбирает половину про поиск. Этот урок строит половину про ответ.

> 🎒 **На пальцах.** Представьте экзамен. Extractive QA — «найди ответ в учебнике и подчеркни». Open-domain — «сначала найди нужную страницу, потом подчеркни». Closed-book — «учебник забрали, отвечай по памяти». Правильный ответ здесь — строка «June 29, 2007», всего 13 символов. Три архитектуры отличаются только тем, откуда эти 13 символов берутся.

## The Concept

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** Кодируем вопрос и пассаж вместе трансформером (семейство BERT). Обучаем две головы: одна предсказывает индекс начального токена ответа, другая — конечного. Функция потерь — кросс-энтропия по допустимым позициям. На выходе — span из пассажа. Никогда не галлюцинирует (по построению) и никогда не справляется с вопросами, на которые пассаж не отвечает (тоже по построению).

> 🎒 **На пальцах.** Модель не пишет ответ, она называет два числа: «начало» и «конец». Если в пассаже 40 токенов, вариантов span всего около 40 × 41 / 2 = 820 — из них модель выбирает один. Отсюда и главное свойство: выдумать текст, которого нет в пассаже, она физически не может.

**Retrieval-augmented (RAG).** Два этапа. Сначала retriever находит top-`k` пассажей в корпусе. Потом reader (extractive или генеративный) выдаёт ответ по этим пассажам. Разделение на retriever и reader позволяет обучать и оценивать их по отдельности. Современный RAG часто ставит между ними reranker.

> 🎒 **На пальцах.** Это библиотекарь плюс студент. Библиотекарь (retriever) приносит `k` книг с полки — обычно 2-5 штук из миллиона. Студент (reader) читает только их и отвечает. Если библиотекарь принёс не ту книгу, студент обречён: ответа в его руках просто нет.

**Generative.** Decoder-only LLM (GPT, Claude, Llama) отвечает из выученных весов. Шага поиска нет. Отлично справляется с общеизвестным, катастрофически плохо — с редким и свежим. Частота галлюцинаций обратно связана с тем, как часто факт встречался в предобучающих данных.

> 🎒 **На пальцах.** Дата выхода первого iPhone встречалась в интернете миллионы раз — модель её знает. Дата вашего последнего релиза встречалась ноль раз — и модель её выдумает, причём уверенным тоном. Правило простое: чем реже факт, тем выше шанс галлюцинации.

```figure
qa-span
```

## Build It

### Step 1: extractive QA with a pretrained model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

> 🎒 **На пальцах.** Разберите вывод по частям. `start: 57` и `end: 70` — это позиции символов в пассаже; 70 − 57 = 13, ровно длина строки «June 29, 2007». `score: 0.98` — уверенность модели. Модель ничего не сочинила, она просто показала пальцем на кусок текста.

`deepset/roberta-base-squad2` обучен на SQuAD 2.0, где есть вопросы без ответа. По умолчанию пайплайн `question-answering` возвращает span с наибольшим score даже тогда, когда выигрывает null score — пустой ответ он *не* отдаёт автоматически. Чтобы получить явное «ответа нет», передайте в вызов пайплайна `handle_impossible_answer=True`: тогда пустой ответ вернётся только если null score выше любого span. В любом случае всегда смотрите на поле `score`.

> 🎒 **На пальцах.** Без этого флага модель всегда что-нибудь подчеркнёт — даже если спросить у пассажа про iPhone, когда родился Пушкин. Она вернёт какой-нибудь span со score вроде 0.03. Отсюда практическое правило: заведите порог (скажем, 0.2) и всё, что ниже, считайте отказом.

### Step 2: a retrieval-augmented pipeline (sketch)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

> 🎒 **На пальцах.** Корпус здесь — 4 предложения, `top_k=2`, то есть reader получает половину корпуса. Функция `retrieve` считает косинусные близости одним умножением матриц: 4 вектора корпуса на 1 вектор вопроса, `np.argsort(-sims)` сортирует по убыванию и берёт два верхних индекса. На вопрос про iPhone всплывут строки 0 и 1, а про Android и iPod — нет.

Двухэтапный пайплайн. Dense retriever (Sentence-BERT) находит подходящие пассажи по смысловой близости. Extractive reader (RoBERTa-SQuAD) вытаскивает span с ответом из склеенных верхних пассажей. На маленьких корпусах работает как есть. Для корпуса в миллион документов берите FAISS или векторную базу.

> 🎒 **На пальцах.** Перебор всех документов подряд честен, но линеен. 4 документа — мгновенно, миллион — уже сотни миллисекунд на каждый запрос. FAISS строит индекс, который вместо миллиона сравнений делает несколько тысяч и находит почти те же ближайшие соседи.

### Step 3: generative with RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

> 🎒 **На пальцах.** Здесь `top_k=3`, значит в промпт уедут три пассажа в виде маркированного списка. Ключевая строка — «If the context does not contain the answer, say "I don't know."». Это разрешение молчать: без него модель обязана что-то придумать, потому что так её учили.

Шаблон промпта имеет значение. Если явно велеть модели опираться на контекст и отвечать «I don't know», когда контекста не хватает, частота галлюцинаций падает на 40-60% по сравнению с наивным промптом. Более сложные шаблоны добавляют цитаты, оценки уверенности и структурированный вывод.

> 🎒 **На пальцах.** 40-60% — это не мелочь. Если из 100 ответов раньше врали 20, то теперь врут 8-12. Цена вопроса — одно предложение в промпте. Ни дообучения, ни новой модели.

### Step 4: evaluation that reflects the real world

SQuAD использует **Exact Match (EM)** и **token-level F1**. EM — строгое совпадение после нормализации (нижний регистр, снятие пунктуации, удаление артиклей): либо предсказание совпало точно, либо ноль. F1 считается по пересечению токенов предсказания и эталона и даёт частичный зачёт. Обе метрики недооценивают перефразировки: «June 29, 2007» против «June 29th, 2007» обычно даёт EM = 0 (порядковое числительное ломает нормализацию), но заметный F1 за счёт общих токенов.

> 🎒 **На пальцах.** Посчитаем этот пример руками. После нормализации получаем токены [june, 29, 2007] и [june, 29th, 2007] — по 3 в каждом. Общих токенов 2 (june и 2007), значит precision = 2/3, recall = 2/3, F1 = 0.67. Человек сказал бы «ответ верный», EM говорит «ноль». Вот почему на проде одного EM мало.

Для production QA:

- **Answer accuracy** (оценка LLM или человеком, потому что метрики не ловят смысловую эквивалентность).
- **Citation accuracy.** Действительно ли процитированный пассаж подтверждает ответ? Проверяется автоматически обычным сравнением строк между сгенерированными цитатами и найденными пассажами.
- **Refusal calibration.** Когда ответа в найденных пассажах нет, система честно говорит «I don't know»? Меряйте долю ложной уверенности.
- **Retrieval recall.** До оценки reader проверьте, попадает ли нужный пассаж в top-`k` у retriever. Reader не починит пассаж, которого ему не дали.

> 🎒 **На пальцах.** Последний пункт — самый важный и самый пропускаемый. Если retrieval recall@5 равен 0.7, то потолок вашей системы — 70% ответов, каким бы умным ни был reader. Меняя LLM, вы боретесь за оставшиеся 70%, а теряете 30% ещё до неё.

### RAGAS: the 2026 production eval framework

`RAGAS` создан специально для RAG-систем и в 2026 году идёт по умолчанию. Он оценивает четыре измерения и не требует эталонных ответов:

- **Faithfulness.** Каждое ли утверждение в ответе взято из найденного контекста? Меряется через NLI-выводимость. Ваша главная метрика галлюцинаций.
- **Answer relevance.** Отвечает ли ответ на вопрос? Меряется так: из ответа генерируют гипотетические вопросы и сравнивают их с настоящим.
- **Context precision.** Какая доля найденных чанков реально относилась к делу? Низкая precision означает шум в промпте.
- **Context recall.** Содержал ли найденный набор всю нужную информацию? Низкий recall означает, что reader обречён.

Оценка без эталонов позволяет мерить качество прямо на живом трафике, без вручную собранных правильных ответов. Сверху добавьте LLM-as-judge для открытых вопросов, где метрики точного совпадения бесполезны.

`pip install ragas`. Подключите свои retriever и reader. Получите четыре числа на каждый запрос. Настройте алерты на просадки.

> 🎒 **На пальцах.** Четыре числа делятся на две пары. Context precision и context recall — это отчёт по библиотекарю. Faithfulness и answer relevance — отчёт по студенту. Если faithfulness 0.9, а context recall 0.4, менять модель бессмысленно: студент честен, книги не те.

## Use It

Стек 2026 года.

| Use case | Recommended |
|---------|-------------|
| Пассаж дан, надо найти span с ответом | `deepset/roberta-base-squad2` |
| Фиксированный корпус, closed-book неприемлем | RAG: dense retriever + LLM reader |
| Реальное время поверх хранилища документов | RAG с гибридным (BM25 + dense) retriever + reranker (урок 14) |
| Диалоговый QA (уточняющие вопросы) | LLM с историей диалога + RAG на каждом ходу |
| Строго фактические, регулируемые домены | Extractive поверх авторитетного корпуса; никогда не генеративная модель в одиночку |

Extractive QA в 2026 году немодно, потому что RAG с LLM покрывает больше случаев. Но оно всё ещё едет в прод там, где нужна дословная цитата: юридические исследования, регуляторный комплаенс, аудиторские инструменты.

> 🎒 **На пальцах.** Обратите внимание на нижнюю строку таблицы. Юристу нужна не «правильная по смыслу» фраза, а точная цитата с номером страницы. Extractive QA даёт именно её: два индекса, 13 символов, ноль сочинительства. Генеративная модель перескажет тот же смысл своими словами — и в суде это уже не документ.

## Ship It

Сохраните как `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

> 🎒 **На пальцах.** Скилл строит выбор в фиксированном порядке: архитектура, retriever, reader, оценка. Четыре решения, и все четыре надо назвать вслух. Отдельно обратите внимание на запрет: closed-book LLM в регуляторных вопросах запрещён всегда, без исключений.

## Exercises

1. **Easy.** Разверните описанный выше extractive-пайплайн SQuAD на 10 пассажах из Википедии. Придумайте 10 вопросов вручную. Померьте, как часто ответ верный. Если пассажи и вопросы чистые, вы должны получить 7-9 правильных.
2. **Medium.** Добавьте классификатор отказа. Когда лучший score поиска ниже порога (скажем, 0.3 по косинусу), возвращайте «I don't know» вместо вызова reader. Подберите порог на отложенной выборке.
3. **Hard.** Соберите RAG-пайплайн поверх корпуса из 10 000 документов на ваш выбор. Реализуйте гибридный поиск (BM25 + dense) со слиянием через RRF (см. урок 14). Померьте точность ответов с гибридным шагом и без него. Опишите, каким типам вопросов он помогает сильнее всего.

> 🎒 **На пальцах.** Подсказка ко второму заданию: не выдумывайте порог, посмотрите на числа. Прогоните 20 вопросов, на которые ответ в корпусе есть, и 20, на которые его нет, и запишите лучший косинус для каждого. Скорее всего первые дадут 0.4-0.7, а вторые 0.1-0.3. Порог ставится в зазор между этими кучками — часто это как раз около 0.3.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive QA | Найти span с ответом | Предсказать индексы начала и конца ответа внутри данного пассажа. |
| Open-domain QA | QA по корпусу | Пассаж не дан; надо сначала найти, потом ответить. |
| RAG | Найти, потом сгенерировать | Retrieval-augmented generation. Пайплайн из retriever и reader. |
| SQuAD | Канонический бенчмарк | Stanford Question Answering Dataset. Метрики EM и F1. |
| Hallucination | Выдуманный ответ | Ответ reader, не подтверждённый найденным контекстом. |
| Refusal calibration | Умение вовремя замолчать | Система честно говорит «I don't know», когда ответить не может. |

## Further Reading

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — статья про сам бенчмарк.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, канонический dense retriever для QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — статья, давшая имя RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — исчерпывающий обзор RAG.
