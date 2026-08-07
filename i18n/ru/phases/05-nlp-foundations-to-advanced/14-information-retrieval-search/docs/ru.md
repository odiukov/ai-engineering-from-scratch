<!-- i18n:manual -->
# Информационный поиск и поисковые системы

> BM25 точен, но хрупок. Dense забрасывает широкую сеть, но теряет ключевые слова. Гибрид — дефолт 2026 года. Всё остальное — настройка.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## The Problem

Пользователь набирает «what happens if someone lies to get money» и рассчитывает найти статью, которая это действительно покрывает: «Section 420 IPC». Поиск по ключевым словам не найдёт её вообще — общих слов нет. Семантический поиск тоже промахнётся, если эмбеддинги не учили на юридических текстах. Настоящему поиску приходится уметь и то, и другое.

IR — это пайплайн под любой RAG-системой, любой строкой поиска, любым нечётким лукапом в документации. Работающая в проде архитектура 2026 года — не один метод. Это цепочка дополняющих друг друга методов, где каждый ловит промахи предыдущего.

Этот урок собирает каждое звено и называет, какие промахи оно ловит.

> 🎒 **На пальцах.** Запрос и нужный документ здесь не делят ни одного слова: в запросе «lies», «money», в статье «Section 420 IPC», «cheating». Совпадений ноль, значит поиск по словам вернёт ноль. Зато если пользователь введёт именно «420 IPC», выиграет уже поиск по словам, а смысловой утонет среди тысяч других номеров. Отсюда вся конструкция урока: два разных инструмента для двух разных промахов.

## The Concept

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Четыре слоя. Берите те, что нужны.

1. **Sparse retrieval (BM25).** Быстрый, точный на дословных совпадениях, беспомощный в семантике. Именно он правильно находит номера статей, артикулы товаров, тексты ошибок и имена собственные. Цифра «меньше 10 мс на запрос по миллионам документов», которую все повторяют, относится к движкам вроде Lucene, Elasticsearch и OpenSearch: они отдают BM25 из inverted index. Написанная с нуля версия из Step 1 на каждом запросе оценивает вообще все документы корпуса — правильный способ выучить формулу и неправильный способ держать нагрузку.
2. **Dense retrieval.** Кодируем запрос и документы в векторы. Ищем ближайших соседей. Ловит перефразировки и смысловую близость. Промахивается на дословных совпадениях, отличающихся одним символом. 50-200 мс на запрос с FAISS или векторной базой.
3. **Fusion.** Сливаем два ранжированных списка — от sparse и от dense. Reciprocal Rank Fusion (RRF) — простой дефолт, потому что он игнорирует сырые score (они живут в разных шкалах) и смотрит только на позиции в списке. Взвешенное слияние имеет смысл, когда вы точно знаете, что в вашем домене один сигнал сильнее.
4. **Cross-encoder rerank.** Берём top-30 после слияния. Прогоняем cross-encoder (запрос и документ подаются вместе, оценивается каждая пара). Оставляем top-5. Cross-encoder медленнее bi-encoder на каждой паре, но заметно точнее. Экономия в том, что он запускается только на 30 парах.

> 🎒 **На пальцах.** Сравните числа: BM25 из настоящего поискового движка — до 10 мс, dense — 50-200 мс, то есть в 5-20 раз дольше. И это при том, что движок трогает только документы, где встретилось слово из запроса, а наш учебный класс из Step 1 честно пересчитывает весь корпус. А cross-encoder такой тяжёлый, что по миллиону документов его не запустишь никогда. Поэтому его пускают на 30 пар: 30 сравнений вместо 1 000 000: в 33 тысячи раз меньше работы.

Трёхсторонний поиск (BM25 + dense + learned-sparse вроде SPLADE) обгоняет двухсторонний на бенчмарках 2026 года, но требует инфраструктуры под learned-sparse индексы. Для большинства команд оптимум — двухсторонний плюс cross-encoder rerank.

> 🎒 **На пальцах.** Порядок слоёв — это воронка. Первые два дают по 30 кандидатов каждый, RRF сводит их в один список из 30-60, cross-encoder оставляет 5. Каждый следующий слой дороже за документ, но и документов ему достаётся меньше. Это стандартный приём: дёшево отсеять почти всё, дорого разобраться с остатком.

```figure
gx-hybrid-retrieval
```

## Build It

### Step 1: BM25 from scratch

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.doc_freqs = [Counter(d) for d in self.corpus]
        self.doc_lens = [len(d) for d in self.corpus]
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def _score_tokens(self, q_tokens, doc_idx):
        freq = self.doc_freqs[doc_idx]
        dl = self.doc_lens[doc_idx]
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def score(self, query, doc_idx):
        return self._score_tokens(tokenize(query), doc_idx)

    def rank(self, query, top_k=10):
        q_tokens = tokenize(query)
        scored = [
            (s, i)
            for s, i in ((self._score_tokens(q_tokens, i), i) for i in range(self.n_docs))
            if s > 0.0
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:top_k]
```

> 🎒 **На пальцах.** Смотрите на `idf`. Пусть в корпусе 1000 документов. Слово встречается в 10 из них: log(1 + 990.5/10.5) = 4.56. Слово встречается в 500: log(1 + 500.5/500.5) = 0.69. Редкое слово весит в шесть с лишним раз больше частого. Это и есть весь смысл inverted index плюс IDF: искать по редкому, а не по «the».

Полезно знать два параметра. `k1=1.5` управляет насыщением по частоте термина; больше — больше вес повторам. `b=0.75` управляет нормировкой по длине; 0 полностью игнорирует длину документа, 1 нормирует целиком. Значения по умолчанию — рекомендации Робертсона из оригинальной статьи, и трогать их почти никогда не нужно.

> 🎒 **На пальцах.** Проверьте насыщение на бумаге. Пусть документ средней длины, тогда знаменатель равен `f + 1.5`. При f = 1 вклад равен 1×2.5/2.5 = 1.0. При f = 10 он равен 10×2.5/11.5 = 2.17. Десять вхождений слова весят не в десять раз больше одного, а всего вдвое. Именно так BM25 защищается от текстов, где слово вбито в страницу сто раз.

Три детали реализации, которые здесь не для красоты:

- **Частоты терминов и длины документов посчитаны заранее, в `__init__`.** Если пересобирать `Counter(doc)` внутри цикла оценки, каждый запрос превращается в полную повторную обработку корпуса.
- **Запрос токенизируется один раз на вызов `rank`**, а не один раз на документ.
- **`rank` выбрасывает документы с нулевым score и разрывает ничьи по возрастанию индекса.** Сортировка кортежей `(score, idx)` с `reverse=True` упорядочивает ничьи по *убыванию* индекса, а это значит, что ваш top-k зависит от того, в каком порядке документы случайно загрузились. А документ, у которого с запросом нет ни одного общего слова, вообще ничего про этот запрос не сообщает: вернуть его — значит набить пул кандидатов шумом, который следующие этапы слияния прилежно отранжируют.

Этот класс — линейный проход по корпусу: нормально для учебных корпусов из этого урока и неправильно уже на нескольких тысячах документов. Продакшен-BM25 работает поверх inverted index и трогает только те документы, в которых встретился термин из запроса.

### Step 2: dense retrieval with a bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

> 🎒 **На пальцах.** `normalize_embeddings=True` делает длину каждого вектора равной 1. После этого скалярное произведение и косинус — одно и то же число, и всё сравнение корпуса с запросом умещается в одно умножение матриц: (N × 384) на (384 × 1). Для 100 000 документов это 38.4 миллиона умножений — для NumPy доли секунды.

L2-нормируйте эмбеддинги, чтобы скалярное произведение равнялось косинусу. `all-MiniLM-L6-v2` — 384 измерения, быстрый и достаточно сильный для большинства англоязычных задач поиска. Для многоязычной работы берите `paraphrase-multilingual-MiniLM-L12-v2`. За максимальным качеством — `bge-large-en-v1.5` или `e5-large-v2`.

> 🎒 **На пальцах.** Размерность — это цена за документ. 384 числа по 4 байта — 1.5 КБ на документ, значит миллион документов займёт 1.5 ГБ в памяти. У `bge-large` 1024 измерения, то есть уже 4 ГБ. Качество растёт, счёт за память тоже.

### Step 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

> 🎒 **На пальцах.** Посчитайте вручную. Документ на первом месте в BM25 и отсутствующий в dense получает 1/(60+0+1) = 0.0164. Документ, стоящий третьим в обоих списках, получает 1/63 + 1/63 = 0.0317. Второй выигрывает. Формула поощряет «хорошо в обоих» сильнее, чем «отлично в одном» — ровно то поведение, ради которого гибрид и строится.

Константа `k=60` пришла из оригинальной статьи про RRF. Больший `k` сглаживает разницу между позициями; меньший — усиливает господство верхних мест. 60 — опубликованное значение по умолчанию, и подстраивать его почти никогда не нужно.

> 🎒 **На пальцах.** Посмотрите, что делает `k`. При k = 60 первое место даёт 0.0164, десятое — 1/70 = 0.0143, разница всего 15%. При k = 1 первое место даёт 0.5, десятое — 0.091, разница в пять с половиной раз. Большой `k` говорит «оба списка примерно равны», маленький — «верю только верхушке».

### Step 4: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

_RERANKER = None


def get_reranker(model_id="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(model_id)
    return _RERANKER


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=None):
    if reranker is None:
        reranker = get_reranker()
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

> 🎒 **На пальцах.** Проследите за размерами. `pool_size=30`: BM25 даёт до 30 кандидатов (меньше, если в корпусе нашлось меньше документов хотя бы с одним словом из запроса — нулевые он выбрасывает), dense даёт 30, RRF сводит их (уникальных получится от 30 до 60) и снова обрезает до 30. Потом собираются пары «запрос-документ», cross-encoder их оценивает, и наружу выходит `top_k=5`. Из миллиона до пяти за три шага.

Три этапа собраны вместе. BM25 находит лексические совпадения. Dense находит смысловые. RRF сливает два ранжирования без калибровки score. Cross-encoder переоценивает top-30, подавая запрос и документ вместе, и ловит тонкую релевантность, которую упустил bi-encoder. Оставляем top-5.

Обратите внимание на ленивый `get_reranker`. Значения аргументов по умолчанию вычисляются один раз, в момент определения функции, поэтому запись `reranker=CrossEncoder(...)` — как и подстановка модульного экземпляра в дефолт — скачивает и загружает cross-encoder ровно тогда, когда кто угодно импортирует этот модуль: включая сборщик тестов и любой скрипт, которому нужен был только `tokenize`.

> 🎒 **На пальцах.** Разница между bi-encoder и cross-encoder — как между «прочитать резюме заранее» и «провести собеседование». Bi-encoder кодирует документы один раз, заранее, и потом только сравнивает векторы. Cross-encoder читает запрос и документ вместе, для каждой пары заново. Отсюда и точность, и цена.

### Step 5: evaluation

| Metric | Meaning |
|--------|---------|
| Recall@k | Из запросов, где нужный документ существует, как часто он попадает в top-k? |
| MRR (Mean Reciprocal Rank) | Среднее от 1/ранг первого релевантного документа. |
| nDCG@k | Учитывает градации релевантности, а не только «релевантно / нет». |

Для RAG важнее всего **Recall@k** у retriever. Ваш reader не сможет ответить, если нужного пассажа нет в найденном наборе.

Совет по отладке: для проваленных запросов сравните ранжирования sparse и dense. Если один находит нужный документ, а другой нет, у вас либо расхождение словарей (лечится добавлением недостающей половины), либо смысловая неоднозначность (лечится лучшими эмбеддингами или reranker).

> 🎒 **На пальцах.** MRR считается легко. Нужный документ на 1-м месте — 1.0, на 2-м — 0.5, на 3-м — 0.33, на 10-м — 0.1. Средняя по всем запросам и есть MRR. Если по 4 запросам ранги были 1, 2, 4 и 10, то MRR = (1 + 0.5 + 0.25 + 0.1) / 4 = 0.46.

## Use It

Стек 2026 года:

| Scale | Stack |
|-------|-------|
| 1k-100k документов | BM25 в памяти + эмбеддинги `all-MiniLM-L6-v2` + RRF. Отдельная база не нужна. |
| 100k-10M документов | FAISS или pgvector для dense + Elasticsearch / OpenSearch для BM25. Запускать параллельно. |
| 10M+ документов | Qdrant / Weaviate / Vespa / Milvus с поддержкой гибрида. Cross-encoder rerank поверх top-30. |
| Максимальное качество | Трёхсторонний (BM25 + dense + SPLADE) + ColBERT-реранжирование с поздним взаимодействием |

Что бы вы ни выбрали, заложите бюджет на оценку. Сначала померьте recall поиска, потом — сквозную точность RAG. Reader не починит то, что не нашёл retriever.

> 🎒 **На пальцах.** Обратите внимание на первую строку: до 100 000 документов никакая векторная база не нужна вообще. 100 000 × 384 числа — это 150 МБ, они спокойно живут в оперативке обычного ноутбука, а поиск — одно умножение матриц. Инфраструктура начинается там, где кончается память, а не там, где начинается энтузиазм.

### The hard-won lessons from 2026 production RAG

- **80% of RAG failures trace to ingestion and chunking, not the model.** Команды неделями меняют LLM и подкручивают промпты, пока поиск тихо возвращает не тот контекст каждый третий запрос. Сначала чините нарезку на чанки.
- **Chunking strategy matters more than chunk size.** Нарезка фиксированной длины рвёт таблицы, код и вложенные заголовки. Дефолт — нарезка по предложениям; семантическая или LLM-нарезка окупается на технической документации и руководствах к продуктам.
- **Parent-doc pattern.** Ищите по маленьким «дочерним» чанкам ради точности. Когда несколько дочерних из одного родительского раздела попали в выдачу, подставляйте вместо них родительский блок, чтобы сохранить контекст. Это стабильно поднимает качество ответов без всякого переобучения.
- **k_rerank=3 is usually optimal.** Каждый лишний чанк сверх этого добавляет стоимость в токенах и задержку генерации, не улучшая ответ. Если у вас k = 8 всё ещё лучше, чем k = 3, значит reranker работает плохо.
- **HyDE / query expansion.** Сгенерируйте гипотетический ответ на запрос, закодируйте его и ищите по нему. Так закрывается разрыв в формулировках между коротким вопросом и длинным документом. Бесплатный прирост точности без обучения.
- **Context budget under 8K tokens.** Если вы регулярно упираетесь в этот лимит, порог reranker слишком мягкий.
- **Version everything.** Промпты, правила нарезки, модель эмбеддингов, reranker. Любой незаметный дрейф тихо ломает качество ответов. Гейты в CI по faithfulness, context precision и доле неотвеченных вопросов останавливают регрессии до того, как их увидят пользователи.
- **Three-way retrieval (BM25 + dense + learned-sparse like SPLADE) outperforms two-way** на бенчмарках 2026 года, особенно на запросах, где имена собственные смешаны с семантикой. Внедряйте, когда инфраструктура тянет SPLADE-индексы.

Грамотный дизайн поиска снижает галлюцинации на 70-90% по отраслевым замерам 2026 года. Основной прирост качества RAG даёт улучшение поиска, а не дообучение модели.

> 🎒 **На пальцах.** Сложите два числа из этого списка: 80% провалов сидят в нарезке, а хороший поиск убирает 70-90% галлюцинаций. Вывод грубый, но верный: день, потраченный на нарезку документов, стоит больше месяца экспериментов с моделями. И правило k_rerank = 3 туда же — три чанка вместо восьми экономят токены и время, ничего не теряя.

## Ship It

Сохраните как `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

> 🎒 **На пальцах.** Заметьте два отказа в конце. Только dense запрещён там, где есть артикулы и коды ошибок, — потому что эмбеддинги путают SKU-4471 и SKU-4417. А пропускать rerank запрещено в юридических и медицинских задачах, потому что именно финальные 5 документов увидит человек.

## Exercises

1. **Easy.** Реализуйте `hybrid_search` из урока на корпусе из 500 документов. Прогоните 20 запросов. Сравните recall на 5 у BM25-only, dense-only и гибрида.
2. **Medium.** Добавьте расчёт MRR. Для каждого тестового запроса с известным правильным документом найдите ранг этого документа в ранжированиях BM25, dense и гибрида. Отчитайтесь по MRR для каждого.
3. **Hard.** Дообучите dense-энкодер на своём домене через MultipleNegativesRankingLoss (Sentence Transformers). Соберите обучающую выборку из 500 пар «запрос-документ». Сравните recall до и после дообучения.

> 🎒 **На пальцах.** Подсказка к первому заданию: сначала соберите 20 запросов так, чтобы половина была дословной (номера, коды, точные фразы), а половина — описательной. Тогда картина будет читаемой: на первой половине BM25 даст recall около 0.9, а dense около 0.5, на второй — наоборот. Гибрид должен обойти обоих на объединённом наборе; если не обошёл, ищите ошибку в RRF.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BM25 | Поиск по ключевым словам | Okapi BM25. Оценивает документы по частоте термина, IDF и длине. |
| Dense retrieval | Векторный поиск | Кодирует запрос и документ в векторы, ищет ближайших соседей. |
| Bi-encoder | Модель эмбеддингов | Кодирует запрос и документ независимо. Быстрая на этапе запроса. |
| Cross-encoder | Модель-reranker | Кодирует запрос и документ вместе. Медленная, но точная. |
| RRF | Слияние ранжирований | Объединяет два ранжирования суммированием `1/(k + rank)`. |
| Recall@k | Метрика поиска | Доля запросов, где релевантный документ попал в top-k. |

## Further Reading

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — исчерпывающий разбор BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, канонический bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — learned-sparse retriever, догнавший dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — статья про RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — поиск с поздним взаимодействием.
