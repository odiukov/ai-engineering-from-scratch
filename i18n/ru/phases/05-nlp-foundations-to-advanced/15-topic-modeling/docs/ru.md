<!-- i18n:manual -->
# Тематическое моделирование — LDA и BERTopic

> LDA: документы — смеси topic-ов, topic-и — распределения по словам. BERTopic: документы группируются в пространстве эмбеддингов, кластеры и есть topic-и. Цель одна, разложения разные.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## The Problem

У вас 10 000 тикетов поддержки, 50 000 новостных статей или 200 000 твитов. Нужно понять, о чём эта коллекция, не читая её. Размеченных категорий нет. Вы даже не знаете, сколько категорий существует.

Тематическое моделирование отвечает на этот вопрос без учителя. Даёте корпус — получаете небольшой набор связных topic-ов и, для каждого документа, распределение по этим topic-ам.

Доминируют два семейства алгоритмов. LDA (2003) считает каждый документ смесью латентных topic-ов, а каждый topic — распределением по словам. Вывод байесовский. Метод до сих пор едет в прод там, где нужны смешанные принадлежности и объяснимые вероятности на уровне слов.

BERTopic (2020) кодирует документы через BERT, снижает размерность через UMAP, кластеризует через HDBSCAN и вытаскивает слова topic-а через class-based TF-IDF. Он выигрывает на коротких текстах, соцсетях и вообще везде, где смысловая близость важнее совпадения слов. Один документ получает один topic — для длинных текстов это ограничение.

Этот урок даёт интуицию по обоим и говорит, какой выбирать под какой корпус.

> 🎒 **На пальцах.** Представьте 10 000 тикетов. Прочитать их вручную по минуте на штуку — это 167 часов, месяц работы. Тематическая модель за пару минут скажет: «вот 12 topic-ов, самый большой — про оплату, в нём 2300 тикетов». Точность не идеальна, но месяц вы не потратили.

## The Concept

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA generative story.** Каждый topic — распределение по словам. Каждый документ — смесь topic-ов. Чтобы сгенерировать слово в документе, сначала берём topic из смеси документа, потом берём слово из распределения этого topic-а. Вывод разворачивает процесс назад: по наблюдаемым словам восстанавливаем распределение topic-ов в каждом документе и распределение слов в каждом topic-е. Математику делает collapsed Gibbs sampling или вариационный байес.

> 🎒 **На пальцах.** Это две урны. Сначала тянете из урны «о чём этот документ» и получаете, скажем, topic «спорт». Потом лезете в урну «слова спорта» и достаёте «матч». Повторяете для каждого слова текста. LDA видит только вытянутые слова и по ним угадывает, как были устроены обе урны.

Ключевой выход LDA:

- `doc_topic`: матрица `(n_docs, n_topics)`, каждая строка суммируется в 1 (смесь topic-ов документа).
- `topic_word`: матрица `(n_topics, vocab_size)`, каждая строка суммируется в 1 (распределение слов в topic-е).

> 🎒 **На пальцах.** При 5 topic-ах строка `doc_topic` выглядит как [0.60, 0.25, 0.10, 0.03, 0.02] — сумма ровно 1. Читается так: документ на 60% про первый topic и на 25% про второй, остальное шум. Одна статья может быть одновременно про политику и про экономику, и LDA это выражает числами, а не выбором «либо-либо».

**BERTopic pipeline.**

1. Закодировать каждый документ sentence transformer-ом (например, `all-MiniLM-L6-v2`). Векторы на 384 измерения.
2. Снизить размерность через UMAP примерно до 5 измерений. Эмбеддинги BERT слишком многомерны для кластеризации.
3. Кластеризовать через HDBSCAN. Кластеризация по плотности, кластеры получаются разного размера, плюс появляется метка «выброс».
4. Для каждого кластера посчитать class-based TF-IDF по его документам и вытащить топовые слова.

На выходе — один topic на документ (плюс метка -1 для выбросов). Опционально — мягкая принадлежность через вектор вероятностей HDBSCAN.

> 🎒 **На пальцах.** Шаг 2 выглядит как потеря информации: было 384 числа, стало 5. Так и есть, но это лечение, а не болезнь. В 384 измерениях любые две точки почти одинаково далеки друг от друга, и понятие «плотный кластер» теряет смысл. UMAP сжимает облако до 5 измерений так, чтобы близкие точки остались близкими, — и только после этого HDBSCAN видит группы.

```figure
topic-drift
```

## Build It

### Step 1: LDA via scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

> 🎒 **На пальцах.** Пройдитесь по фильтрам `CountVectorizer`. `min_df=2` выбрасывает слова, встретившиеся ровно в одном документе, — обычно это опечатки и случайные имена. `max_df=0.9` выбрасывает слова, которые есть больше чем в 90% документов: на 10 000 тикетов это слова из более чем 9000, вроде «order» в корпусе про заказы. `max_features=1000` оставляет тысячу самых частых из оставшихся. Меньше словарь — чище topic-и.

Обратите внимание: стоп-слова удалены, min_df и max_df отсекают редкие и вездесущие термины, а взят CountVectorizer, а не TfidfVectorizer, потому что LDA ждёт сырые счётчики.

> 🎒 **На пальцах.** Это частая ошибка. TF-IDF выдаёт дробные веса вроде 0.37, а генеративная история LDA требует целых «сколько раз слово вытянули из урны». Подсунете TF-IDF — код запустится и даже что-то напечатает, только topic-и будут хуже, а вы не поймёте почему.

### Step 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

> 🎒 **На пальцах.** `print(info.head(20))` покажет таблицу topic-ов по убыванию размера, и самой первой строкой почти всегда будет topic -1 — куча выбросов. На типичном корпусе туда попадает 10-30% документов. Строка с фильтром `Topic != -1` именно эту кучу и выкидывает, оставляя настоящие topic-и.

Фильтр `Topic != -1` отбрасывает корзину выбросов BERTopic (документы, которые HDBSCAN не смог кластеризовать). `min_topic_size` задаёт минимальный размер кластера для HDBSCAN; в библиотеке по умолчанию 10. Здесь стоит явное 15 под масштаб урока. Для корпусов больше 10 000 документов поднимайте до 50 или 100.

> 🎒 **На пальцах.** `min_topic_size` — единственная ручка, которая заметно меняет картину. Поставьте 10 на 50 000 документах и получите 400 мелких topic-ов, половина из которых дубликаты. Поставьте 100 — получите штук 30 крупных и читаемых. Правило простое: чем больше корпус, тем выше порог.

### Step 3: evaluation

Оба метода выдают слова topic-ов. Вопрос в том, связаны ли эти слова между собой.

- **Topic coherence (c_v).** Считает NPMI (нормированную поточечную взаимную информацию) для пар топовых слов по скользящим окнам контекста, собирает оценки в векторы topic-ов и сравнивает эти векторы по косинусу. Больше — лучше. Используйте `gensim.models.CoherenceModel` с `coherence="c_v"`.
- **Topic diversity.** Доля уникальных слов среди топовых слов всех topic-ов. Больше — лучше (topic-и не перекрываются).
- **Qualitative inspection.** Прочитайте топовые слова каждого topic-а. Называют ли они реальную вещь? Человеческое суждение остаётся последним рубежом обороны.

> 🎒 **На пальцах.** Diversity считается в одну строчку. 5 topic-ов по 10 слов — это 50 позиций; если уникальных слов среди них 40, diversity = 40/50 = 0.8. Упало до 0.4 — значит половина слов повторяется из topic-а в topic, и модель на самом деле нашла не пять тем, а две с половиной. А coherence при этом может остаться высокой, поэтому смотреть надо на обе метрики сразу.

## When to pick which

| Situation | Pick |
|-----------|------|
| Короткие тексты (твиты, отзывы, заголовки) | BERTopic |
| Длинные документы со смесью topic-ов | LDA |
| Нет GPU / мало вычислений | LDA или NMF |
| Нужны распределения по нескольким topic-ам на документ | LDA |
| Интеграция с LLM для называния topic-ов | BERTopic (поддержка из коробки) |
| Развёртывание на edge с ограниченными ресурсами | LDA |
| Максимальная смысловая coherence | BERTopic |

Главное практическое соображение — длина документа. Эмбеддинги BERT обрезают текст; счётчики LDA работают на любой длине. Если документы длиннее контекста модели эмбеддингов, либо нарезайте и агрегируйте, либо берите LDA.

> 🎒 **На пальцах.** Из семи строк таблицы пять голосуют за LDA и две за BERTopic — но это не значит, что LDA лучше. Читайте не итог, а свою строку. Твиты по 20 слов — берите BERTopic; научные статьи по 8000 слов, из которых `all-MiniLM-L6-v2` увидит первые 256 токенов, — берите LDA.

## Use It

Стек 2026 года:

- **BERTopic.** Дефолт для коротких текстов и всего, где важен смысл.
- **`gensim.models.LdaModel`.** Классический LDA для прода, зрелый и обкатанный.
- **`sklearn.decomposition.LatentDirichletAllocation`.** Простой LDA для экспериментов.
- **NMF.** Неотрицательное матричное разложение. Быстрая альтернатива LDA, сопоставимое качество на коротких текстах.
- **Top2Vec.** По устройству похож на BERTopic. Сообщество меньше, но на части бенчмарков хорош.
- **FASTopic.** Новее и быстрее BERTopic на очень больших корпусах.
- **LLM-based labeling.** Запустите любую кластеризацию, потом попросите модель назвать каждый кластер.

> 🎒 **На пальцах.** Последний пункт стоит отдельного внимания и стоит копейки. Модель выдаёт topic как список слов вроде «refund, charge, card, billing, invoice». Скормите эти 10 слов LLM с просьбой дать заголовок — получите «Проблемы с оплатой». Один вызов на topic: для 30 topic-ов это 30 запросов, секунды работы и читаемый отчёт вместо мешка слов.

## Ship It

Сохраните как `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

> 🎒 **На пальцах.** Проверьте формулу числа topic-ов на своих данных. 10 000 документов: sqrt(10000) = 100, значит рекомендация — 100 topic-ов. 400 документов: sqrt(400) = 20. 9 документов: sqrt(9) = 3, но `max(5, ...)` поднимает до 5. Это не закон природы, а разумная стартовая точка, от которой надо плясать вверх и вниз, глядя на coherence.

## Exercises

1. **Easy.** Обучите LDA с 5 topic-ами на датасете 20 Newsgroups. Напечатайте топ-10 слов каждого topic-а. Подпишите каждый topic вручную. Нашёл ли алгоритм настоящие категории?
2. **Medium.** Обучите BERTopic на том же подмножестве 20 Newsgroups. Сравните с LDA число найденных topic-ов, топовые слова и качественную связность. Что чище показывает настоящие категории?
3. **Hard.** Посчитайте coherence c_v для LDA и BERTopic на своём корпусе. Прогоните каждый метод с 5, 10, 20 и 50 topic-ами. Постройте график coherence от числа topic-ов. Отчитайтесь, какой метод стабильнее по числу topic-ов.

> 🎒 **На пальцах.** Подсказка к первому заданию: в 20 Newsgroups ровно 20 настоящих категорий, а вы просите 5 topic-ов. Значит совпадения один в один не будет по определению — LDA обязана склеить группы. Скорее всего склеятся близкие: comp.* соберутся в один «компьютерный» topic, rec.sport.* — в «спортивный». Это не провал алгоритма, это ваш выбор n_topics. Прогоните ещё раз с 20 и сравните.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | То, о чём корпус | Распределение вероятностей по словам (LDA) или кластер похожих документов (BERTopic). |
| Mixed membership | Документ сразу про несколько тем | LDA даёт каждому документу распределение по всем topic-ам. |
| UMAP | Снижение размерности | Обучение на многообразии с сохранением локальной структуры; используется в BERTopic. |
| HDBSCAN | Кластеризация по плотности | Находит кластеры разного размера; выбросам выдаёт метку «шум» (-1). |
| c_v coherence | Метрика качества topic-ов | NPMI для пар топовых слов по скользящим окнам, собранная в векторы topic-ов и сравниваемая по косинусу. (Просто усреднённая PMI — это c_uci; просто усреднённая NPMI — это c_npmi.) |

## Further Reading

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — статья про LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — статья про BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — статья, которая ввела c_v и родственные метрики.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — справочник для прода. Отличные примеры.
