<!-- i18n:manual -->
# Стратегии chunking для RAG

> Настройки chunking влияют на качество retrieval не меньше, чем выбор embedding-модели (Vectara, NAACL 2025). Ошибётесь с chunking — и никакой реранкер вас не спасёт.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## The Problem

Вы загрузили в RAG-систему договор на 50 страниц. Пользователь спрашивает: «Какой пункт о расторжении?» Ретривер возвращает титульный лист. Почему? Потому что модель обучалась на чанках по 512 токенов, а пункт о расторжении лежит на 20-й странице, разорван переносом страницы, и рядом нет ни одного слова из запроса.

Лечится это не покупкой более дорогой embedding-модели. Лечится это chunking. Какого размера? С overlap? Где резать? С окружающим контекстом?

Бенчмарки февраля 2026-го дают неожиданные результаты:

- Исследование Vectara 2026 года: рекурсивный chunking по 512 токенов обошёл семантический — 69% точности против 54%.
- SPLADE + Mistral-8B на Natural Questions: overlap не дал вообще никакого измеримого выигрыша.
- Обрыв контекста: качество ответа резко падает примерно на 2 500 токенах контекста.

«Очевидный» ответ (семантический chunking, 20% overlap, 1 000 токенов) чаще всего неверен. Этот урок даёт интуицию по шести стратегиям и говорит, когда какую брать.

> 🎒 **На пальцах.** Разрыв 69% против 54% — это 15 верных ответов из ста, потерянных на одной настройке нарезки. Ни один реранкер такого не вернёт: если нужный кусок текста вообще не попал в индекс целиком, доставать нечего. Сравните с поиском по книге, где страницы вырваны в случайных местах.

## The Concept

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.** Резать каждые N символов или токенов. Простейшая база. Рвёт предложения посередине. Хорошее сжатие, плохая связность.

**Recursive.** `RecursiveCharacterTextSplitter` из LangChain. Сначала пробуем резать по `\n\n`, потом по `\n`, потом по `.`, потом по пробелу. Аккуратно откатывается на следующий разделитель. Выбор по умолчанию в 2026-м.

> 🎒 **На пальцах.** Рекурсивная стратегия ведёт себя как человек, который режет текст ножницами: сначала пытается по пустой строке между абзацами, не вышло — по концу строки, не вышло — по точке, и только в самом отчаянном случае по пробелу посреди фразы. Fixed режет ровно на 512-м символе, даже если это середина слова «расторж|ение».

**Semantic.** Считаем embedding каждого предложения. Считаем косинусную близость соседних предложений. Режем там, где близость падает ниже порога. Сохраняет тематическую связность. Медленнее; иногда выдаёт крошечные обрывки на 40 токенов, которые портят retrieval.

> 🎒 **На пальцах.** Это как резать фильм по смене сцен, а не каждые 10 минут. Звучит идеально — но на практике две подряд идущие фразы «Спасибо.» и «До свидания.» дадут низкую близость, и вы получите чанк из двух слов. Такой обрывок в индексе бесполезен: он совпадёт с чем угодно вежливым и ни с чем полезным.

**Sentence.** Резать по границам предложений. Одно предложение на чанк или окно из N предложений. До примерно 5 000 токенов не уступает семантическому chunking, а стоит в разы дешевле.

**Parent-document.** Храним маленькие дочерние чанки для поиска *и* больший родительский чанк для контекста. Ищем по дочернему, возвращаем родительский. Деградирует мягко: даже плохой дочерний чанк вернёт вменяемого родителя.

> 🎒 **На пальцах.** Как оглавление и глава. По оглавлению («пункт 7.3, расторжение») быстро находите, а читать отдаёте всю главу целиком. Типичные размеры — дочерний 256 токенов, родительский 2 048: ищете по короткой точной строке, а модель получает восемь раз больше контекста вокруг неё.

**Late chunking (2024).** Сначала считаем embedding всего документа на уровне токенов, потом объединяем токенные embedding в чанковые через pooling. Сохраняет контекст между чанками. Работает с длинноконтекстными моделями (BGE-M3, Jina v3). Дороже по вычислениям.

> 🎒 **На пальцах.** Обычный chunking — это когда каждый кусок текста читает отдельный человек, который остальной документ не видел. Late chunking — когда один человек прочёл всё целиком, а потом пересказал каждый кусок. Слово «компания» в 30-м чанке при late chunking уже «знает», что речь про Apple из первого абзаца.

**Contextual retrieval (Anthropic, 2024).** К каждому чанку приписываем сгенерированную LLM справку о его месте в документе («Этот фрагмент — раздел 3.2 из пунктов о расторжении...»). В собственном бенчмарке Anthropic это дало улучшение retrieval на 35-50%. Дорого при индексации.

> 🎒 **На пальцах.** Считайте цену: один вызов LLM на каждый чанк. Документ на 50 страниц даёт примерно 200 чанков — значит 200 вызовов только чтобы один раз проиндексировать один документ. Зато прирост retrieval в 35-50% — это самое большое улучшение из всего списка. Дорого один раз при индексации, бесплатно на каждом запросе.

### The rule that beats every default

Подбирайте размер чанка под тип запроса:

| Query type | Chunk size |
|------------|-----------|
| Factoid ("what is the CEO's name?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

Это бенчмарк NVIDIA 2026 года. Чанк должен быть достаточно большим, чтобы вместить ответ и локальный контекст, и достаточно маленьким, чтобы в топ-K ретривера попадал ответ, а не шум вокруг него.

> 🎒 **На пальцах.** Логика простая: чем сложнее вопрос, тем больше текста нужно рядом. «Как зовут гендиректора?» — ответ умещается в одном предложении, хватит 256 токенов. «Чем стратегия компании в 2025-м отличалась от 2024-го?» — нужно два абзаца минимум, отсюда 1 024. Восьмикратная разница в размере из-за одного лишь типа вопроса.

```figure
n5-chunk-cuts
```

## Build It

### Step 1: fixed and recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(seps):
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                # seps[i + 1:], а не seps[1:] — рекурсия должна идти по ещё не
                # опробованным разделителям, иначе мы снова предлагаем отвергнутые.
                chunks.extend(chunk_recursive(p, size=size, seps=seps[i + 1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

> 🎒 **На пальцах.** Обратите внимание на строку `step = size - overlap`. При `size=512, overlap=0` шаг равен 512 — чанки идут встык. При `overlap=100` шаг 412, и на текст в 10 000 символов вы получите 25 чанков вместо 20. Это на 25% больше векторов в индексе и на 25% больше денег за хранение — ради выигрыша, который бенчмарки 2026-го найти не смогли.

> 🎒 **На пальцах.** Второй важный момент — `seps[i + 1:]` в рекурсивном вызове. Смысл рекурсивной стратегии в том, чтобы спускаться по списку разделителей сверху вниз: не вышло по `\n\n` — пробуем `\n`, потом `. `, потом пробел. Если написать `seps[1:]`, спуск ломается: на разделителе `. ` (индекс 2) рекурсия получит `("\n", ". ", " ")` и снова предложит `\n`, который на этом фрагменте уже не сработал. Именно поэтому в цикле стоит `enumerate` — нужен номер текущего разделителя, чтобы отрезать список ровно после него.

### Step 2: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    # Цикл проверяет длину только у чанка, который закрывает, и никогда — у того,
    # который открывает: значит последний чанк может оказаться короче min_chars.
    # Подклеиваем такой огрызок обратно к предыдущему.
    if len(chunks) > 1 and sum(len(s) for s in chunks[-1]) < min_chars:
        chunks[-2].extend(chunks.pop())

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Подбирайте `threshold` на своём домене. Слишком высокий — обрывки. Слишком низкий — один гигантский чанк.

> 🎒 **На пальцах.** Смотрите на условие `if sim < threshold and current_len >= min_chars`. Вторая половина — тот самый предохранитель от обрывков: даже если близость упала до 0.2, резать нельзя, пока в текущем чанке меньше 200 символов. Уберите `min_chars` — и на диалоге получите чанки по одному слову.

> 🎒 **На пальцах.** Но у этого предохранителя есть дырка, и она ровно в том месте, которое урок сам назвал главной ловушкой. `current_len` считает длину чанка, который мы *закрываем*. За тем, что получилось в чанке, который мы *открыли*, никто больше не следит — цикл просто кончился. Разрежьте по смыслу текст, у которого последняя фраза — «Спасибо, до свидания.», и в индекс уедет чанк на 21 символ, хотя `min_chars=200`. Поэтому после цикла стоят три строки, которые смотрят на хвост и, если он не дотянул до `min_chars`, подклеивают его к предыдущему чанку: `chunks[-2].extend(chunks.pop())`. Проверка `len(chunks) > 1` нужна на случай, когда чанк всего один — приклеивать его некуда, и короткий документ имеет право остаться коротким.

### Step 3: parent-document

```python
import numpy as np


def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, child_top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:child_top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Ключевая деталь: родителей надо дедуплицировать. Несколько детей могут указывать на одного родителя, и вернуть его несколько раз — значит впустую потратить контекст. Обратите внимание, что именно ограничивает отсечка: `child_top_k` ограничивает число оцениваемых *детей*, поэтому родителей вы получите не больше этого числа, а часто меньше. Если нужно гарантированное количество родителей, расширяйте пул детей и останавливайтесь, когда набор родителей заполнен.

> 🎒 **На пальцах.** Множество `seen` тут делает всю работу. При `child_top_k=3` три лучших дочерних чанка вполне могут лежать в одном абзаце — без дедупликации вы отправите модели один и тот же родитель три раза и займёте 6 144 токена вместо 2 048. Модель получит одну мысль трижды и ноль новой информации.

> 🎒 **На пальцах.** Отсюда и переименование параметра из `top_k` в `child_top_k`: имя `top_k` обманывает. Обычно «top_k=3» читается как «мне вернут три штуки», а тут три — это сколько детей мы посмотрели. Если все три ребёнка оказались из одного родителя, на выходе будет один чанк, а не три. Хотите три родителя — берите `child_top_k=10` или `20` и останавливайтесь, когда трёх набрали.

### Step 4: contextual retrieval (Anthropic pattern)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Индексируйте контекстуализированные чанки. На запросе retrieval выигрывает от дополнительного окружающего сигнала.

> 🎒 **На пальцах.** Промпт просит 50-100 слов справки. Приписав их к чанку в 256 токенов, вы увеличиваете его примерно в полтора раза — но эти лишние слова содержат имена и заголовки разделов, которых в самом чанке нет. Именно поэтому запрос «пункт о расторжении» вдруг начинает находить абзац, где слова «расторжение» вообще не было.

### Step 5: evaluate

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Всегда меряйте. «Лучшая» стратегия для вашего корпуса может не совпасть ни с одним постом в блоге.

> 🎒 **На пальцах.** Метрика тут честная и грубая: попал ли хоть один правильный чанк в топ-5. Если из 50 запросов попало 40, recall@5 = 0.8. Между рекурсивным и семантическим chunking разница обычно в 3-8 запросах из 50 — и без этого замера вы просто гадаете.

## Pitfalls

- **Chunking evaluated only on factoid queries.** На multi-hop запросах победители оказываются совсем другими. Используйте набор для оценки, расслоённый по типам запросов.
- **Semantic chunking without a minimum size.** Порождает обрывки на 40 токенов, которые портят retrieval. Всегда задавайте `min_tokens`.
- **Overlap as cargo cult.** Исследования 2026 года находят, что overlap часто не даёт ничего и при этом удваивает стоимость индекса. Меряйте, а не предполагайте.
- **No min/max enforcement.** И чанк из 5 токенов, и чанк из 5 000 одинаково ломают retrieval. Ограничивайте с обеих сторон.
- **Cross-doc chunking.** Никогда не позволяйте чанку захватывать два документа. Всегда режьте по документу, потом объединяйте.

> 🎒 **На пальцах.** Чанк, склеенный из хвоста одного документа и головы другого, — самая коварная из этих ошибок. Он выглядит нормальным текстом, embedding считается без ошибок, поиск его находит, а модель уверенно отвечает, смешав два разных договора. Ошибок в логах не будет ни одной.

## Use It

Стек 2026 года:

| Situation | Strategy |
|-----------|----------|
| First build, unknown corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| Heavy cross-reference (contracts, papers) | Late chunking or contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| Short utterances (tweets, reviews) | One document = one chunk |

Начинайте с рекурсивного chunking по 512. Померьте recall@5 на наборе из 50 запросов. Дальше подкручивайте.

> 🎒 **На пальцах.** Последняя строка таблицы — самая недооценённая. Твит длиной 200 символов резать не надо вообще: один документ — один чанк. Всякий раз, когда документ короче целевого размера чанка, chunking просто не нужен, и любая «стратегия» тут только добавит багов.

## Ship It

Сохраните как `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Exercises

1. **Easy.** Нарежьте один документ на 20 страниц тремя способами: fixed(512, 0), recursive(512, 0) и recursive(512, 100). Сравните количество чанков и качество границ.
2. **Medium.** Соберите набор из 30 запросов по 5 документам. Померьте recall@5 для рекурсивного, семантического и parent-document подходов. Кто выиграл? Совпало ли с тем, что пишут в блогах?
3. **Hard.** Реализуйте contextual retrieval. Померьте прирост MRR относительно базового рекурсивного chunking. Отчитайтесь по стоимости индексации (число вызовов LLM) против прироста точности.

> 🎒 **На пальцах.** Подсказка к первому заданию: 20 страниц — это примерно 50 000 символов. При size=512 и overlap=0 ждите около 100 чанков, при overlap=100 — около 122. Смотрите не только на числа: откройте первые пять чанков от fixed и от recursive и сравните, сколько предложений разорвано посередине.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Chunk | «Кусок документа» | Подъединица документа, которую считают в embedding, индексируют и достают при поиске. |
| Overlap | «Запас прочности» | N токенов, общих для соседних чанков; в бенчмарках 2026 года часто бесполезен. |
| Semantic chunking | «Умная нарезка» | Резать там, где падает близость embedding соседних предложений. |
| Parent-document | «Двухуровневый retrieval» | Искать по маленьким детям, возвращать больших родителей. |
| Late chunking | «Нарезка после embedding» | Считать embedding всего документа по токенам, потом собирать в чанковые векторы. |
| Contextual retrieval | «Трюк от Anthropic» | Сгенерированная LLM справка, приписанная к каждому чанку перед индексацией. |
| Context cliff | «Стена на 2 500 токенах» | Падение качества примерно на 2,5 тыс. токенов контекста в RAG (январь 2026). |

## Further Reading

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — вариант по умолчанию в продакшене.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking важен не меньше выбора embedding.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — статья про late chunking.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — улучшение retrieval на 35-50% за счёт сгенерированных LLM префиксов.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — размер чанка в зависимости от типа запроса.
