<!-- i18n:manual -->
# Оценка long context — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro заявляет 10M токенов контекста. На 1M токенов 8-needle MRCR падает до 26.3%. Заявленное ≠ пригодное. Оценка long context показывает реальную ёмкость модели, на которой вы запускаете продукт.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## The Problem

У вас договор на 200 страниц. Модель заявляет context window на 1M токенов. Вы вставляете договор целиком и спрашиваете: «Какой тут пункт о расторжении?» Модель отвечает — но отвечает по титульному листу, потому что пункт о расторжении лежит на глубине 120k токенов, дальше того места, куда модель на самом деле смотрит.

Это разрыв между заявленной и настоящей ёмкостью контекста, каким он выглядит в 2026 году. В спецификации написано 1M или 10M. На практике пригодны 60-70% от этого, и слово «пригодны» зависит от задачи.

- **Retrieval (single needle in haystack):** на фронтирных моделях почти идеально вплоть до заявленного максимума.
- **Multi-hop / aggregation:** у большинства моделей резко проседает после ~128k.
- **Reasoning over dispersed facts:** ломается первым.

> 🎒 **На пальцах.** Заявленный 1M токенов — это как рекламный расход топлива «5 литров на сотню». По факту 60-70% от 1M — это 600-700k токенов на простой поиск, а на рассуждение остаётся 25-50%, то есть 250-500k. Договор на 200 страниц весит примерно 130k токенов: найти в нём один пункт модель сможет, а сравнить все пункты о штрафах — уже вряд ли.

Оценка long context измеряет эти оси. Урок называет бенчмарки, объясняет, что каждый из них реально меряет, и показывает, как собрать свой needle-тест под вашу предметную область.

> 🎒 **На пальцах.** Три оси из списка выше — как три уровня сложности в игре. Найти одну иголку легко. Найти три и сложить их вместе — труднее. Собрать вывод из фактов, разбросанных по всему тексту, — самый сложный уровень, и ломается он первым, задолго до заявленного миллиона.

## The Concept

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Кладём факт («the magic word is pineapple») на заданную глубину в длинном контексте. Просим модель его достать. Перебираем глубину × длину. Самый первый бенчмарк для long context. Фронтирные модели его уже насыщают: это необходимая, но недостаточная база.

> 🎒 **На пальцах.** Стог сена тут буквальный: берём длинный скучный текст-наполнитель и вставляем внутрь одну фразу про волшебное слово. Потом спрашиваем модель, что это за слово. Если перебрать 5 глубин × 4 длины, получится 20 прогонов — маленькая таблица, по которой сразу видно, где модель слепнет.

**RULER (Nvidia, 2024).** 13 типов задач в 4 категориях: retrieval (single / multi-key / multi-value), multi-hop tracing (отслеживание переменных), aggregation (частота самого частого слова), QA. Длина контекста настраивается, от 4k до 128k+. Вскрывает модели, которые насыщают NIAH, но валятся на multi-hop. В релизе 2024 года только половина из 17 моделей, заявлявших 32k+ контекста, удержала качество на 32k.

> 🎒 **На пальцах.** RULER — это экзамен из 13 заданий вместо одного. Цифра 2024 года отрезвляет: из 17 моделей, обещавших 32k, качество на 32k удержали только 8-9. Половина спецификаций не выдержала собственного обещания.

**LongBench v2 (2024).** 503 вопроса с выбором ответа, контексты от 8k до 2M слов, шесть категорий задач: single-doc QA, multi-doc QA, long in-context learning, long dialogue, code repo, long structured data. Продакшн-бенчмарк для реального поведения на длинных входах.

**MRCR (Multi-Round Coreference Resolution).** Разрешение кореференции через много turn и на большом масштабе. Варианты на 8, 24 и 100 needles. Показывает, сколько фактов модель удерживает одновременно, прежде чем внимание начинает разъезжаться.

> 🎒 **На пальцах.** MRCR — это про «а кто такой он?» в длинном разговоре. Разница между 8 и 100 needles как между списком покупок на 8 пунктов и на 100: первый вы держите в голове, второй уже нет. Из эпиграфа: 8 needles на 1M токенов дают 26.3%, то есть три ответа из четырёх неверны.

**NoLiMa.** «Нелексическая иголка». Needle и вопрос не пересекаются ни одним словом; чтобы найти нужное, нужен один шаг смыслового вывода. Сложнее, чем NIAH.

**HELMET.** Склеивает много документов и задаёт вопрос по одному из них. Проверяет избирательное внимание.

**BABILong.** Прячет цепочки рассуждений bAbI внутри постороннего стога. Проверяет рассуждение внутри стога, а не только retrieval.

> 🎒 **На пальцах.** Три бенчмарка — три разные подлости. NoLiMa убирает подсказку по словам, найти по совпадению букв уже нельзя. HELMET подсовывает кучу документов, а спрашивает про один. BABILong требует не найти, а подумать. Пройти NIAH и провалить все три — обычное дело.

### What to actually report

- **Advertised context window.** Число из спецификации.
- **Effective retrieval length.** Длина, на которой NIAH ещё проходит выбранный порог (например, 90%).
- **Effective reasoning length.** Длина, на которой тот же порог держат multi-hop или aggregation.
- **Degradation curve.** Точность в зависимости от длины контекста, отдельная кривая на каждый тип задачи.

Два числа для вашей внутренней спецификации: retrieval-effective и reasoning-effective. Обычно reasoning-effective составляет 25-50% от заявленного окна.

> 🎒 **На пальцах.** Две цифры вместо одной, как «максимальная скорость» и «крейсерская». Если модель заявляет 1M, а reasoning-effective равен 25-50%, честно писать в своей документации так: поиск до ~700k, рассуждение до ~250-500k. В архитектуру закладывайте вторую цифру.

```figure
gx-niah-decay
```

## Build It

### Step 1: a custom NIAH for your domain

Смотрите `code/main.py`. Скелет:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Перебирайте `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Постройте heatmap. Это и есть NIAH-карточка вашей целевой модели.

> 🎒 **На пальцах.** Функция `build_haystack` делает ровно то, что описано словами: режет наполнитель на две части и кладёт иголку между ними. При `depth_ratio = 0.25` и `total_tokens = 16000` иголка окажется примерно на 4000-м токене. При `0` — в самом начале, при `1.0` — в самом конце.

### Step 2: a multi-needle variant

```python
def build_multi_needle(filler_text, needles, total_tokens, depths=(0.1, 0.4, 0.7)):
    filler_tokens = tokenize(filler_text)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    needle_tokens = [tokenize(n) for n in needles]
    body_len = max(total_tokens - sum(len(n) for n in needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    out, cursor = [], 0
    for depth, needle in sorted(zip(depths, needle_tokens)):
        pos = min(int(body_len * depth), body_len)
        out.extend(filler_tokens[cursor:pos])
        out.extend(needle)
        cursor = pos
    out.extend(filler_tokens[cursor:])
    return " ".join(out)
```

Вопросы вроде «What are the three magic words?» требуют достать все три. Успех на одной иголке не предсказывает успех на трёх.

Следите за единицами измерения: `total_tokens` — это количество токенов, поэтому каждый срез здесь тоже токенный. Если резать *строку* наполнителя по `total_tokens`, вы отмеряете символы, и построенный стог окажется в несколько раз короче той длины, которую вы подписываете на оси X своего heatmap. Единственный курсор, проходящий по наполнителю, вдобавок гарантирует, что окна глубин выкладывают документ встык, а не перекрывают друг друга.

> 🎒 **На пальцах.** Три иголки на глубинах 0.1, 0.4 и 0.7 — это как попросить запомнить три числа из разных концов длинной лекции. Модель, которая находит одну иголку в 95% случаев, на трёх легко даёт 60%: вероятности не складываются, а перемножаются. 0.95³ ≈ 0.86, и это ещё оптимистичная оценка.

> 🎒 **На пальцах.** Функция построена по тому же рецепту, что `build_haystack` в Step 1, только точек вставки не одна, а три. Сначала считаем, сколько токенов останется на наполнитель: `body_len = total_tokens - сумма длин иголок`. Потом повторяем наполнитель, пока его не хватит, и обрезаем ровно до `body_len`. Дальше идём по глубинам от меньшей к большей (за это отвечает `sorted`) и на каждой доливаем наполнитель от `cursor` до `pos`, а затем кладём иголку. Последняя строка `out.extend(filler_tokens[cursor:])` доливает хвост после самой глубокой иголки. Итоговая длина получается ровно `total_tokens` — а именно это число вы подписываете на графике.

> 🎒 **На пальцах.** Почему нельзя было обойтись срезом строки вроде `filler[:int(total_tokens * 0.1)]`. Потому что `total_tokens` — токены, а срез строки в Python отсчитывает символы. Один токен в среднем весит примерно 4 символа, значит на `total_tokens = 64000` такой срез отмерит 6 400 символов — это около 1 600 токенов, а не 6 400. Вы напишете в отчёте «на 64k модель нашла все три иголки», хотя реально проверили примерно 16k, то есть в четыре раза меньше. Ошибка в единицах измерения не роняет программу — она молча делает весь замер неправдой.

### Step 3: multi-hop variable tracing (RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Чтобы ответить, нужно связать три присваивания подряд. Фронтирные модели на 128k здесь часто падают до 50-70% точности.

> 🎒 **На пальцах.** В примере спрятана цепочка: X1 = 42, значит X2 = 42 + 10 = 52, значит X3 = 52 × 2 = 104. Человек считает это в уме за три секунды. Модель на 128k ошибается в трети-половине случаев, потому что должна удержать три строчки, разбросанные по сотне тысяч токенов.

### Step 4: LongBench v2 on your stack

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Отчитывайтесь по каждой категории отдельно. Общая усреднённая цифра прячет большую разницу между задачами.

> 🎒 **На пальцах.** Средняя оценка обманывает так же, как средняя температура по больнице. Модель может дать 85% на single-doc QA и 45% на multi-doc QA, а в отчёте вы увидите красивые 65% — и не узнаете, что ровно ваш сценарий провален.

## Pitfalls

- **NIAH-only evaluation.** Пройденный NIAH на 1M токенов не говорит ничего про multi-hop. Всегда запускайте RULER или свой тест на несколько шагов.
- **Uniform depth sampling.** Многие реализации проверяют только глубину 0.5. Проверяйте 0, 0.25, 0.5, 0.75, 1.0 — эффект «lost in the middle» вполне реален.
- **Lexical overlap with filler.** Если иголка делит слова с наполнителем, поиск становится тривиальным. Берите needle без общих слов, в духе NoLiMa.
- **Ignoring latency.** Промпт на 1M токенов обсчитывается 30-120 секунд до первого токена. Меряйте time-to-first-token вместе с точностью.
- **Vendor-self-reported numbers.** OpenAI, Google и Anthropic публикуют собственные оценки. Всегда перепроверяйте независимо и на своём сценарии.

> 🎒 **На пальцах.** Самая дорогая из пяти ловушек — предпоследняя. Промпт на 1M токенов ждёт первого токена 30-120 секунд: в чате пользователь уйдёт раньше, чем модель начнёт отвечать. А красивая цифра из блога вендора не заменит прогон на ваших собственных документах.

## Use It

Стек 2026 года:

| Situation | Benchmark |
|-----------|-----------|
| Быстрая проверка на вменяемость | Свой NIAH на 3 глубинах × 3 длинах |
| Выбор модели для продакшена | RULER (13 задач) на вашей целевой длине |
| Качество QA на реальных данных | Подмножество single-doc-QA из LongBench v2 |
| Рассуждение в несколько шагов | BABILong или своя трассировка переменных |
| Диалоговые сценарии | MRCR на 8 needles на вашей целевой длине |
| Регрессия при обновлении модели | Зафиксированный внутренний стенд NIAH + RULER, прогон на каждой новой модели |

Правило для продакшена: не верьте заявленному context window, пока не прогнали NIAH плюс хотя бы одну задачу на рассуждение на нужной вам длине.

> 🎒 **На пальцах.** Таблица читается как «какой инструмент под какую задачу». Быстрая проверка — это 3 × 3 = 9 прогонов, то есть минуты. Выбор модели в прод — это RULER с его 13 задачами, то есть часы. Дешёвую проверку делайте всегда, дорогую — перед тем как подписаться на модель.

## Ship It

Сохраните как `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

> 🎒 **На пальцах.** Обратите внимание на два порога в готовом промпте: 90% для retrieval и 70% для reasoning. Планка для рассуждения ниже не по доброте, а потому что 70% на реальных многошаговых задачах — уже хороший результат. Требовать 90% значит не найти ни одной подходящей модели.

## Exercises

1. **Easy.** Соберите NIAH на 3 глубинах (0.25, 0.5, 0.75) × 3 длинах (1k, 4k, 16k). Прогоните на любой модели. Нарисуйте долю успехов как heatmap 3×3.
2. **Medium.** Добавьте вариант с тремя иголками. Померьте, как часто находятся все три на каждой длине. Сравните с долей успехов на одной иголке при той же длине.
3. **Hard.** Постройте задачу трассировки переменных (X1 → X2 → X3, три шага) внутри 64k наполнителя. Померьте точность на трёх фронтирных моделях. Отчитайтесь по effective reasoning length для каждой.

> 🎒 **На пальцах.** Подсказка к третьему заданию: сначала прогоните трассировку на 1k наполнителя. Если модель ошибается уже там, дело не в длине контекста, а в самой задаче, и мерить 64k бессмысленно. Убедитесь, что на коротком входе точность близка к 100%, и только потом растягивайте.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| NIAH | «Needle in haystack» | Прячем факт в наполнителе и просим модель его достать. |
| RULER | «NIAH на стероидах» | 13 типов задач по категориям retrieval / multi-hop / aggregation / QA. |
| Effective context | «Настоящая ёмкость» | Длина, на которой точность ещё держится выше порога. |
| Lost in the middle | «Перекос по глубине» | Модель хуже смотрит на то, что лежит в середине длинного входа. |
| Multi-needle | «Много фактов сразу» | Несколько закладок одновременно; проверяет жонглирование вниманием, а не только retrieval. |
| MRCR | «Многоходовая кореференция» | Кореференция на 8, 24 или 100 needles; вскрывает насыщение внимания. |
| NoLiMa | «Needle без общих слов» | Needle и вопрос не имеют общих токенов; нужен смысловой вывод. |

## Further Reading

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — исходный репозиторий NIAH.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — бенчмарк на много задач сразу.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — оценка long context на реальных данных.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — иголки посложнее.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — рассуждение внутри стога.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — статья про перекос по глубине.
