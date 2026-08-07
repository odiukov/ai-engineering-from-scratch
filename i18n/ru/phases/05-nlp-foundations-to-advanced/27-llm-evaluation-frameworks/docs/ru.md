<!-- i18n:manual -->
# Оценка LLM — RAGAS, DeepEval, G-Eval

> Exact-match и F1 не видят смысловое совпадение. Ручная проверка не масштабируется. LLM-as-judge — рабочий ответ для продакшена, если откалибровать его настолько, чтобы числу можно было верить.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## The Problem

Ваша RAG-система отвечает: "June 29th, 2007."
Эталонный ответ: "June 29, 2007."
Exact Match даёт 0. F1 даёт около 75%. Человек поставил бы 100%.

Теперь умножьте это на 10 000 тестовых случаев. Умножьте ещё раз — на каждое изменение ретривера, нарезки, промпта или модели. Вам нужен оценщик, который понимает смысл, дёшево работает в масштабе, не врёт про регрессии и показывает нужные виды поломок.

В 2026 году эту задачу закрывают три фреймворка.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Четыре метрики для RAG (faithfulness, answer-relevance, context-precision, context-recall) на бэкендах NLI и LLM-judge. С исследовательской базой, лёгкий.
- **DeepEval.** Pytest для LLM. Метрики G-Eval, выполнения задачи, галлюцинаций, предвзятости. Родной для CI/CD.
- **G-Eval.** Метод (и метрика внутри DeepEval): LLM-as-judge с цепочкой рассуждений, своими критериями и оценкой от 0 до 1.

Все три опираются на LLM-as-judge. Этот урок даёт интуицию про сам метод и про слой доверия вокруг него.

> 🎒 **На пальцах.** «June 29th» и «June 29» отличаются двумя буквами, но Exact Match ставит ноль. Одна такая мелочь на 10 000 тестов — и вы видите падение качества там, где ничего не сломалось. Судья-человек не ошибётся, но 10 000 ответов по 15 секунд — это больше 40 часов работы на один прогон.

## The Concept

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** Заменяем статическую метрику на LLM, которая оценивает выходы по заданной рубрике. На вход `(query, context, answer)`, промпт судье: «поставь от 0 до 1 за faithfulness». Возвращаем оценку.

Почему это работает: LLM приближают человеческое суждение за крошечную долю стоимости. GPT-4o-mini при цене около $0.003 за оценённый случай позволяет прогонять регрессионную оценку на 1000 примеров меньше чем за $5.

> 🎒 **На пальцах.** Посчитайте сами: 1000 × $0.003 = $3. Один прогон стоит как чашка кофе, поэтому его можно ставить на каждый pull request. Человек за те же 1000 ответов возьмёт несколько дней. Вот и вся причина, по которой LLM-as-judge победил.

Почему это тихо ломается:

1. **Judge bias.** Судьи предпочитают длинные ответы, ответы моделей своего же семейства и ответы в стиле самого промпта.
2. **JSON parsing failures.** Битый JSON → оценка NaN → случай молча выпадает из среднего. Пользователи RAGAS знают эту боль. Ставьте try/except и явный режим отказа.
3. **Drift over model versions.** Обновили судью — поехали все метрики. Фиксируйте модель судьи вместе с версией.

> 🎒 **На пальцах.** Самое коварное здесь второе. Если из 1000 случаев 50 вернули битый JSON и молча выпали, среднее считается по 950 — и как раз выпасть могли самые сложные примеры. Ваш «0.85» окажется завышенным, а вы даже не узнаете. Всегда логируйте, сколько случаев не распарсилось.

**The RAG four.**

| Metric | Question | Backend |
|--------|----------|---------|
| Faithfulness | Взят ли каждый факт из ответа именно из найденного контекста? | Проверка следования через NLI |
| Answer relevance | Отвечает ли ответ на заданный вопрос? | Сгенерировать по ответу гипотетические вопросы и сравнить с настоящим |
| Context precision | Какая доля найденных фрагментов оказалась релевантной? | LLM-judge |
| Context recall | Нашёл ли ретривер всё нужное? | LLM-judge против эталонного ответа |

> 🎒 **На пальцах.** Четыре метрики делятся на две пары. Faithfulness и answer relevance судят генератор: не выдумал ли он и по делу ли ответил. Context precision и recall судят ретривер: не принёс ли лишнего и не потерял ли нужное. Если faithfulness низкий, а context recall высокий — виноват генератор, ретривер своё дело сделал.

**G-Eval.** Задайте свой критерий: «сослался ли ответ на правильный источник?». Фреймворк сам разворачивает его в шаги оценки с рассуждением и выставляет оценку от 0 до 1. Хорошо закрывает доменные измерения качества, которых нет в RAGAS.

**Calibration.** Никогда не верьте сырой оценке судьи, пока не сопоставили её с человеческой разметкой. Разметьте руками 100 примеров. Постройте график «судья против человека». Посчитайте ро Спирмена. Если ро меньше 0.7, рубрику судьи надо дорабатывать.

> 🎒 **На пальцах.** Ро Спирмена — это «насколько судья и человек одинаково расставляют примеры от худшего к лучшему». 1.0 — идеальное совпадение порядка, 0 — судья тасует случайно. Порог 0.7 не священный, но ниже него разница между 0.82 и 0.79 в отчёте уже ничего не значит. Сотня размеченных примеров — это примерно два часа работы, и они окупаются сразу.

```figure
n5-judge-gauge
```

## Build It

### Step 1: faithfulness with NLI (RAGAS-style)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    # Отфильтровываем пустые строки: подтвердить их нельзя, а в знаменатель
    # `faithfulness` каждая попадёт — и молча потянет метрику вниз.
    return [line.strip() for line in llm(prompt).splitlines() if line.strip()]


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Разбираем ответ на атомарные утверждения. Каждое проверяем через NLI против найденного контекста. Faithfulness — доля подтверждённых.

> 🎒 **На пальцах.** Ответ «Первый iPhone вышел 29 июня 2007 года, его представил Стив Джобс» распадается на два утверждения. Если контекст подтверждает только дату, faithfulness = 1/2 = 0.5. Порог `entail["score"] > 0.5` означает: модель NLI должна быть уверена хотя бы наполовину, иначе утверждение считается неподтверждённым.

> 🎒 **На пальцах.** Теперь посмотрите, зачем в `atomic_claims` стоит `if line.strip()`. LLM почти всегда отвечает списком через пустую строку между пунктами, и без фильтра `splitlines()` вернёт не два утверждения, а четыре: два настоящих и два пустых. Пустую строку NLI подтвердить не может никогда, а в знаменатель `supported / len(claims)` она встанет как полноценное утверждение. Вместо честного 1/2 = 0.5 вы получите 1/4 = 0.25 — метрика упала вдвое из-за форматирования ответа, а не из-за галлюцинации. Такие поломки самые дорогие: цифра выглядит правдоподобно, и искать вы будете в ретривере.

### Step 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Если по ответу восстанавливаются не те вопросы, что были заданы, релевантность падает.

> 🎒 **На пальцах.** Приём обратный привычному: не сравниваем ответ с вопросом напрямую, а просим модель придумать по ответу 3 вопроса и смотрим, похожи ли они на настоящий. Если ответ был про погоду, а вопрос про курс валют, косинусы получатся около 0.1, среднее из трёх — тоже около 0.1. Число `n = 3` — компромисс: одного вопроса мало для устойчивости, десять стоят дороже.

### Step 3: G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Шаги оценки и есть рубрика. Явно прописанные шаги устойчивее, чем неявное «поставь оценку от 0 до 1».

> 🎒 **На пальцах.** Здесь пять шагов вместо одной расплывчатой фразы. Разница как между «оцени сочинение» и «проверь орфографию, потом пунктуацию, потом структуру». В нашем примере ответ "June 29th, 2007." против эталона "June 29, 2007." даст одно фактическое утверждение, подтверждённое полностью — оценка 1.0, а не 0, как у Exact Match.

### Step 3b: gate the judge's JSON

Любой судья, который отдаёт JSON, однажды отдаст не-JSON. Разбирайте ответ защитно и считайте отказы, а не давайте им растворяться:

```python
import json
import math


def judge_score(prompt: str, llm: LLM) -> float | None:
    """Return None — never NaN — when the judge does not produce parseable JSON."""
    raw = llm(prompt)
    try:
        score = float(json.loads(raw)["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def aggregate(scores: list[float | None]) -> tuple[float, int]:
    ok = [s for s in scores if s is not None]
    mean = sum(ok) / len(ok) if ok else 0.0
    return mean, len(scores) - len(ok)
```

Число нераспарсенных случаев докладывайте рядом со средним, всегда. NaN, попавший в `sum() / len()`, тихо портит агрегат; NaN, выброшенный `numpy.nanmean`, тихо уменьшает выборку. «0.85 на 1000 случаях» и «0.85 на тех 700, что распарсились» — это два разных утверждения, и сигналом о регрессии является только одно из них.

> 🎒 **На пальцах.** Вот тот самый гейтинг, про который в разделе Concept сказано «ставьте try/except и явный режим отказа». Ключевое решение здесь — возвращать `None`, а не `float("nan")`. `NaN` коварен тем, что притворяется числом: он пройдёт любую проверку типа, сложится с другими оценками и превратит всё среднее в `nan` — либо, если вы взяли `numpy.nanmean`, бесшумно выпадет из выборки. `None` так не умеет: его нельзя сложить, поэтому вы обязаны решить, что с ним делать, ещё на этапе написания кода. `aggregate` возвращает пару: среднее по распарсенным и количество отказов. Если из 1000 случаев 300 не распарсились, вы увидите `(0.85, 300)` — и сразу поймёте, что верить этому 0.85 нельзя.

> 🎒 **На пальцах.** Список в `except` выглядит длинным, но каждый пункт — реальный способ, которым судья ломает разбор. `json.JSONDecodeError` — судья написал «Score: 0.9» вместо JSON или обернул ответ в markdown-блок с тройными обратными кавычками. `KeyError` — JSON валидный, но ключ называется `rating`, а не `score`. `TypeError` — вернулся список, а не объект, и по строковому ключу его не проиндексировать. `ValueError` — в поле лежит `"high"`, и `float()` на этом падает. Последняя строка добивает `inf` и `nan`, которые `float()` разбирает молча: `float("Infinity")` — совершенно законный вызов.

### Step 4: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Оформляйте как pytest-файл. Запускайте на каждом pull request. Блокируйте слияние при регрессиях.

> 🎒 **На пальцах.** Пороги 0.85 и 0.7 — это черта, ниже которой merge не проходит. Заметьте: порог для faithfulness выше, потому что выдуманный факт в ответе хуже, чем немного лишний фрагмент в контексте. Числа берите не с потолка, а из текущего замера: измерили 0.88 — ставьте 0.85 и не давайте качеству сползать.

### Step 5: toy eval from scratch

Смотрите `code/main.py`. Там приближения faithfulness (пересечение утверждений ответа с контекстом) и релевантности (пересечение токенов ответа с токенами вопроса) только на стандартной библиотеке. Не для продакшена. Показывает форму решения.

> 🎒 **На пальцах.** Версия на пересечении слов не поймёт, что «29 июня» и «двадцать девятое июня» — одно и то же. Зато она работает без единого сетевого вызова и стоит ноль. Полезна для отладки самого каркаса: сначала убедитесь, что цифры вообще текут по трубе, а потом подключайте настоящего судью.

## Pitfalls

- **No calibration.** Судья с корреляцией 0.3 к человеческой разметке — это шум. Требуйте калибровочный прогон до выката.
- **Self-evaluation.** Если одна и та же LLM и генерирует, и судит, оценки завышаются на 10-20%. Судью берите из другого семейства моделей.
- **Positional bias in pairwise judging.** При попарном сравнении судьи предпочитают первый вариант. Всегда перемешивайте порядок и прогоняйте оба.
- **Raw aggregate hides failures.** Среднее 0.85 часто прячет 5% катастрофических провалов. Всегда смотрите нижний квантиль.
- **Golden dataset rot.** Неверсионированные наборы для оценки со временем расползаются и ломают сравнение по времени. Помечайте набор тегом при каждом изменении.
- **LLM cost.** В масштабе вызовы судьи съедают основную часть бюджета. Берите самую дешёвую модель, проходящую порог калибровки. GPT-4o-mini, Claude Haiku, Mistral-small.

> 🎒 **На пальцах.** Про среднее: пусть 950 случаев дали 0.87, а 50 — полный ноль. Среднее выходит 0.826 — вроде бы приличная цифра. Но эти 50 нулей и есть те ответы, из-за которых пользователь уйдёт. Смотрите худшие 10%, а не среднее.

## Use It

Стек 2026 года:

| Use case | Framework |
|---------|-----------|
| RAG quality monitoring | RAGAS (4 метрики) |
| CI/CD regression gates | DeepEval плюс pytest |
| Custom domain criteria | G-Eval внутри DeepEval |
| Online live-traffic monitoring | RAGAS в режиме без эталонов |
| Human-in-the-loop spot checks | LangSmith или Phoenix с интерфейсом разметки |
| Red-teaming / safety eval | Promptfoo плюс DeepEval |

Типичный набор: RAGAS для мониторинга, DeepEval для CI, G-Eval для новых измерений. Запускайте все три — их расхождения полезны.

> 🎒 **На пальцах.** Строка про live-traffic важнее, чем кажется: в бою эталонных ответов нет, поэтому годятся только метрики без reference. Из четырёх метрик RAGAS в этом режиме доступны faithfulness и answer relevance, а context recall — нет, ему нужен эталон. Отсюда и разделение таблицы на шесть сценариев.

## Ship It

Сохраните как `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

> 🎒 **На пальцах.** Порог «не меньше 50 размеченных человеком примеров» — минимум, ниже которого корреляция сама по себе шумная. На 50 примерах доверительный интервал для ро Спирмена всё ещё широкий, поэтому в разделе Concept и стоит цифра 100. Пятьдесят — это граница, за которой отказ, а не цель.

## Exercises

1. **Easy.** Прогоните RAGAS на 10 RAG-примерах с заранее известными галлюцинациями. Проверьте, что метрика faithfulness ловит каждую.
2. **Medium.** Разметьте руками 50 ответов на вопросы оценками 0 или 1 по корректности. Оцените их через G-Eval. Посчитайте ро Спирмена между судьёй и человеком.
3. **Hard.** Соберите CI-гейт на pytest с DeepEval. Намеренно ухудшите ретривер. Убедитесь, что гейт падает. Добавьте оповещение по нижнему квантилю: проверку порога на худших 10%.

> 🎒 **На пальцах.** Подсказка к третьему заданию: чтобы «намеренно ухудшить ретривер», не надо ничего переписывать — верните top-1 вместо top-5 или подмешайте случайные фрагменты. Faithfulness сразу уедет ниже порога 0.85, и assert упадёт. Гейт, который вы ни разу не видели красным, ещё не гейт.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| LLM-as-judge | Оцениваем с помощью LLM | Промптим модель-судью выставить оценку от 0 до 1 по рубрике. |
| RAGAS | Библиотека метрик для RAG | Открытый фреймворк оценки с 4 метриками для RAG без эталонов. |
| Faithfulness | Опирается ли ответ на контекст? | Доля утверждений ответа, следующих из найденного контекста. |
| Context precision | Были ли найденные фрагменты релевантны? | Доля фрагментов из top-K, которые реально пригодились. |
| Context recall | Нашёл ли ретривер всё? | Доля утверждений эталонного ответа, подтверждённых найденными фрагментами. |
| G-Eval | Свой LLM-судья | Рубрика плюс шаги рассуждения плюс оценка от 0 до 1. |
| Calibration | Доверяй, но проверяй | Корреляция Спирмена между оценкой судьи и оценкой человека. |

## Further Reading

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — статья про RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — статья про G-Eval.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — открытый продакшен-стек.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — предвзятость, калибровка, пределы метода.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — объединяющий фреймворк, который интегрирует RAGAS, DeepEval и Phoenix.
