<!-- i18n:manual -->
# Структурированные ответы и constrained decoding

> Просишь у LLM JSON. Обычно получаешь JSON. В продакшене проблема именно в слове «обычно». Constrained decoding превращает «обычно» во «всегда», правя логиты до сэмплирования.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## The Problem

Классификатор просит LLM: «Верни одно из {positive, negative, neutral}». Модель возвращает «The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...». Ваш парсер падает. F1 классификатора равен 0,0.

> 🎒 **На пальцах.** Вы попросили ответить «да» или «нет», а получили сочинение. Ответ по смыслу верный, но программа его не понимает и роняет весь конвейер. Одна лишняя фраза — и F1 не 0,9, а ровно 0,0.

Свободная генерация — это не контракт. Это пожелание. Продакшен-системе нужен контракт.

В 2026 году существует три уровня.

1. **Prompting.** Попросить вежливо. «Верни только JSON-объект». Работает примерно в 80 % случаев на передовых моделях и хуже на маленьких.
2. **Native structured output APIs.** `response_format` у OpenAI, tool use у Anthropic, JSON-режим у Gemini. Надёжно на поддерживаемых схемах. Привязка к вендору.
3. **Constrained decoding.** Менять логиты на каждом шаге генерации так, чтобы модель *не могла* выдать невалидный токен. 100 % валидность по построению. Работает с любой локальной моделью.

> 🎒 **На пальцах.** Три уровня — это три способа не получить кривой ответ. Просьба — как знак «просьба не сорить»: работает в 80 случаях из 100. Constrained decoding — как забор: сорить физически негде. Разница между 80 % и 100 % на потоке в 10 000 запросов — это 2 000 упавших разборов против нуля.

Этот урок даёт интуицию по всем трём и объясняет, когда за каким тянуться.

## The Concept

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**How constrained decoding works.** На каждом шаге генерации LLM выдаёт вектор логитов по всему vocabulary (около 100k токенов). Между моделью и сэмплером стоит *logit processor*. Он вычисляет, какие токены допустимы в текущей позиции целевой grammar — JSON schema, регулярное выражение, контекстно-свободная грамматика — и ставит логиты всех недопустимых токенов в минус бесконечность. Softmax по оставшимся логитам раскладывает всю вероятностную массу только на валидные продолжения.

> 🎒 **На пальцах.** Минус бесконечность в логите после softmax даёт ровно ноль вероятности. То есть из 100 000 вариантов на шаге остаются, скажем, 3 допустимых, а остальные 99 997 модель не выберет никогда — не «почти никогда», а математически никогда.

Реализации в 2026 году:

- **Outlines.** Компилирует JSON schema или регулярное выражение в конечный автомат. Для каждого токена — проверка допустимости за O(1). Основан на автомате, поэтому рекурсивные схемы приходится разворачивать в плоские.
- **XGrammar / llguidance.** Движки контекстно-свободных грамматик. Тянут рекурсивные JSON schema. Накладные расходы на декодирование почти нулевые. OpenAI упомянула llguidance в своей реализации структурированных ответов в 2025 году.
- **vLLM guided decoding.** Встроенные `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` через бэкенды Outlines, XGrammar или lm-format-enforcer.
- **Instructor.** Обёртка на Pydantic поверх любой LLM. Повторяет запрос при ошибке валидации. Работает у всех провайдеров, но логиты не трогает — держится на повторах и промптах, заточенных под структурированный вывод.

> 🎒 **На пальцах.** Первые три подхода не дают модели ошибиться, четвёртый ловит ошибку задним числом. Если Instructor настроен на 3 попытки, то в худшем случае вы платите за 3 запроса вместо одного и ждёте втрое дольше. Outlines платит один раз — за компиляцию автомата на старте.

### The counterintuitive result

Constrained decoding часто *быстрее* обычной генерации. Причин две. Во-первых, он сужает пространство поиска следующего токена. Во-вторых, толковые реализации вообще пропускают генерацию для вынужденных токенов (каркас вроде `{"name": "` — там каждый байт предопределён).

> 🎒 **На пальцах.** Посчитайте: в `{"name": "` десять символов, и ни один из них не требует выбора. Модель могла бы потратить на них 4-5 шагов генерации, а constrained decoder просто дописывает их сам и переходит к первому реально свободному месту — самому значению имени.

### The pitfall that costs you

Порядок полей имеет значение. Поставьте `answer` перед `reasoning` — и модель зафиксирует ответ до того, как подумает. JSON валиден. Ответ неверен. Никакая валидация этого не поймает.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Порядок полей в схеме — это логика, а не форматирование.

> 🎒 **На пальцах.** Модель пишет слева направо и не умеет возвращаться назад. Если первым идёт `"answer": "yes"`, то дальше она будет придумывать обоснование именно для «yes», даже если правильный ответ «no». Поменяли два поля местами — и на некоторых задачах точность прыгает на несколько пунктов, а код при этом не изменился ни на строку.

```figure
constrained-decoder
```

## Build It

### Step 1: regex-constrained generation from scratch

Смотрите `code/main.py` — там самостоятельная реализация автомата. Основная идея в 30 строк:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm, sample, max_tokens=256):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    for _ in range(max_tokens):
        if fsm.is_accept(state):
            break
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        if not valid:
            break
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

Автомат отслеживает, какие части grammar мы уже выполнили. `valid_tokens(state, tokenizer)` вычисляет, какие токены vocabulary могут продвинуть автомат, не уводя его с пути в принимающее состояние. `sample` — это тот же сэмплер, которым вы пользуетесь и так: функция вида `(logits) -> token_id`.

> 🎒 **На пальцах.** Автомат — это как турникеты в метро. В каждом состоянии открыт только один или несколько проходов, остальные закрыты. `mask_logits` и есть тот сотрудник, который закрывает лишние турникеты: он ставит `-inf` во все позиции, кроме перечисленных в `valid`.

Потолок `max_tokens` и проверка на пустой `valid` — не украшение. Grammar с циклом (или автомат, из текущего состояния которого принимающее просто недостижимо) никогда не выполнит условие `is_accept`, и бесконечный `while` превращает эту ошибку в описании грамматики в намертво повисшую генерацию. Всегда ограничивайте цикл и всегда обрабатывайте случай «здесь нет ни одного допустимого токена».

> 🎒 **На пальцах.** Разница между `while not fsm.is_accept(state)` и `for _ in range(max_tokens)` — это разница между «висим до конца времён» и «сдаёмся через 256 шагов». Ошиблись в грамматике, забыли выход из состояния — с `while` процесс просто перестаёт отвечать, и вы полдня ищете, где он застрял. С `for` вы получаете обрезанный ответ, но получаете его через секунду, и по обрезку сразу видно, на каком месте автомат заблудился. То же и с пустым `valid`: если допустимых токенов ноль, генерировать нечего — надо выходить, а не маскировать все 100 000 логитов в `-inf` и сэмплировать из ничего.

### Step 2: Outlines for JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Ноль ошибок валидации. Всегда. Автомат делает невалидный вывод недостижимым.

> 🎒 **На пальцах.** Поле `sentiment` объявлено как `Literal` из трёх значений. Значит на шаге, где начинается его значение, автомат разрешает ровно 3 продолжения из 100 000. Модель может выбрать любое из трёх — но не может написать «The sentiment is positive because...» даже при большом желании.

### Step 3: Instructor for provider-agnostic Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Механизм другой. Instructor не трогает логиты. Он вставляет схему в промпт, разбирает вывод и повторяет запрос при ошибке валидации (по умолчанию 3 раза). Работает с любым провайдером. Повторы добавляют задержку и стоимость. Продаётся он именно переносимостью между провайдерами.

> 🎒 **На пальцах.** `total_usd: float = Field(ge=0)` означает «число не меньше нуля». Если модель напишет −420, Pydantic откажет, и Instructor пойдёт на второй круг. Красиво, но за второй круг вы платите второй раз: и деньгами за токены, и секундой задержки.

### Step 4: native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Constrained decoding на стороне сервера. По надёжности на поддерживаемых схемах не уступает Outlines. Локальную модель держать не нужно. Привязывает вас к вендору.

> 🎒 **На пальцах.** Схема здесь описана прямо в запросе: тип object, обязательное поле `sentiment`, а его значение — enum из трёх строк. Тот же смысл, что у Pydantic-класса из шага 2, просто записанный на JSON schema и отправленный на чужой сервер.

## Pitfalls

- **Recursive schemas.** Outlines разворачивает рекурсию до фиксированной глубины. Древовидным выводам (вложенные комментарии, AST) нужны XGrammar или llguidance на контекстно-свободных грамматиках.
- **Huge enums.** Enum на 10 000 вариантов компилируется медленно или выпадает по таймауту. Переходите на ретривер: сначала предскажите top-k кандидатов, потом ограничивайте выбор ими.
- **Grammar too strict.** Заставьте поле `date` подчиняться регулярке `"YYYY-MM-DD"` — и модель не сможет написать `"unknown"` там, где даты нет. Она компенсирует это выдуманной датой. Разрешайте `null` или специальное значение-заглушку.
- **Premature commitment.** Смотрите про порядок полей выше. Всегда ставьте рассуждение первым.
- **Vendor JSON mode without schema.** Чистый JSON-режим гарантирует только валидный JSON, а не валидный *для вашей задачи*. Всегда передавайте полную схему.

> 🎒 **На пальцах.** Самая коварная ловушка здесь — третья. Слишком строгая grammar не защищает от вранья, а провоцирует его: у модели просто нет способа сказать «не знаю», и она выдумывает дату. Один разрешённый `null` в схеме убирает целый класс галлюцинаций.

## Use It

Стек 2026 года:

| Situation | Pick |
|-----------|------|
| Модель OpenAI/Anthropic/Google, простая схема | Родной структурированный вывод вендора |
| Любой провайдер, работа через Pydantic, повторы допустимы | Instructor |
| Локальная модель, нужна 100 % валидность, плоская схема | Outlines (FSM) |
| Локальная модель, рекурсивная схема | XGrammar или llguidance |
| Свой сервер инференса | vLLM guided decoding |
| Пакетная обработка, повторы приемлемы | Instructor + самая дешёвая модель |

> 🎒 **На пальцах.** Читайте таблицу по двум вопросам: чья модель и насколько сложна схема. Локальная модель плюс плоская схема — Outlines. Локальная модель плюс дерево — XGrammar. Чужой API — берите то, что вендор уже сделал сам, это почти всегда быстрее и дешевле своей обвязки.

## Ship It

Сохраните как `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

> 🎒 **На пальцах.** Обратите внимание на целевую метрику: schema compliance rate = 100 %, а не 99 %. При миллионе вызовов в месяц один процент — это 10 000 упавших ответов. Именно ради этого одного процента и существует constrained decoding.

## Exercises

1. **Easy.** Попросите небольшую модель с открытыми весами (например, Llama-3.2-3B) выдать `Review(sentiment, confidence, evidence_span)` без constrained decoding. Измерьте долю ответов, которые разбираются как валидный JSON, на 100 отзывах.
2. **Medium.** Тот же корпус через JSON-режим Outlines. Сравните долю соответствия схеме, задержку и смысловую точность.
3. **Hard.** Напишите с нуля декодер с ограничением по регулярке для телефонных номеров (`\d{3}-\d{3}-\d{4}`). Убедитесь, что на 1000 сэмплов нет ни одного невалидного вывода.

> 🎒 **На пальцах.** Подсказка к третьему заданию: регулярка `\d{3}-\d{3}-\d{4}` — это автомат ровно из 13 состояний, по одному на каждый выданный символ. В состояниях 3 и 7 допустим единственный токен — дефис. В остальных — только десять цифр. Больше в этом автомате ничего нет.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Constrained decoding | Force valid output | Маскировать логиты недопустимых токенов на каждом шаге генерации. |
| Logit processor | The thing that constrains | Функция вида `(logits, state) -> masked_logits`. |
| FSM | Finite-state machine | Скомпилированное представление grammar; проверка допустимого следующего токена за O(1). |
| CFG | Context-free grammar | Грамматика, умеющая в рекурсию; медленнее, но выразительнее автомата. |
| Schema field order | Does it matter? | Да — первое поле фиксирует решение; всегда ставьте рассуждение перед ответом. |
| Guided decoding | vLLM's name for it | То же самое, встроенное в сервер инференса. |
| JSON mode | OpenAI's early version | Гарантирует синтаксис JSON; НЕ гарантирует соответствие схеме. |

## Further Reading

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — статья про Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — быстрый constrained decoding на контекстно-свободных грамматиках.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — интеграция в сервер инференса.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — справочник по API и подводные камни.
- [Instructor library](https://python.useinstructor.com/) — Pydantic и повторы у разных провайдеров.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — сравнение 6 фреймворков constrained decoding.
