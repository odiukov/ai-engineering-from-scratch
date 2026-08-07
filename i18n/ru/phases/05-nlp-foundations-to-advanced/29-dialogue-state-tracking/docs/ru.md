<!-- i18n:manual -->
# Отслеживание состояния диалога (dialogue state tracking)

> «I want a cheap restaurant in the north... actually make it moderate... and add Italian.» Три turn — три обновления состояния. Dialogue state tracking держит словарь slot-значений в актуальном виде, чтобы бронирование сработало.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 minutes

## The Problem

В диалоговой системе, заточенной под задачу, цель пользователя записывается набором пар slot-значение: `{cuisine: italian, area: north, price: moderate}`. Каждый turn пользователя может добавить slot, изменить его или убрать. Система обязана прочитать весь разговор и выдать текущее состояние без ошибок.

Ошибка в одном slot — и система бронирует не тот ресторан, ставит не тот рейс или списывает деньги не с той карты. DST — это шарнир между тем, что сказал пользователь, и тем, что выполнит бэкенд.

> 🎒 **На пальцах.** Slot — это графа в бланке заявки. Пустой бланк: cuisine, area, price. Пользователь говорит фразу — вы заполняете графу. Ошибка в одной графе из трёх портит всю заявку, ровно как одна неверная цифра в номере карты.

Почему это по-прежнему важно в 2026 году, несмотря на LLM:

- Области с требованиями к комплаенсу (банки, медицина, продажа авиабилетов) требуют детерминированных значений slot, а не свободной генерации.
- Агентам с инструментами всё равно нужно разрешить slot, прежде чем звать API.
- Правки через несколько turn сложнее, чем кажется: «actually no, make it Thursday».

Современный конвейер: классические идеи DST + LLM-экстракторы + ограничители на структурированный вывод.

> 🎒 **На пальцах.** Разница между «понял смысл» и «заполнил бланк». LLM прекрасно пересказывает, но банку нужно ровно значение `moderate` из трёх допустимых, а не «средний ценовой сегмент, наверное». Самый коварный пункт из трёх — последний: «actually no, make it Thursday» меняет один slot и не должен трогать остальные.

## The Concept

![DST: dialog history → slot-value state](../assets/dst.svg)

**Task structure.** Схема описывает домены (restaurant, hotel, taxi) и их slot (cuisine, area, price, people). Каждый slot может быть пустым, заполненным значением из закрытого списка (price: {cheap, moderate, expensive}) или свободным текстом (name: "The Copper Kettle").

> 🎒 **На пальцах.** Схема — это меню с фиксированными вариантами. Price выбирается из трёх: cheap, moderate, expensive, свою цифру туда не впишешь. А name свободный: «The Copper Kettle» никаким списком не предусмотришь. В примере урока пять slot: cuisine, area, price, people, day.

**Two DST formulations.**

- **Classification.** Для каждой пары (slot, candidate_value) предсказываем да/нет. Работает для slot с закрытым словарём. Стандарт до 2020 года.
- **Generation.** По диалогу генерируем значения slot свободным текстом. Работает для slot с открытым словарём. Современный вариант по умолчанию.

> 🎒 **На пальцах.** Classification — это опросник с галочками: для трёх значений price надо задать три вопроса, «это cheap?», «это moderate?», «это expensive?». Generation — просто написать ответ словами. При 5 slot и 5 значениях в среднем классификации нужно 25 проверок на каждый turn, а генерации — один вызов.

**Metric.** Joint Goal Accuracy (JGA) — доля turn, где верны *все* slot сразу. Всё или ничего. Таблица лидеров MultiWOZ 2.4 в 2026 году упирается примерно в 83%.

> 🎒 **На пальцах.** JGA беспощаден как контрольная без частичных баллов. Угадали 4 slot из 5 — turn засчитан как ошибка. Поэтому 83% на MultiWOZ 2.4 звучит скромно, хотя точность по отдельным slot там заметно выше 90%.

**Architectures.**

1. **Rule-based (slot regex + keyword).** Крепкая база для узких доменов. Легко отлаживать.
2. **TripPy / BERT-DST.** Генерация копированием поверх BERT-энкодера. Стандарт до эпохи LLM.
3. **LDST (LLaMA + LoRA).** LLM, дообученная на инструкциях, с промптом по парам «домен-slot». Достаёт уровень ChatGPT на MultiWOZ 2.4.
4. **Ontology-free (2024–26).** Схему выбрасываем; имена slot и значения генерируем напрямую. Тянет открытые домены.
5. **Prompt + structured output (2024–26).** LLM плюс Pydantic-схема плюс ограниченное декодирование. Пять строк кода, готово к продакшену.

> 🎒 **На пальцах.** Пять архитектур — это пять поколений подряд, от регулярок до пяти строк с Pydantic. Обратите внимание на направление: чем ниже пункт, тем меньше кода и больше вычислений. Первый вариант вы отладите за вечер, и он не будет стоить ни копейки за вызов; пятый напишете за десять минут, но заплатите за каждый turn.

### The classic failure modes

- **Co-reference across turns.** «Let's stay with the first option.» Нужно понять, о каком варианте речь.
- **Over-write vs append.** Пользователь говорит «add Italian». Вы заменяете cuisine или дописываете?
- **Implicit confirmations.** «OK cool» — это согласие на предложенную бронь или нет?
- **Correction.** «Actually make it 7 pm.» Надо обновить время и не стереть остальные slot.
- **Coreference to previous system utterance.** «Yes, that one.» Какой именно «that»?

> 🎒 **На пальцах.** Все пять провалов — про местоимения и передумывание. Живой человек в разговоре с официантом делает так постоянно: «то же самое, но подешевле», «нет, лучше в четверг». Регулярка не поймёт ни одного из пяти примеров — ради этого и написан весь остальной урок.

```figure
n5-slot-tracker
```

## Build It

### Step 1: rule-based slot extractor

Смотрите `code/main.py`. Регулярки и словари синонимов покрывают 70% канонических фраз в узких доменах:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

За пределами канонического словаря всё хрупко. Зато работает там, где значения slot подтверждаются детерминированно.

> 🎒 **На пальцах.** Словарь синонимов — это шпаргалка «как ещё это называют». Для italian здесь четыре варианта: italian, pasta, pizza, italy. Значит фраза «I fancy some pizza» вернёт `italian`. А «I want carbonara» вернёт `None`, потому что carbonara в списке нет — вот вам и обещанные 70% покрытия.

### Step 2: state update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
            continue
        if is_negation(utterance, slot):
            new_state[slot] = None
    return new_state
```

Три инварианта:

- Никогда не сбрасывайте slot, который пользователь не трогал.
- Явное отрицание («never mind the cuisine») обязано очищать.
- Правка пользователя («actually...») обязана перезаписывать, а не дописывать.

Обратите внимание на `continue`. Внутри одного turn извлечение важнее отрицания, потому что фраза «never mind the cuisine, any food is fine» одновременно отменяет slot и заполняет его заново — очистка после извлечения выбросила бы значение, которое пользователь только что назвал.

> 🎒 **На пальцах.** Функция `update_state` начинается со строки, где старое состояние копируется целиком. Это и есть первый инвариант, записанный кодом: трогаем только те slot, для которых экстрактор что-то нашёл. Начнись функция с пустого словаря — каждый turn стирал бы всё сказанное раньше.

> 🎒 **На пальцах.** Второй и третий инварианты живут в одном цикле, и порядок в нём решает всё. Заметьте, что и извлечение, и проверка на отрицание идут по одному и тому же slot внутри одной итерации: `is_negation(utterance, slot)` спрашивает не «есть ли в фразе отрицание вообще», а «отменяет ли пользователь именно этот slot». Разберите фразу «never mind the cuisine, any food is fine». Экстрактор cuisine находит в ней `any` и ставит `cuisine = "any"`. Отрицание «never mind the cuisine» тоже сработало бы — и обнулило бы slot. Кто прав? Пользователь: он сказал, что кухня не важна, то есть `any`, а не «графа не заполнена». Поэтому `continue` уводит нас из итерации сразу после успешного извлечения, и до проверки отрицания дело просто не доходит. Уберите `continue` — и slot очистится на той же фразе, где пользователь его заполнил.

### Step 3: LLM-driven DST with structured output

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor
from anthropic import Anthropic


class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


client = instructor.from_anthropic(Anthropic())


def render_dialog(turns):
    return "\n".join(f"  user: {u}" for u in turns)


def llm_dst(history):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render_dialog(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        response_model=RestaurantState,
        messages=[{"role": "user", "content": prompt}],
    )
```

Instructor плюс Pydantic гарантируют валидный объект состояния. Ни регулярок, ни расхождений со схемой, ни выдуманных slot. Аргумент `response_model` — это и есть то, что добавляет instructor: он оборачивает клиент провайдера, поэтому сам вызов остаётся обычным `create` этого провайдера, а на выходе вместо сырого сообщения приходит проверенный `RestaurantState`.

> 🎒 **На пальцах.** `Optional[Literal[...]]` — тот же бланк с выпадающим списком, только для модели. Cuisine принимает ровно 5 значений плюс пустое; вписать «japanese» модель физически не сможет, библиотека не пропустит. Пять полей, ноль регулярок.

> 🎒 **На пальцах.** Строка `client = instructor.from_anthropic(Anthropic())` объясняет весь фокус. `Anthropic()` — обычный клиент API, у него есть метод `messages.create`. Instructor не заменяет его своим API, а надевает поверх обёртку: тот же `client.messages.create` с теми же `model`, `max_tokens` и `messages`, плюс один новый аргумент `response_model`. Разница только в том, что возвращается: без instructor вы получаете объект сообщения и сами вытаскиваете из него текст, потом сами разбираете JSON, потом сами проверяете поля. С instructor вы сразу получаете `RestaurantState`, у которого можно писать `state.cuisine`. Если модель наврала со схемой, instructor сам отправит ей ошибку валидации и попросит переписать ответ.

> 🎒 **На пальцах.** `render_dialog` — одна строка кода и один важный момент: диалог подаётся модели как текст, а не как список сообщений. Каждый turn превращается в строку `  user: ...`, и все они склеиваются переводами строк. Именно поэтому в промпте стоит `{render_dialog(history)}` — f-строка подставляет туда готовую расшифровку разговора. Формат тут почти не важен, важна консистентность: если на обучающих примерах вы показывали модели один вид расшифровки, а в бою подаёте другой, качество упадёт без всякой причины в коде.

### Step 4: JGA evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    if len(predicted_states) != len(gold_states):
        raise ValueError("predicted and gold must have the same length")
    if not predicted_states:
        return 0.0
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Обе проверки важны, когда метрика стоит в CI-гейте: на пустом наборе turn вы получите не метрику, а `ZeroDivisionError`, а расхождение длин заставит `zip` молча обрезаться по более короткому списку — и прогон, потерявший половину предсказаний, покажет идеальный результат.

Откалибруйтесь: в какой доле turn система угадывает ВСЕ slot? На MultiWOZ 2.4 лучшие системы 2026 года дают 80-83%. Ваша система в своём узком домене должна быть выше, иначе базовая LLM обыгрывает вас без всякой разработки.

> 🎒 **На пальцах.** Смысл метрики — в строке `p == g`: словари сравниваются целиком. Если из 100 turn полностью верны 80, JGA равен 0.80. Один лишний slot в предсказании — turn уже неверен, хотя все остальные значения совпали.

> 🎒 **На пальцах.** Две проверки сверху добавлены не для красоты, они закрывают два способа обмануть себя. Первая: `zip` в Python не жалуется на списки разной длины, он просто останавливается на конце короткого. Если ваш трекер упал на 50-м turn из 100 и вернул 50 предсказаний, `zip` сравнит эти 50 с первыми 50 эталонами, а `len(predicted_states)` тоже даст 50 — и вы увидите JGA = 1.0 на прогоне, где половина данных потерялась. Вторая: на пустом списке `correct / len(...)` — это деление на ноль, то есть падение с `ZeroDivisionError` вместо числа. В CI такое падение выглядит как сломанный тест, а не как «эталонного набора не нашлось», и искать вы будете не там.

### Step 5: handling correction

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Используйте это как сигнал маршрутизации, а не как правку состояния. «Перезаписать последний изменённый slot» звучит как решение, но трекер на правилах выше вообще не помнит, какой slot был записан и когда — вам пришлось бы вести по каждому slot журнал «обновлён на turn N» и надеяться, что правка касается именно этого slot, а не какого-то более раннего. На практике правка почти всегда несёт с собой собственное значение («actually make it moderate»), так что экстракторы и без вас перезапишут нужный slot; а вот сам факт правки стоит записать в лог, чтобы пометить turn на подтверждение или отправить его в LLM. Это и есть современный приём: заметив правку, дайте LLM пересобрать всё состояние из истории, вместо того чтобы латать один slot — при пересборке правки обрабатываются сами собой.

> 🎒 **На пальцах.** Список подсказок для правок — четыре фразы: actually, no wait, on second thought, change that to. Расширить его несложно, но обратите внимание, чего эта функция НЕ делает: она ничего не меняет в состоянии. Она возвращает `True`/`False` — и всё. Считайте её лампочкой на приборной панели, а не гаечным ключом.

> 🎒 **На пальцах.** Почему совет «перезаписать последний изменённый slot» не работает. Посмотрите на `update_state` из Step 2: она копирует словарь и раскладывает по нему новые значения. Нигде не сохраняется, какой slot менялся последним, — состояние это просто `{cuisine: italian, area: north}`, без истории. Чтобы узнать «последний изменённый», нужен второй словарь вида `{cuisine: turn 1, area: turn 3}`, и его надо честно обновлять на каждом turn. А потом всё равно угадывать: во фразе «actually, no wait, make that the south» правится area, но кто сказал, что пользователь правит именно последний по времени slot, а не тот, о котором говорили три turn назад? К счастью, чинить это обычно не нужно: «actually make it moderate» содержит слово `moderate`, экстрактор price его находит и перезаписывает slot сам, без всякого журнала. Настоящая ценность `is_correction` — в том, что на таком turn можно переспросить пользователя или отдать turn LLM.

## Pitfalls

- **Full-history regeneration cost.** Если LLM пересобирает состояние на каждом turn, суммарный расход токенов растёт как O(n²). Ограничивайте историю или сжимайте старые turn.
- **Schema drift.** Добавили новый slot задним числом — сломали старые обучающие данные. Версионируйте схему.
- **Case sensitivity.** «Italian», «italian» и «ITALIAN» — нормализуйте везде.
- **Implicit inheritance.** Если пользователь раньше сказал «for 4 people», новый запрос про другое время не должен обнулять people. Всегда передавайте полную историю.
- **Free-form vs closed-set.** Имена, время и адреса требуют свободных slot; кухни и районы закрыты. В схеме будут и те, и другие.

> 🎒 **На пальцах.** Про O(n²) на конкретных числах: если каждый turn добавляет 100 токенов, а состояние пересобирается по всей истории, то 10 turn стоят примерно 5500 токенов, а 50 turn — уже 127 500. Диалог вырос в 5 раз, счёт вырос в 23 раза. Отсюда и совет ограничивать историю.

## Use It

Стек 2026 года:

| Situation | Approach |
|-----------|----------|
| Узкий домен (один-два intent) | Правила плюс регулярки |
| Широкий домен, есть размеченные данные | LDST (LLaMA + LoRA на данных в духе MultiWOZ) |
| Широкий домен, разметки нет, нужен продакшен | LLM + Instructor + Pydantic-схема |
| Голос и распознавание речи | ASR + нормализатор + LLM-DST |
| Многодоменный сценарий бронирования | LLM по схеме, своя Pydantic-модель на каждый домен |
| Чувствительно к комплаенсу | Правила основным путём, LLM запасным, с явным подтверждением |

> 🎒 **На пальцах.** Таблица — это шкала «дёшево и предсказуемо» против «дорого и гибко». Верхняя строка: узкий домен и регулярки, ноль стоимости за вызов и полная воспроизводимость. Нижняя строка: комплаенс, правила основные, LLM подстраховывает, и перед списанием денег система обязательно переспрашивает.

## Ship It

Сохраните как `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Exercises

1. **Easy.** Соберите в `code/main.py` трекер состояния на правилах для трёх slot (cuisine, area, price). Проверьте на 10 диалогах, написанных руками. Померьте JGA.
2. **Medium.** Тот же набор данных, но с Instructor, Pydantic и небольшой LLM. Сравните JGA. Разберите самые трудные turn.
3. **Hard.** Реализуйте оба подхода и настройте маршрутизацию: правила основным путём, LLM запасным, когда правила извлекли меньше двух slot уверенно. Померьте общий JGA и стоимость вывода на один turn.

> 🎒 **На пальцах.** Подсказка к третьему заданию: порог «меньше двух slot» — не догма, а ручка настройки. Прогоните свои 10 диалогов с порогом 1, 2 и 3 и посмотрите, как меняются JGA и доля обращений к LLM. Скорее всего окажется, что в LLM уходят 20-30% turn, а JGA растёт на несколько пунктов — вот и вся экономика решения.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DST | «Dialogue state tracking» | Поддерживать словарь slot-значений на протяжении всех turn диалога. |
| Slot | «Единица намерения пользователя» | Именованный параметр, который нужен бэкенду (cuisine, date). |
| Domain | «Область задачи» | Restaurant, hotel, taxi — наборы slot. |
| JGA | «Joint Goal Accuracy» | Доля turn, где верны все slot сразу. Всё или ничего. |
| MultiWOZ | «Тот самый бенчмарк» | Многодоменный WOZ-датасет; стандартная оценка DST. |
| Ontology-free DST | «Без схемы» | Имена slot и значения генерируются напрямую, фиксированного списка нет. |
| Correction | «Actually...» | Turn, который перезаписывает уже заполненный slot. |

## Further Reading

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — канонический бенчмарк.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — дообучение LLaMA с LoRA на инструкциях для DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — рабочая лошадка DST на копировании.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — диалог без разметки через EM-алгоритм.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — канонические результаты по DST.
