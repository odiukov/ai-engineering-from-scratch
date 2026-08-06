<!-- i18n:manual -->
# Модель награды и RLHF

> Человек не может написать функцию награды для «хорошего ответа ассистента», зато легко сравнит два ответа и выберет лучший. Обучите на этих сравнениях модель награды, а потом дообучите языковую модель через RL против неё. Кристиано, 2017. InstructGPT, 2022. Рецепт, который превратил GPT-3 в ChatGPT. В 2026-м его почти везде вытеснил DPO — но мысленная модель осталась прежней.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## The Problem

Вы обучили языковую модель предсказывать следующий токен. Она пишет грамматически правильно. Ещё она врёт, растекается мыслью и не умеет отказывать. Дополнительным предобучением это не лечится: тексты из интернета — это причина проблемы, а не лекарство.

Вам нужна *скалярная награда*, которая говорит: «для инструкции X ответ A лучше ответа B». Написать такую функцию руками невозможно. «Полезность» не выражается формулой над токенами. Зато человек может сравнить два ответа и отметить, какой лучше. Собирать такие метки дёшево и в больших количествах.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) превращает предпочтения в модель награды, а потом оптимизирует языковую модель через PPO против этой награды. Три шага: SFT → RM → PPO. Именно этот рецепт выпустил в мир ChatGPT, Claude, Gemini и все остальные выровненные LLM 2023–2025 годов.

В 2026 году шаг PPO чаще всего заменяют на DPO (Phase 10 · 08): дешевле и почти так же хорошо для выравнивания. Но сама *модель награды* никуда не делась — на ней держится любой Best-of-N сэмплер, любой пайплайн RL с проверяемыми наградами и каждая reasoning-модель с process reward model. Поймёте RLHF — поймёте весь стек выравнивания.

> 🎒 **На пальцах.** Представьте дегустацию: описать формулой «вкусное блюдо» невозможно, а сказать «вот это вкуснее того» может любой. RLHF ровно про это: люди дают только сравнения пар, а модель награды сама достраивает из них числовую шкалу. Разметчику на одну пару нужно секунд тридцать, поэтому 50 000 сравнений — это реальный бюджет, а 50 000 написанных вручную идеальных ответов — нет.

## The Concept

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** Берём предобученную базовую модель. Дообучаем на написанных людьми примерах нужного поведения (ответы по инструкции, полезные реплики и так далее). Получаем модель `π_SFT`, которая *смещена в сторону хорошего поведения*, но пространство её действий по-прежнему ничем не ограничено.

**Stage 2: Reward Model training.**

- Собираем пары ответов `(y_+, y_-)` на промпты `x`, где человек отметил: «`y_+` лучше, чем `y_-`».
- Обучаем модель награды `R_φ(x, y)` давать более высокий балл ответу `y_+`.
- Функция потерь — **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ — сигмоида. Разница наград задаёт логарифм шансов предпочтения. Модель Брэдли-Терри держится в стандарте с 1952 года и до сих пор остаётся главным выбором в RLHF.

- `R_φ` обычно инициализируют из SFT-модели, приклеив сверху скалярную голову. Тот же трансформер, один линейный слой на выходе выдаёт награду.

> 🎒 **На пальцах.** Пусть модель дала хорошему ответу 2.0, а плохому 0.5. Тогда σ(2.0 − 0.5) = σ(1.5) ≈ 0.82, потери = −log(0.82) ≈ 0.20 — модель почти права, штраф маленький. Если бы она поставила обоим поровну, σ(0) = 0.5 и потери = 0.69. Важна только *разница* баллов: сдвиньте оба на +100, и ничего не изменится.

**Stage 3: PPO against the RM with KL penalty.**

- Обучаемую политику `π_θ` инициализируем из `π_SFT`. Рядом держим замороженную *референсную* модель `π_ref = π_SFT`.
- Награда в конце ответа `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL-штраф не даёт `π_θ` уползти сколь угодно далеко от `π_SFT` — это *регуляризатор*, а не жёсткий trust region. Обычно `β` берут `0.01`-`0.05`.
- Запускаем PPO (урок 08) с этой наградой. Преимущества считаются по траектории на уровне токенов, а модель награды оценивает только ответ целиком.

> 🎒 **На пальцах.** Подставьте числа: модель награды дала 3.0, KL до референса вышел 2.0, β = 0.02. Итоговая награда = 3.0 − 0.02 · 2.0 = 2.96, штраф почти незаметен. А если политика разошлась и KL дорос до 50, останется 3.0 − 1.0 = 2.0 — вот тогда штраф начинает тянуть модель назад.

**Why the KL?** Без него PPO с удовольствием найдёт способы взломать награду — модель награды обучалась только на ответах из своего распределения. Ответ из-за пределов этого распределения может получить балл выше, чем любой написанный человеком. KL удерживает `π_θ` рядом с той областью, где модель награды что-то понимает. Это самая важная ручка во всём RLHF.

> 🎒 **На пальцах.** Это как студент, который вместо учёбы разгадал привычки проверяющего: пишет «во-первых, во-вторых, в-третьих», льёт воду — и получает пятёрки за пустой текст. Модель награды тоже проверяющий с привычками, и политика найдёт их быстрее человека. KL-штраф — это правило «отвечай примерно в том же стиле, что и раньше», которое обрубает такие фокусы.

**2026 status:**

- **DPO** (Rafailov 2023): алгебраическое преобразование схлопывает шаги 2 и 3 в одну обычную supervised-функцию потерь прямо на данных предпочтений. Ни модели награды, ни PPO. Качество на бенчмарках выравнивания то же за долю вычислений. Разбирается в Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO с групповой относительной базой вместо критика, награда приходит от *верификатора* (код запустился, ответ по математике совпал), а не от обученной на людях модели награды. Стандарт для reasoning-моделей. Разбирается в Phase 9 · 12.
- **Process reward models (PRMs):** оценивают частичные решения (каждый шаг рассуждения), используются и в RLHF, и в вариантах GRPO для рассуждений.
- **Constitutional AI / RLAIF:** предпочтения генерирует выровненная LLM вместо человека. Так масштабируется бюджет на разметку.

> 🎒 **На пальцах.** Разница между RLHF и GRPO — в источнике награды. В RLHF награду угадывает обученная сеть, и её можно взломать. В GRPO награда объективная: тест на коде либо прошёл, либо нет. Поэтому для математики и кода взяли GRPO, а для «будь вежливым и полезным» — где никакого автотеста не придумать — остались модели награды.

```figure
reward-model
```

## Build It

В этом уроке «промпты» и «ответы» — крошечные синтетические строки. Модель награды — линейный скорер над мешком токенов. Никакой настоящей LLM: важна *форма* пайплайна, а не масштаб. Смотрите `code/main.py`.

### Step 1: synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

В настоящем RLHF здесь сидят живые разметчики. Форма данных — `(prompt, preferred_response, rejected_response)` — ровно та же.

> 🎒 **На пальцах.** Здесь мы жульничаем сознательно: «хороший» ответ — это два слова из `GOOD_WORDS`, например «clear thorough», а «плохой» — два из `BAD_WORDS`, например «rude wrong». Разметчика заменяет само правило генерации, так что метки идеально чистые. В жизни примерно треть меток шумная, и это меняет всё — смотрите раздел Pitfalls.

### Step 2: Bradley-Terry reward model

Линейная оценка: `R(x, y) = w · bag(y)`. Обучаем минимизировать попарные логистические потери Брэдли-Терри:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

После нескольких сотен обновлений `w` даёт положительные веса токенам-«хорошим словам» и отрицательные — «плохим».

> 🎒 **На пальцах.** Разберём одно обновление. В начале все веса нули, значит `r_pos = r_neg = 0`, `p = sigmoid(0) = 0.5`. При `lr = 0.1` и `cnt = 1` каждое слово хорошего ответа получает +0.05, каждое слово плохого −0.05. Чем увереннее модель (`p` ближе к 1), тем меньше множитель `(1 - p)` и тем слабее шаг — на уже понятных парах обучение само затухает.

### Step 3: PPO-like policy on top of RM

Наша игрушечная политика выдаёт один токен из словаря. Оцениваем токен моделью награды, считаем `log π_θ(token | prompt)`, добавляем KL-штраф до референса и применяем клиппированный суррогат PPO.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

> 🎒 **На пальцах.** Обратите внимание: `reward` считается по одному-единственному токену, но структура кода та же, что в настоящем RLHF, где модель награды оценивает ответ на 500 токенов. Если модель награды дала токену 1.2, а KL между `probs` и `probs_ref` равен 0.4, при `beta = 0.1` награда = 1.2 − 0.04 = 1.16. Замените один токен на целый ответ, а словарь на 128 тысяч токенов — и это уже InstructGPT.

### Step 4: monitor the KL

Следите за средним `KL(π_θ || π_ref)` на каждом обновлении. Если он переползает `~5-10`, политика ушла далеко от `π_SFT` — либо `β` слишком мал, либо начинается взлом награды. В настоящем RLHF это диагностика номер один.

> 🎒 **На пальцах.** KL можно читать как «насколько сильно модель стала другой». KL = 0 — политика в точности копия SFT. KL около 1-2 — заметно изменилась, но узнаваема. KL = 20 — это уже другая модель, и баллы от RM про неё ничего не значат, потому что таких текстов она не видела. Поэтому график KL смотрят раньше, чем график награды.

### Step 5: the production recipe with TRL

Когда игрушечный пайплайн стал понятен, посмотрим на тот же цикл глазами обычного пользователя библиотеки. [TRL](https://huggingface.co/docs/trl) от Hugging Face — эталонная реализация: `RewardTrainer` для этапа 2 и `PPOTrainer` (со встроенным KL до референса) для этапа 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

> 🎒 **На пальцах.** Смотрите на `num_labels=1`: это и есть «скалярная голова» из раздела Concept — вместо распределения по классам модель выдаёт одно число. А строки датасета `{"prompt", "chosen", "rejected"}` — те же самые `(x, y_+, y_-)` из формулы Брэдли-Терри, только названные человеческими словами.

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Три вещи библиотека делает за вас. `adap_kl_ctrl=True` включает адаптивное расписание β: если наблюдаемый KL выше `target_kl`, β удваивается; если ниже половины — уполовинивается. Референсная модель заморожена по договорённости — важно случайно не разделить параметры с `policy`. А голова ценности живёт на том же бэкбоне, что и политика (`AutoModelForCausalLMWithValueHead` навешивает скалярный MLP), поэтому TRL отдельно печатает `policy/kl` и `value/loss`.

> 🎒 **На пальцах.** Проследите адаптацию β на числах из конфига: `target_kl=6.0`, старт `init_kl_coef=0.05`. Замерили KL = 9 (выше цели) — β становится 0.1, штраф жёстче, модель прижимается к референсу. Замерили KL = 2 (ниже половины от 6) — β падает до 0.025, поводок отпускают. Так система сама держит KL около шестёрки, вместо того чтобы вы подбирали β руками.

## Pitfalls

- **Over-optimization / reward hacking.** Модель награды несовершенна; `π_θ` находит состязательные ответы с высоким баллом и низким качеством. Симптомы: награда растёт без остановки, а человеческая оценка стоит на месте или падает. Лечение: ранняя остановка, больший `β`, более разнообразные данные для RM.
- **Length hacking.** Модели награды, обученные на полезных ответах, обычно неявно поощряют длину. Политика учится лить воду. Лечение: нормировать награду по длине или RLAIF с учитывающей длину моделью награды.
- **Too-small RM.** Важно, способна ли RM *судить* выходы политики, а не совпадает ли она с политикой по числу параметров: в InstructGPT политику на 175B оценивала RM на 6B. Но если RM мала для самой задачи судейства, её точность на предпочтениях упирается в потолок и её становится тривиально взломать. Проверяйте достаточность RM по pairwise-точности на отложенных парах, а не по соотношению размеров.
- **KL tuning.** Слишком маленький β — дрейф и взлом награды. Слишком большой β — политика почти не меняется. Стандартный приём: *адаптивный* β, который целится в фиксированный KL за шаг.
- **Preference-data noise.** Около 30% человеческих меток шумные или спорные. Калибруйте: обучайте RM на данных, отфильтрованных по согласию разметчиков, или добавьте температуру в Брэдли-Терри.
- **Off-policy problems.** После первой эпохи данные PPO уже слегка off-policy. Следите за долей клиппинга, как в уроке 08.

> 🎒 **На пальцах.** Взлом длины видно невооружённым глазом: средняя длина ответа за обучение уезжает с 200 токенов до 800, балл RM растёт с 1.5 до 4.0, а люди в слепом сравнении выбирают старую модель. Никакой магии — в обучающих парах длинные ответы чаще помечали как лучшие, и RM выучила «длиннее = лучше». Всегда логируйте среднюю длину рядом с наградой.

## Use It

RLHF в 2026 году устроен слоями:

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) предпочтительнее RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO с наградой от верификатора (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO с process reward models по шагам. |
| Safety / refusal behavior | Safety | RLHF-PPO с отдельной safety-моделью награды или Constitutional AI. |
| Best-of-N at inference | Fast alignment | Модель награды применяется при декодировании; политику обучать не нужно. |
| Reward distillation | Inference compute | Обучить маленькую «голову награды» поверх замороженной LM. |

RLHF был *тем самым* методом в 2022–2024 годах. В 2026-м продакшен-пайплайны выравнивания начинают с DPO, а PPO оставляют для шагов, где без модели награды не обойтись, или где на кону безопасность.

> 🎒 **На пальцах.** Самая недооценённая строка таблицы — Best-of-N. Модель награды у вас уже есть, обучать политику не надо: генерируете 8 ответов, оцениваете каждый и отдаёте лучший. Стоит это ровно в 8 раз дороже по инференсу, зато выкатывается за день и откатывается одной строкой конфига. Прежде чем запускать PPO на неделю, попробуйте это.

## Ship It

Сохраните как `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM whose held-out pairwise accuracy was never measured (parameter count relative to the policy is not the test — InstructGPT used a 6B RM for a 175B policy). Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

> 🎒 **На пальцах.** Обратите внимание на пункт про holdout human eval: слепой набор, размеченный людьми, — единственный детектор взлома награды. Балл RM растёт всегда, потому что вы буквально оптимизируете именно его. Если 200 отложенных примеров, которые модель награды не видела, показывают падение — обучение пора останавливать, каким бы красивым ни был график награды.

## Exercises

1. **Easy.** Обучите модель награды Брэдли-Терри из `code/main.py` на 500 синтетических парах предпочтений. Замерьте попарную точность на отложенных 100 парах. Должно получиться больше 90%.
2. **Medium.** Прогоните игрушечный цикл PPO-RLHF с `β ∈ {0.0, 0.1, 1.0}`. Для каждого случая постройте график балла RM против KL до референса по ходу обновлений. Где происходит взлом награды?
3. **Hard.** Реализуйте DPO (замкнутая функция потерь на правдоподобии предпочтений) на тех же данных и сравните с пайплайном RLHF-PPO по потраченным вычислениям и по достигнутому баллу RM.

> 🎒 **На пальцах.** Подсказка ко второму заданию: `β = 0.0` — это выключенный KL-штраф, и именно там взлом виден лучше всего. График пойдёт так: балл RM растёт до 5-6, а KL улетает за 50 — политика нашла пару токенов, которые RM любит, и повторяет их. При `β = 1.0` картина обратная: KL держится около нуля, но и награда почти не растёт. Полезное поведение — посередине.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| RLHF | «RL для выравнивания» | Трёхэтапный пайплайн SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | «Сетка, которая ставит баллы» | Обученная скалярная функция, подогнанная под попарные предпочтения через Брэдли-Терри. |
| Bradley-Terry | «Попарные логистические потери» | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; стандартная целевая функция для RM. |
| KL penalty | «Держись рядом с референсом» | `β · KL(π_θ \|\| π_ref)` в награде; регуляризатор против взлома награды. |
| Reward hacking | «Закон Гудхарта» | Политика эксплуатирует дыры RM; симптомы: награда вверх, человеческая оценка на месте. |
| RLAIF | «Предпочтения размечает ИИ» | RLHF, где метки даёт другая языковая модель вместо человека. |
| PRM | «Process Reward Model» | Оценивает частичные шаги рассуждения; используется в reasoning-пайплайнах. |
| Constitutional AI | «Метод Anthropic» | Предпочтения генерирует ИИ по явно прописанным правилам. |

## Further Reading

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — статья, с которой начался RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — рецепт, стоящий за ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — более ранний RLHF для суммаризации.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO, вариант по умолчанию после эпохи RLHF в 2026 году.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF и цикл самокритики.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — та самая статья про HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — продакшен-реализации `RewardTrainer` и `PPOTrainer`. Прочитайте исходники тренеров ради деталей адаптивного KL и головы ценности.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — каноничный разбор трёхэтапного пайплайна с картинками.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — сама библиотека; в `examples/` лежат сквозные RLHF-скрипты для Llama, Mistral и Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — взгляд через гипотезу награды; обязательная база для разговора про взлом награды.
