"""
Constitutional AI и самоулучшение — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

import math
import re

# Конституция: список принципов. У каждого есть id, по которому критик
# выбирает проверку, и человекочитаемый текст — именно его в реальной
# системе подставляют в промпт «раскритикуй свой ответ по этому правилу».
CONSTITUTION = (
    {"id": "no_refusal", "text": "The response must not refuse a reasonable, benign request."},
    {"id": "no_padding", "text": "The response must not include unnecessary filler or padding."},
    {"id": "no_hedging", "text": "The response must directly answer the question asked, without hedging."},
    {"id": "plain_number", "text": "If the question has a single numeric answer, state the number plainly."},
)

REFUSAL_PREFIXES = ("i can't", "i cannot", "i'm unable", "as an ai")

HEDGE_WORDS = frozenset(
    {"maybe", "perhaps", "possibly", "probably", "might", "seems", "apparently"}
)

MAX_WORDS = 40           # предел для принципа no_padding
PLAIN_ANSWER_WORDS = 12  # предел для принципа plain_number


def critique(response, principle):
    """Критика одного ответа по ОДНОМУ принципу конституции.

    Вернуть {"principle": <id принципа>, "problems": [...]} — список из
    нуля или одной найденной проблемы.

    critique("Paris.", CONSTITUTION[0])
        ->  {"principle": "no_refusal", "problems": []}
    critique("I cannot help. Paris.", CONSTITUTION[0])
        ->  {"principle": "no_refusal", "problems": ["unwarranted refusal"]}

    Что проверяет каждый принцип:
      no_refusal    ответ начинается с отказа из REFUSAL_PREFIXES
                        -> "unwarranted refusal"
      no_padding    больше MAX_WORDS слов          -> "padding"
      no_hedging    есть слово из HEDGE_WORDS      -> "hedging"
      plain_number  есть цифра и больше PLAIN_ANSWER_WORDS слов
                        -> "answer buried in prose"

    В настоящей CAI-системе критик — это сама модель, которой дали текст
    принципа. Здесь вместо LLM-вызова рубрика: пайплайн становится
    воспроизводимым, а форма контракта та же.

    Ловушка: сравнивай в нижнем регистре и после strip. "  I Cannot ..." —
    это тот же отказ.
    """
    text = response.strip()
    lowered = text.lower()
    words = text.split()
    problems = []

    pid = principle["id"]
    if pid == "no_refusal":
        if lowered.startswith(REFUSAL_PREFIXES):
            problems.append("unwarranted refusal")
    elif pid == "no_padding":
        if len(words) > MAX_WORDS:
            problems.append("padding")
    elif pid == "no_hedging":
        # пунктуацию по краям слова снимаем, иначе "maybe," не совпадёт
        if any(w.lower().strip(".,!?;:") in HEDGE_WORDS for w in words):
            problems.append("hedging")
    elif pid == "plain_number":
        if any(ch.isdigit() for ch in text) and len(words) > PLAIN_ANSWER_WORDS:
            problems.append("answer buried in prose")

    return {"principle": pid, "problems": problems}


def revise(response, critique_result):
    """Переписать ответ так, чтобы найденная критиком проблема исчезла.

    revise("Paris.", {"principle": "no_refusal", "problems": []})
        ->  "Paris."                        (проблем нет — не трогаем)
    revise("I cannot help. Paris.", {"principle": "no_refusal",
                                     "problems": ["unwarranted refusal"]})
        ->  "Paris."

    Правила починки:
      "unwarranted refusal"      выкинуть первое предложение
      "padding"                  оставить первое предложение, не длиннее
                                 MAX_WORDS слов
      "hedging"                  выкинуть слова из HEDGE_WORDS
      "answer buried in prose"   оставить только последнее число

    Контракт, который проверяют тесты: после revise повторная critique по
    тому же принципу обязана вернуть пустой список проблем. Цикл
    «критика -> правка -> критика» должен сходиться за один проход, иначе
    в проде он зациклится.

    Если починить нечем (нечего выкидывать, числа нет), возвращай исходный
    ответ, а не пустую строку: потерять ответ хуже, чем оставить его плохим.

    В настоящей CAI это второй промпт к модели: «вот критика, перепиши».
    """
    problems = critique_result["problems"]
    if not problems:
        return response

    text = response.strip()
    problem = problems[0]

    if problem == "unwarranted refusal":
        head, sep, tail = text.partition(".")
        return tail.strip() if sep and tail.strip() else text

    if problem == "padding":
        first = text.partition(".")[0].strip() or text
        return " ".join(first.split()[:MAX_WORDS]) + "."

    if problem == "hedging":
        kept = [w for w in text.split() if w.lower().strip(".,!?;:") not in HEDGE_WORDS]
        return " ".join(kept) if kept else text

    if problem == "answer buried in prose":
        numbers = re.findall(r"-?\d+", text)
        return numbers[-1] + "." if numbers else text

    return text


def reward_math(prompt, response):
    """Правило-верификатор для арифметики: 1.0 за верный ответ, иначе 0.0.

    reward_math("What is 15 * 7?", "The answer is 105.")  ->  1.0
    reward_math("What is 15 * 7?", "I think it is 104.")  ->  0.0
    reward_math("Who wrote Hamlet?", "Shakespeare")       ->  0.0

    Из промпта достаём выражение вида "<число> <оператор> <число>",
    оператор из + - *. Из ответа берём ПОСЛЕДНЕЕ целое число: модель часто
    рассуждает вслух ("10*7=70, 5*7=35, итого 105"), и правильный ответ
    стоит в конце.

    Промпт без выражения или ответ без чисел — награда 0.0, а не исключение:
    верификатор обязан работать на любом входе, включая мусор.

    Осторожно: считать выражение через eval() нельзя. Промпт приходит извне,
    а eval исполнит в нём что угодно. Разбирай регуляркой.

    Это ORM — outcome reward model. Никаких обученных весов, никакой
    разметки: ровно так тренировали DeepSeek-R1-Zero.
    """
    match = re.search(r"(-?\d+)\s*([+\-*])\s*(-?\d+)", prompt)
    if not match:
        return 0.0
    left, op, right = int(match.group(1)), match.group(2), int(match.group(3))
    expected = {"+": left + right, "-": left - right, "*": left * right}[op]

    numbers = re.findall(r"-?\d+", response)
    if not numbers:
        return 0.0
    return 1.0 if int(numbers[-1]) == expected else 0.0


def reward_format(response):
    """Правило-верификатор формата: ответ обёрнут в <answer>...</answer>?

    reward_format("<answer>105</answer>")  ->  1.0
    reward_format("105")                   ->  0.0

    Вторая награда R1-Zero, кроме правильности. Она копеечная (регулярка),
    но без неё модель выдаёт правильные ответы в произвольной обёртке,
    и парсер на выходе ломается.

    Ловушка: тег может закрываться не сразу и содержать переносы строк.
    """
    return 1.0 if re.search(r"<answer>.*?</answer>", response, re.DOTALL) else 0.0


def total_reward(prompt, response, format_weight=0.1):
    """Суммарная награда: правильность плюс маленькая доплата за формат.

    total_reward("What is 3 + 4?", "<answer>7</answer>")  ->  1.1
    total_reward("What is 3 + 4?", "7")                   ->  1.0
    total_reward("What is 3 + 4?", "<answer>8</answer>")  ->  0.1

    format_weight маленький СПЕЦИАЛЬНО. Сделай его равным 1.0 — и модель
    научится ставить теги вокруг неверных ответов: за формат дают столько
    же, сколько за правильность. Это reward hacking в две строки.
    """
    return reward_math(prompt, response) + format_weight * reward_format(response)


def group_relative_advantage(rewards):
    """Преимущества GRPO: z-оценка награды внутри группы ответов.

    group_relative_advantage([0.0, 1.0])       ->  [-1.0, 1.0]
    group_relative_advantage([1.0, 1.0, 1.0])  ->  [0.0, 0.0, 0.0]
    group_relative_advantage([])               ->  []

    Формула: A_i = (r_i - mean(r)) / std(r), std — популяционная (делим на N,
    а не на N-1: группа это вся выборка, а не подвыборка).

    Если весь разброс схлопнулся (std почти ноль), верни нули. Это не
    аварийный случай, а сигнал: промпт либо решается всегда, либо не решается
    никогда, градиента из него не выжать, шаг надо пропустить.

    Здесь и вся суть GRPO против PPO: базовой линией служит сама группа,
    отдельная value-сеть размером с политику не нужна.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def grpo_step(policy_logprobs, ref_logprobs, advantages, beta=0.01, clip_eps=0.2):
    """Один шаг GRPO: обрезанный surrogate плюс KL-штраф к reference.

    Вернуть словарь с ключами "policy_loss", "kl", "total_loss",
    "mean_ratio", "clipped_fraction".

    grpo_step([0.0], [0.0], [1.0])  ->  ratio 1.0, kl 0.0, policy_loss -1.0
    grpo_step([1.0], [0.0], [1.0])  ->  ratio e, но surrogate обрезан до 1.2

    Считаем по шагам:
      ratio_i         = exp(policy_logprob_i - ref_logprob_i)
      surrogate_i     = min(ratio_i * A_i, clip(ratio_i, 1-eps, 1+eps) * A_i)
      policy_loss     = -mean(surrogate)
      kl              = mean(policy_logprob - ref_logprob)
      total_loss      = policy_loss + beta * kl
      clipped_fraction = доля i, у которых ratio вышел за [1-eps, 1+eps]

    Это тот же clipped surrogate, что и в PPO (урок 07). Разница ровно одна:
    advantages пришли из z-оценки группы, а не из обученной value-сети.

    Ловушка: работаем с ЛОГАРИФМАМИ вероятностей, поэтому отношение — это
    exp разности, а не частное. Поделишь логарифмы — получишь молча
    неправильное число, без всякого исключения.
    """
    n = len(advantages)
    if n == 0:
        return {
            "policy_loss": 0.0, "kl": 0.0, "total_loss": 0.0,
            "mean_ratio": 0.0, "clipped_fraction": 0.0,
        }

    lo, hi = 1.0 - clip_eps, 1.0 + clip_eps
    surrogate_sum = 0.0
    ratio_sum = 0.0
    kl_sum = 0.0
    clipped = 0

    for pi, ref, adv in zip(policy_logprobs, ref_logprobs, advantages):
        ratio = math.exp(pi - ref)
        clamped = min(max(ratio, lo), hi)
        surrogate_sum += min(ratio * adv, clamped * adv)
        ratio_sum += ratio
        # оценка KL по сэмплам из политики: E_pi[log pi - log pi_ref].
        # На одном сэмпле она может быть отрицательной, в среднем — нет
        kl_sum += pi - ref
        if ratio < lo or ratio > hi:
            clipped += 1

    policy_loss = -surrogate_sum / n
    kl = kl_sum / n
    return {
        "policy_loss": policy_loss,
        "kl": kl,
        "total_loss": policy_loss + beta * kl,
        "mean_ratio": ratio_sum / n,
        "clipped_fraction": clipped / n,
    }


def self_improvement_round(prompts, sampler, rng, group_size=8):
    """Один раунд самоулучшения: сэмплируем группу, оцениваем правилом.

    sampler вызывается как sampler(prompt, rng) и возвращает строку-ответ.

    Вернуть {"per_prompt": [...], "overall_mean": <среднее mean_reward>}.
    Каждая запись per_prompt: "prompt", "mean_reward", "best_reward",
    "std_reward", "best_response", "advantages".

    self_improvement_round(["What is 3 + 4?"], lambda p, r: "7", rng, 4)
        ->  mean_reward 1.0, std_reward 0.0, advantages [0, 0, 0, 0]

    rng передаётся ЯВНО и уходит в sampler. Обращаться к глобальному random
    здесь нельзя: раунд обязан воспроизводиться от того же seed, иначе
    отладить расходящееся обучение невозможно.

    Три кривые — это продовый health check пайплайна:
      mean_reward растёт по раундам   — обучение идёт;
      std_reward держится выше нуля   — разнообразие живо;
      std_reward упал в ноль          — mode collapse, раунды пора
                                        останавливать.
    """
    per_prompt = []
    for prompt in prompts:
        responses = [sampler(prompt, rng) for _ in range(group_size)]
        rewards = [total_reward(prompt, r) for r in responses]
        # advantages считаем ДО поиска лучшего: обе величины смотрят на одну
        # и ту же группу, пересэмплировать между ними нельзя
        advantages = group_relative_advantage(rewards)

        mean = sum(rewards) / len(rewards)
        var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        best_index = max(range(len(rewards)), key=lambda i: rewards[i])

        per_prompt.append({
            "prompt": prompt,
            "mean_reward": mean,
            "best_reward": rewards[best_index],
            "std_reward": math.sqrt(var),
            "best_response": responses[best_index],
            "advantages": advantages,
        })

    overall = (
        sum(m["mean_reward"] for m in per_prompt) / len(per_prompt)
        if per_prompt
        else 0.0
    )
    return {"per_prompt": per_prompt, "overall_mean": overall}
