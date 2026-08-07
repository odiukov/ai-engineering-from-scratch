"""
Constitutional AI и самоулучшение

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l09-constitutional-ai-self-improvement
Разбор:  /check-code p10-l09-constitutional-ai-self-improvement
"""

import math
import re

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def reward_format(response):
    """Правило-верификатор формата: ответ обёрнут в <answer>...</answer>?

    reward_format("<answer>105</answer>")  ->  1.0
    reward_format("105")                   ->  0.0

    Вторая награда R1-Zero, кроме правильности. Она копеечная (регулярка),
    но без неё модель выдаёт правильные ответы в произвольной обёртке,
    и парсер на выходе ломается.

    Ловушка: тег может закрываться не сразу и содержать переносы строк.
    """
    raise NotImplementedError


def total_reward(prompt, response, format_weight=0.1):
    """Суммарная награда: правильность плюс маленькая доплата за формат.

    total_reward("What is 3 + 4?", "<answer>7</answer>")  ->  1.1
    total_reward("What is 3 + 4?", "7")                   ->  1.0
    total_reward("What is 3 + 4?", "<answer>8</answer>")  ->  0.1

    format_weight маленький СПЕЦИАЛЬНО. Сделай его равным 1.0 — и модель
    научится ставить теги вокруг неверных ответов: за формат дают столько
    же, сколько за правильность. Это reward hacking в две строки.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
