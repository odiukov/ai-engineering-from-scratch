"""
Reward modeling и RLHF

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p09-l09-reward-modeling-rlhf
Разбор:  /check-code p09-l09-reward-modeling-rlhf
"""

import math
from collections import Counter


def sigmoid(x):
    """Сигмоида: сжимает любое число в (0, 1).

    sigmoid(0.0)     ->  0.5
    sigmoid(2.0)     ->  0.8807...
    sigmoid(-1000.0) ->  0.0     (без OverflowError!)

    Ловушка: наивная 1/(1+exp(-x)) падает на x = -1000, потому что exp(1000)
    переполняется. Разбери случай x < 0 отдельно через e^x / (1 + e^x) —
    математически то же самое, численно безопасно.

    В Bradley-Terry именно sigmoid превращает разницу наград в вероятность
    предпочтения: P(y_+ лучше y_-) = sigmoid(R(y_+) - R(y_-)).
    """
    raise NotImplementedError


def reward_score(w, tokens):
    """Линейная reward model: R(y) = сумма веса каждого токена ответа.

    reward_score({"clear": 1.5, "rude": -2.0}, ["clear", "clear"])  ->  3.0
    reward_score({"clear": 1.5}, ["unknown"])                       ->  0.0
    reward_score({"clear": 1.5}, [])                                ->  0.0

    w — словарь token -> вес. Токена нет в словаре — его вклад ноль, а не
    KeyError: reward model обязана уметь оценивать невиданный ответ.

    Настоящая RM — это трансформер со скалярной головой. Здесь bag-of-tokens
    вместо трансформера, но контракт тот же: строка на входе, одно число
    на выходе.
    """
    raise NotImplementedError


def bt_loss(w, y_pos, y_neg):
    """Bradley-Terry pairwise loss: -log sigmoid(R(y_pos) - R(y_neg)).

    bt_loss({}, ["a"], ["b"])                     ->  0.6931...  (== log 2)
    bt_loss({"a": 10.0}, ["a"], ["b"])            ->  ~0.0       (уверенно верно)
    bt_loss({"b": 10.0}, ["a"], ["b"])            ->  ~10.0      (уверенно неверно)

    Лосс всегда положителен и зависит ТОЛЬКО от разницы наград, не от их
    абсолютного уровня: прибавь 100 к обеим — ничего не изменится. Поэтому
    шкала reward model не определена, и сравнивать её числа между запусками
    бессмысленно.

    Это `-torch.nn.functional.logsigmoid(r_pos - r_neg)` — ровно то, что
    считает `trl.RewardTrainer`.
    """
    raise NotImplementedError


def bt_gradient(w, y_pos, y_neg):
    """Градиент bt_loss по весам. Словарь token -> dL/dw[token].

    bt_gradient({}, ["a"], ["b"])  ->  {"a": -0.5, "b": 0.5}

    Вывод короткий: p = sigmoid(R_pos - R_neg), тогда
        dL/dw[t] = -(1 - p) * (count_pos[t] - count_neg[t])
    Токен, встречающийся в обоих ответах одинаково часто, получает ровно
    нулевой градиент — предпочтение о нём ничего не говорит.

    В словаре должны быть все токены обоих ответов, даже с нулевым
    градиентом: иначе тест на взаимное сокращение не пройдёт.
    """
    raise NotImplementedError


def train_reward_model(pairs, lr=0.1, epochs=1, w=None):
    """Обучить reward model на парах предпочтений. Вернуть новый словарь весов.

    pairs — список кортежей (y_pos, y_neg): список токенов предпочтённого
    ответа и список токенов отвергнутого.

    train_reward_model([(["good"], ["bad"])], lr=1.0)
        ->  {"good": 0.5, "bad": -0.5}

    Обычный градиентный спуск: w[t] -= lr * grad[t] на каждой паре.
    Знак минуса принципиален — лосс МИНИМИЗИРУЕТСЯ. Перепутаешь, и модель
    выучит предпочтения наоборот, а лосс будет расти.

    w на входе не мутируется: вернуть надо новый словарь.
    """
    raise NotImplementedError


def pairwise_accuracy(w, pairs):
    """Доля пар, где модель дала предпочтённому ответу СТРОГО больший скор.

    pairwise_accuracy({"a": 1.0}, [(["a"], ["b"])])   ->  1.0
    pairwise_accuracy({}, [(["a"], ["b"])])           ->  0.0   (ничья не считается)
    pairwise_accuracy({}, [])                         ->  0.0

    Ничья — это не половина успеха: необученная модель со всеми нулями даёт
    равные скоры и получает 0.0, а не 0.5. Так метрика честно показывает, что
    модель ничего не выучила.

    Основная приёмка stage 2 RLHF. Ниже ~0.65 на holdout — либо данные
    шумные, либо модель мала.
    """
    raise NotImplementedError


def kl_divergence(p, q):
    """KL(p || q) = sum p * log(p / q). В натах.

    kl_divergence([0.5, 0.5], [0.5, 0.5])   ->  0.0
    kl_divergence([1.0, 0.0], [0.5, 0.5])   ->  0.6931...  (== log 2)

    Две ловушки. Нулевые p пропускай: предел p*log p равен нулю. Нулевые q
    подпирай снизу маленькой константой, иначе log(0) уронит расчёт.

    KL несимметрична: KL(p||q) != KL(q||p). В RLHF всегда берут
    KL(pi_theta || pi_ref) — штрафуют политику за то, что она уходит туда,
    где reference её никогда не отправлял бы.
    """
    raise NotImplementedError


def penalized_reward(rm_score, policy_probs, ref_probs, beta=0.1):
    """Итоговая награда RLHF: скор reward model минус beta * KL до reference.

    penalized_reward(1.0, [0.5, 0.5], [0.5, 0.5])        ->  1.0
    penalized_reward(1.0, [1.0, 0.0], [0.5, 0.5], 0.1)   ->  0.9306...
    penalized_reward(1.0, [1.0, 0.0], [0.5, 0.5], 0.0)   ->  1.0

    beta = 0 полностью снимает поводок: политика уедет туда, где reward model
    выдаёт огромные числа на ответах, которых она никогда не видела. Это и
    есть reward hacking, и KL-штраф — единственное, что его сдерживает.

    Это то, что `trl.PPOTrainer` считает под капотом с init_kl_coef.
    """
    raise NotImplementedError
