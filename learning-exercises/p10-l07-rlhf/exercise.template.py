"""
RLHF: reward model и PPO

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p10-l07-rlhf
Разбор:  /check-code p10-l07-rlhf
"""

import math


def sigmoid(x):
    """Сигмоида: сжимает любое число в интервал (0, 1).

    sigmoid(0.0)     ->  0.5
    sigmoid(2.0)     ->  0.8807970779...
    sigmoid(-1000.0) ->  0.0     (без OverflowError)

    Ловушка: наивная 1/(1+exp(-x)) падает на x = -1000, потому что exp(1000)
    переполняется. Разбери случай x < 0 отдельно через e^x / (1 + e^x) —
    математически то же самое, численно безопасно.

    В Bradley-Terry именно sigmoid превращает разницу наград в вероятность
    предпочтения: P(chosen лучше rejected) = sigmoid(R_chosen - R_rejected).
    """
    raise NotImplementedError


def bradley_terry_loss(reward_chosen, reward_rejected):
    """Лосс reward model на одной паре предпочтений.

    Формула: -log(sigmoid(R_chosen - R_rejected)).

    bradley_terry_loss(0.0, 0.0)   ->  0.6931...  (== log 2, модель не знает)
    bradley_terry_loss(5.0, 0.0)   ->  0.0067...  (уверенно и правильно)
    bradley_terry_loss(0.0, 5.0)   ->  5.0067...  (уверенно и неправильно)

    Лосс зависит ТОЛЬКО от разницы наград. Прибавь 100 к обеим — ничего
    не изменится. Поэтому абсолютная шкала reward model не определена, и
    сравнивать её числа между двумя обучениями бессмысленно.

    Ловушка: не пиши -log(sigmoid(...)) в лоб через math.log — при большой
    отрицательной разнице sigmoid обнулится и math.log выбросит ValueError.
    Здесь достаточно math.log(sigmoid(diff)), пока diff > -700; на всякий
    случай посчитай через softplus: -log(sigmoid(d)) == log(1 + e^(-d)).
    """
    raise NotImplementedError


def bradley_terry_grad(reward_chosen, reward_rejected):
    """Аналитический градиент bradley_terry_loss по обеим наградам.

    Вернуть кортеж (dL/dR_chosen, dL/dR_rejected).

    bradley_terry_grad(0.0, 0.0)  ->  (-0.5, 0.5)
    bradley_terry_grad(5.0, 0.0)  ->  (-0.0066..., 0.0066...)

    Вывод: L = -log s, где s = sigmoid(d), d = R_chosen - R_rejected.
    dL/dd = -(1 - s). Дальше цепное правило: dd/dR_chosen = +1,
    dd/dR_rejected = -1.

    Две суммы всегда дают ноль: лосс двигает награды навстречу друг другу,
    поднимая chosen ровно настолько, насколько опускает rejected.
    """
    raise NotImplementedError


def reward_model_accuracy(scored_pairs):
    """Доля пар, где reward model поставила chosen выше rejected.

    scored_pairs — список кортежей (R_chosen, R_rejected).

    reward_model_accuracy([(1.0, 0.0), (2.0, 1.0)])  ->  1.0
    reward_model_accuracy([(1.0, 0.0), (0.0, 1.0)])  ->  0.5
    reward_model_accuracy([])                        ->  0.0

    Ничья (равные награды) считается ошибкой: модель не сделала выбор.

    Случайная модель даёт 0.5. У InstructGPT reward model было около 0.72 —
    это кажется мало, но и согласие между людьми-разметчиками было 0.73.
    Выше согласия людей подняться нечему.
    """
    raise NotImplementedError


def softmax(logits):
    """Логиты -> распределение вероятностей: неотрицательные, сумма 1.

    softmax([0.0, 0.0])       ->  [0.5, 0.5]
    softmax([1.0, 1.0, 1.0])  ->  [0.3333..., 0.3333..., 0.3333...]
    softmax([1000.0, 0.0])    ->  [1.0, 0.0]   (без OverflowError)

    Ловушка: exp(1000) переполняется. Вычти максимум логитов перед exp —
    softmax(x) == softmax(x - c) для любой константы c, результат не меняется,
    а переполнения не будет.
    """
    raise NotImplementedError


def kl_divergence(p, q):
    """KL(p || q) = sum p_i * log(p_i / q_i), в натуральных логарифмах.

    kl_divergence([0.5, 0.5], [0.5, 0.5])  ->  0.0
    kl_divergence([1.0, 0.0], [0.5, 0.5])  ->  0.6931...  (== log 2)

    Свойства, на которых держится KL-штраф в RLHF:
      * KL >= 0 всегда, и ноль только при p == q;
      * KL несимметрична: KL(p||q) != KL(q||p).

    Ловушки: при p_i == 0 слагаемое равно нулю (предел x*log x), а не nan —
    такой член надо пропустить. При q_i == 0 и p_i > 0 KL бесконечна;
    подстрахуйся нижней границей 1e-12 на q_i, как это делают в проде.
    """
    raise NotImplementedError


def rlhf_objective(reward, policy_logits, ref_logits, beta=0.02):
    """Цель RLHF на одном ответе: reward минус KL-штраф за уход от reference.

    Вернуть словарь с ключами "reward", "kl", "penalty", "objective".

    rlhf_objective(1.0, [0.0, 0.0], [0.0, 0.0])        ->  kl 0.0, objective 1.0
    rlhf_objective(1.0, [5.0, 0.0], [0.0, 0.0], 0.5)   ->  objective заметно < 1.0

    Формула: objective = reward - beta * KL(policy || reference), где обе
    политики получаются softmax'ом своих логитов.

    Зачем штраф: reward model обучена на конечном наборе предпочтений и имеет
    слепые зоны. Без штрафа политика их найдёт и начнёт выдавать бессмыслицу
    с высокой наградой — это и есть reward hacking. Штраф говорит: улучшаться
    можно, но становиться другой моделью — нет.
    """
    raise NotImplementedError


def ppo_clipped_loss(ratio, advantage, eps=0.2):
    """Clipped surrogate loss PPO для одного действия.

    Формула: -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).

    ppo_clipped_loss(1.0, 1.0)   ->  -1.0
    ppo_clipped_loss(1.5, 1.0)   ->  -1.2   (обрезано: дальше 1+eps не пускают)
    ppo_clipped_loss(0.5, -1.0)  ->   0.8   (обрезано с другой стороны)

    ratio = pi_new(a|s) / pi_old(a|s), advantage = reward - baseline.

    Смысл обрезки: если один ответ случайно получил огромную награду,
    необрезанный ratio утащил бы политику к нему одним шагом. Обрезка делает
    лосс ПЛОСКИМ за границей [1-eps, 1+eps], и производная там ровно ноль —
    шаг не растёт, сколько ни увеличивай ratio.

    Ловушка асимметрии: плоско становится не всегда. При A > 0 обрезается
    большой ratio, при A < 0 — маленький. В зеркальных случаях (A > 0 и
    маленький ratio, A < 0 и большой) min выбирает НЕобрезанную ветку, и
    градиент продолжает работать. Так и задумано: уводить политику от плохого
    действия PPO не мешает.
    """
    raise NotImplementedError
