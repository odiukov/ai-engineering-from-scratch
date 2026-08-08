"""
Перцептрон — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""


def step(z):
    """Ступенчатая активация перцептрона: 1, если z >= 0, иначе 0.

    step(0.0)   ->  1
    step(2.5)   ->  1
    step(-0.1)  ->  0

    Ноль относится к единице — это соглашение, а не мелочь: граница
    w*x + b = 0 целиком принадлежит положительному классу. Тесты это
    проверяют, потому что `z > 0` вместо `z >= 0` ломает обучение на
    нулевых входах.
    """
    # ровно одно сравнение: перцептрон должен быть дешёвым
    return 1 if z >= 0 else 0


def perceptron_output(weights, bias, inputs):
    """Ответ одного перцептрона: step(скалярное произведение + bias).

    perceptron_output([1.0, 1.0], -1.5, [1, 1])  ->  1
    perceptron_output([1.0, 1.0], -1.5, [1, 0])  ->  0
    perceptron_output([0.0, 0.0], 0.5, [0, 0])   ->  1

    Третий пример важен: bias позволяет нейрону сработать даже на
    полностью нулевом входе. Без bias граница обязана проходить через
    начало координат, и половина задач становится нерешаемой.
    """
    total = sum(w * x for w, x in zip(weights, inputs)) + bias
    return step(total)


def update_once(weights, bias, inputs, target, lr):
    """Одно применение правила обучения. Вернуть (новые веса, новый bias).

    update_once([0.0, 0.0], 0.0, [1, 1], 1, 0.1)  ->  ([0.0, 0.0], 0.0)
    update_once([0.0, 0.0], -0.5, [1, 1], 1, 0.1) ->  ([0.1, 0.1], -0.4)

    Правило: error = target - предсказание, дальше
      w_i += lr * error * x_i,   bias += lr * error.

    Первый пример показывает главное свойство: при верном ответе
    error = 0 и ничего не меняется. Перцептрон учится только на ошибках.
    Знак ошибки задаёт направление: предсказали мало — веса растут.
    """
    error = target - perceptron_output(weights, bias, inputs)
    # вход с нулевой координатой не двигает свой вес — вклада не было
    new_weights = [w + lr * error * x for w, x in zip(weights, inputs)]
    return new_weights, bias + lr * error


def train_perceptron(data, lr=0.1, epochs=100):
    """Обучить перцептрон на data. Вернуть (weights, bias).

    data — список пар (inputs, target), target это 0 или 1.
    Веса и bias стартуют с нулей: обучение обязано быть воспроизводимым.

    and_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    train_perceptron(and_data)  ->  веса, дающие 100% точность

    Проход по всем примерам — одна эпоха. Если за целую эпоху не было ни
    одной ошибки, дальше учиться нечему: выходим досрочно. Без этого
    выхода функция всегда крутит все epochs итераций впустую.
    """
    n_inputs = len(data[0][0])
    weights = [0.0] * n_inputs
    bias = 0.0
    for _ in range(epochs):
        mistakes = 0
        for inputs, target in data:
            if perceptron_output(weights, bias, inputs) != target:
                mistakes += 1
            weights, bias = update_once(weights, bias, inputs, target, lr)
        if mistakes == 0:
            break
    return weights, bias


def accuracy(weights, bias, data):
    """Доля верных ответов на наборе: число от 0.0 до 1.0.

    and_data = [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)]
    accuracy([0.0, 0.0], 0.0, and_data)  ->  0.25

    В примере перцептрон всегда отвечает 1 (нулевые веса, нулевой bias,
    а step(0) = 1) и угадывает только одну строку из четырёх.
    """
    correct = sum(
        1 for inputs, target in data if perceptron_output(weights, bias, inputs) == target
    )
    return correct / len(data)


def perceptron_converged(data, epochs=200):
    """Сошёлся ли перцептрон к 100% точности за заданное число эпох.

    perceptron_converged([([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)])  ->  True
    perceptron_converged([([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)])  ->  False

    True доказывает, что найден разделитель. False означает только таймаут:
    набор может быть неразделимым, а может быть разделимым с очень маленьким
    запасом и требовать больше эпох. Теорема сходимости обещает конечное число
    шагов для разделимых данных, но заранее не делает epochs достаточным.

    Второй пример — XOR: он действительно не разделяется одной прямой, но
    один лишь этот ограниченный прогон не служит доказательством.
    """
    weights, bias = train_perceptron(data, epochs=epochs)
    return accuracy(weights, bias, data) == 1.0


def xor_network(x1, x2):
    """XOR из трёх перцептронов с выставленными руками весами.

    xor_network(0, 0)  ->  0
    xor_network(0, 1)  ->  1
    xor_network(1, 1)  ->  0

    Схема: XOR = (x1 OR x2) AND NOT(x1 AND x2).
      OR   : веса [1, 1],   bias -0.5
      NAND : веса [-1, -1], bias 1.5
      AND  : веса [1, 1],   bias -1.5

    Один слой XOR не берёт, два — берут. Отсюда и растут все глубокие
    сети: каждый следующий слой работает с признаками, которые построил
    предыдущий, а не с сырым входом.
    """
    or_out = perceptron_output([1.0, 1.0], -0.5, [x1, x2])
    nand_out = perceptron_output([-1.0, -1.0], 1.5, [x1, x2])
    return perceptron_output([1.0, 1.0], -1.5, [or_out, nand_out])
