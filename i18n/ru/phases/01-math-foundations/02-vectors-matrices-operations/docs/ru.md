<!-- i18n:manual -->
# Векторы, матрицы и операции

> Любая нейросеть — это умножение матриц с дополнительными шагами.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Построить класс Matrix с поэлементными операциями, умножением матриц, транспонированием, определителем и обратной матрицей
- Отличать поэлементное умножение от матричного и понимать, когда применяется каждое
- Реализовать один полносвязный слой нейросети (`relu(W @ x + b)`), используя только свой класс Matrix
- Объяснить правила broadcasting и то, как во фреймворках работает прибавление смещения (bias)

> 🎒 **На пальцах.** Прошлый урок был про смысл. Этот — про руки: вы напишете калькулятор матриц сами, а в конце соберёте из него настоящий слой нейросети. Три строки кода, и это уже AI.

## The Problem

Вы хотите собрать нейросеть. Открываете код и видите:

```
output = activation(weights @ input + bias)
```

Этот `@` — умножение матриц. `weights` — матрица. `input` — вектор. Если вы не знаете, что делают эти операции, строка выглядит магией. Если знаете — это весь прямой проход слоя в три операции.

Каждое изображение, которое обрабатывает модель, — матрица значений пикселей. Каждый word embedding — вектор. Каждый слой каждой нейросети — матричное преобразование. Строить AI-системы без свободного владения матричными операциями невозможно ровно так же, как писать код без понимания переменных.

Этот урок даёт такое владение с нуля.

> 🎒 **На пальцах.** Сравните с кулинарией. `weights` — рецепт, `input` — продукты, `@` — «приготовить по рецепту», `bias` — щепотка соли в конце, `activation` — «попробовать и выбросить, если гадость». Одна строка кода = одно блюдо.

## The Concept

### Vectors: ordered lists of numbers

Вектор — это список чисел с направлением и длиной. В AI векторы представляют точки данных, признаки или параметры.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Двумерный вектор `[3, 4]` указывает на координаты (3, 4) на плоскости. Его длина равна 5 (тот самый треугольник 3-4-5).

> 🎒 **На пальцах.** Треугольник 3-4-5 знаком со школы: катеты 3 и 4, гипотенуза 5. Проверяем: 3² + 4² = 9 + 16 = 25, корень из 25 = 5. Ровно так компьютер меряет длину любого вектора — хоть из 2 чисел, хоть из 768.

### Matrices: grids of numbers

Матрица — это двумерная таблица. Строки и столбцы. Матрица m x n имеет m строк и n столбцов.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

В нейросетях матрицы весов преобразуют входные векторы в выходные. Слой с 784 входами и 128 выходами использует матрицу весов 128x784.

> 🎒 **На пальцах.** Откуда 784: картинка рукописной цифры 28×28 пикселей, 28 × 28 = 784 числа. Матрица 128x784 сжимает эти 784 числа до 128 — как если бы вы пересказали длинное сочинение 128 короткими пунктами.

### Why shapes matter

У умножения матриц строгое правило: `(m x n) @ (n x p) = (m x p)`. Внутренние размерности обязаны совпадать.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Если вы получили ошибку несовпадения размерностей в PyTorch — вот почему.

> 🎒 **На пальцах.** Как детали LEGO: соединяются только совпадающими креплениями. Внутренние числа — крепления (784 и 784, совпали, стыкуемся), внешние — что получилось на выходе (128 x 1). Хотите проверить себя: (3 x 5) @ (5 x 2) = (3 x 2), а (3 x 5) @ (2 x 5) не соединится вообще.

### The operations map

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Сложение | Поэлементно объединить | Прибавление bias к выходу |
| Умножение на скаляр | Масштабировать каждый элемент | Learning rate * градиенты |
| Умножение матриц | Преобразовать векторы | Прямой проход слоя |
| Транспонирование | Поменять строки и столбцы местами | Обратное распространение |
| Определитель | Сводка одним числом | Проверка обратимости |
| Обратная матрица | Отменить преобразование | Решение линейных систем |
| Единичная матрица | Матрица «ничего не делать» | Инициализация, residual connections |

> 🎒 **На пальцах.** Читайте таблицу как список инструментов в ящике. «Умножение на скаляр» — это буквально строка `learning rate * градиенты`: при lr = 0.01 и градиенте 5 шаг будет 0.05, крошечный. А единичная матрица — это «умножить на 1»: для размера 2 это [[1, 0], [0, 1]], и A @ I вернёт ту же самую A, ничего не изменив.

### Element-wise vs matrix multiplication

На этом различии новички спотыкаются постоянно.

Поэлементно: перемножаем совпадающие позиции. Обе матрицы должны быть одной формы.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Матричное умножение: скалярные произведения строк на столбцы. Внутренние размерности должны совпадать.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Разные операции, разные результаты, разные правила.

> 🎒 **На пальцах.** Поэлементно — как складывать два одинаковых списка покупок: яблоки к яблокам, хлеб к хлебу. Матрично — как считать стоимость чека: берёте строку «сколько чего купили» и столбец «сколько что стоит», перемножаете попарно и складываете. Первое просто, второе даёт другое число — 19 вместо 5. Не перепутайте: в PyTorch это `*` и `@`, и обе строки запустятся без ошибки, просто ответ будет неверный.

### Broadcasting

Когда вы прибавляете вектор смещения к матрице выходов, формы не совпадают. Broadcasting растягивает меньший массив, чтобы подошёл.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Любой современный фреймворк делает это автоматически. Понимание broadcasting спасает от недоумения, когда формы кажутся неправильными, а код работает.

> 🎒 **На пальцах.** В классе 30 учеников, всем добавляют одинаковые +5 баллов за олимпиаду. Никто не пишет «+5» тридцать раз — говорят «всем плюс пять». Broadcasting делает ровно это: одну строчку [10, 20, 30] мысленно копирует на все ряды.

```figure
vector-projection
```

## Build It

### Step 1: Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

> 🎒 **На пальцах.** Проверим два последних метода руками на `Vector([3, 4])`. Метод `dot` с самим собой даёт 3×3 + 4×4 = 25, а `magnitude` — корень из 25, то есть 5. Скалярное произведение отвечает на вопрос «насколько два вектора смотрят в одну сторону», а magnitude — просто длина стрелки, тот же треугольник 3-4-5.

### Step 2: Matrix class with core operations

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

> 🎒 **На пальцах.** Самый хитрый метод здесь — `determinant`: для 2x2 он считает по формуле в одну строку, а для больших матриц вызывает сам себя на минорах, каждый раз отрезая один столбец и первую строку. Матрица 4x4 развернётся в четыре 3x3, каждая из них — в три 2x2, итого 12 маленьких определителей. Растёт это как факториал, поэтому так определители считают только в учебниках, а NumPy внутри использует совсем другой алгоритм.

### Step 3: See it work

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: Connect to neural networks

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Это один полносвязный слой: `output = relu(W @ x + b)`. Каждый полносвязный слой в каждой нейросети делает ровно это.

> 🎒 **На пальцах.** Посчитайте определитель матрицы `[[1, 2], [3, 4]]` на бумаге: 1×4 − 2×3 = 4 − 6 = −2. Не ноль — значит обратная матрица существует, `inverse_2x2()` не упадёт. А функция `relu_matrix` — самое простое правило на свете: «отрицательное заменить нулём, положительное оставить». Всё. Ради этого правила и существует слово «активация».

## Use It

NumPy делает всё вышеперечисленное в меньшем количестве строк и на порядки быстрее.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Оператор `@` в Python вызывает `__matmul__`. NumPy реализует его через оптимизированные BLAS-процедуры на C и Fortran. Та же математика, в 100 раз быстрее.

Broadcasting в NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy автоматически растягивает одномерный bias по обеим строкам. Именно так работает прибавление смещения в любом фреймворке нейросетей.

> 🎒 **На пальцах.** Обратите внимание: `A * B` и `A @ B` в NumPy дают разное. Первое — [[5, 12], [21, 32]], второе — [[19, 22], [43, 50]]. Оба запустятся без ошибки. Это одна из самых частых ошибок новичка: код работает, модель не учится, а виноват один символ.

## Ship It

Этот урок производит промпт для обучения матричным операциям через геометрическую интуицию. Смотрите `outputs/prompt-matrix-operations.md`.

Класс Matrix, построенный здесь, — фундамент мини-фреймворка нейросетей, который мы соберём в Phase 3, Lesson 10.

## Exercises

1. **Verify the inverse.** Перемножьте `A @ A.inverse_2x2()` и убедитесь, что получилась единичная матрица. Попробуйте с тремя разными матрицами 2x2. Что произойдёт, если определитель равен нулю?

2. **Implement 3x3 inverse.** Расширьте класс Matrix, чтобы он считал обратные матрицы 3x3 методом присоединённой матрицы. Сверьте результат с `np.linalg.inv` из NumPy.

3. **Build a two-layer network.** Используя только свой класс Matrix (без NumPy), создайте двухслойную нейросеть: вход (3) -> скрытый слой (4) -> выход (2). Инициализируйте случайные веса, прогоните прямой проход и проверьте, что все формы корректны.

> 🎒 **На пальцах.** Подсказка к третьему заданию: формы должны стыковаться как LEGO. Первая матрица весов — (4 x 3), она превращает 3 числа в 4. Вторая — (2 x 4), она превращает 4 числа в 2. Проверка: (2 x 4) @ (4 x 3) @ (3 x 1) = (2 x 1). Если внутренние числа не совпали, вы ошиблись в порядке строк и столбцов.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | «Стрелочка» | Упорядоченный список чисел. В AI: точка в многомерном пространстве. |
| Matrix | «Таблица чисел» | Линейное преобразование. Переводит векторы из одного пространства в другое. |
| Matrix multiply | «Просто перемножить числа» | Скалярные произведения каждой строки первой матрицы на каждый столбец второй. Порядок важен. |
| Transpose | «Перевернуть» | Поменять строки и столбцы местами. Превращает матрицу m x n в n x m. Ключевая операция в обратном распространении. |
| Determinant | «Какое-то число из матрицы» | Мера того, во сколько раз матрица растягивает площадь (2D) или объём (3D). Ноль означает, что преобразование схлопывает измерение. |
| Inverse | «Отменить матрицу» | Матрица, обращающая преобразование вспять. Существует, только если определитель не ноль. |
| Identity matrix | «Скучная матрица» | Матричный аналог умножения на 1. Используется в residual connections (ResNets). |
| Broadcasting | «Магическая починка форм» | Растягивание меньшего массива под больший повторением вдоль недостающих измерений. |
| Element-wise | «Обычное умножение» | Перемножение совпадающих позиций. Оба массива должны быть одной формы (или совместимы по broadcasting). |

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - визуальная интуиция для каждой операции из этого урока
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - точные правила, которым следует NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - сжатый справочник по линейной алгебре для ML
