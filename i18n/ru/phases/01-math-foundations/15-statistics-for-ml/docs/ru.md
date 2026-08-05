<!-- i18n:manual -->
# Статистика для машинного обучения

> Статистика — то, как вы узнаёте, ваша модель действительно работает или ей просто повезло.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## Learning Objectives

- Вычислять с нуля описательные статистики, корреляции Пирсона и Спирмена, ковариационные матрицы
- Проводить проверку гипотез (t-тест, хи-квадрат) и правильно толковать p-значения и доверительные интервалы
- Строить доверительные интервалы для любой метрики бутстрэпом, без предположений о распределении
- Отличать статистическую значимость от практической через размер эффекта

> 🎒 **На пальцах.** Подбросили монету 10 раз, выпало 6 орлов. Монета кривая? Скорее всего нет — просто повезло. А если 600 из 1000? Вот тогда да. Статистика — это правила, по которым отличают везение от закономерности.

## The Problem

Вы обучили две модели. Модель A даёт 0.87 на тестовом наборе. Модель B — 0.89. Вы выкатываете B. Через три недели продакшн-метрики хуже, чем были. Что случилось?

Модель B на самом деле не была лучше A. Разница в 0.02 была шумом. Тестовый набор был слишком мал, или дисперсия слишком велика, или и то и другое. Вы выкатили случайность, переодетую в улучшение.

Это происходит постоянно. Перетряски лидербордов Kaggle. Статьи, которые не воспроизводятся. A/B-тесты, объявляющие победителя по нескольким сотням примеров. Причина всегда одна: кто-то пропустил статистику.

Статистика даёт инструменты для отделения сигнала от шума. Она говорит, когда разница настоящая, насколько вы должны быть уверены и сколько данных нужно, прежде чем результату можно верить. Каждому ML-конвейеру, каждому сравнению моделей, каждому эксперименту нужна статистика. Без неё вы гадаете.

> 🎒 **На пальцах.** Две контрольные: у Пети 87 баллов, у Васи 89. Кто умнее? Вопрос бессмысленный, пока вы не знаете, из скольких заданий состояла работа. Из тысячи — разница заметная. Из десяти — Вася просто удачно угадал в одном вопросе.

## The Concept

### Descriptive Statistics: Summarizing Your Data

Прежде чем что-то моделировать, надо понять, как выглядят данные. Описательные статистики сжимают набор данных в несколько чисел, схватывающих его форму.

**Measures of central tendency** отвечают на вопрос «где середина?»

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Среднее — точка равновесия. Медиана — отметка «половина здесь, половина там». Когда они расходятся, распределение перекошено. У доходов среднее сильно больше медианы (перекос вправо из-за миллиардеров). У функции потерь при обучении часто среднее меньше медианы (перекос влево из-за лёгких примеров).

**Measures of spread** отвечают на вопрос «насколько разбросаны данные?»

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles** делят отсортированные данные на 100 равных частей. 25-й процентиль (Q1) означает, что 25% значений лежат ниже. 50-й процентиль — медиана. 75-й — Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

В ML процентили важны для задержки инференса, распределений уверенности предсказаний и понимания распределения ошибок. Модель с низкой средней ошибкой, но ужасной ошибкой на P99 может быть непригодна там, где цена ошибки высока.

**Sample vs population statistics.** При вычислении дисперсии по выборке делите на (n-1), а не на n. Это поправка Бесселя. Она компенсирует тот факт, что выборочное среднее — не истинное среднее генеральной совокупности. С n в знаменателе вы систематически занижаете дисперсию. С (n-1) оценка несмещённая.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

На практике: если n велико (тысячи примеров), разница пренебрежима. Если мало (десятки), она важна.

> 🎒 **На пальцах.** Смотрите на пример в коде: [1, 2, 3, 4, 1000]. Среднее — 202, хотя четыре числа из пяти меньше пяти. Медиана — 3, что честно описывает набор. Поэтому про зарплаты всегда спрашивайте медиану, а не среднюю: один директор в отделе поднимает «среднюю» на весь этаж.

### Correlation: How Variables Move Together

Корреляция измеряет силу и направление линейной связи между двумя переменными.

**Pearson correlation coefficient** измеряет линейную связь:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Пирсон предполагает линейность связи и примерную нормальность обеих переменных. Он чувствителен к выбросам. Одна крайняя точка способна утащить r с 0.1 до 0.9.

**Spearman rank correlation** измеряет монотонную связь:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**When to use each:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**The golden rule:** корреляция не означает причинность. Продажи мороженого и утопления скоррелированы, потому что и то и другое растёт летом. Точность вашей модели и число параметров скоррелированы, но добавление параметров не улучшает точность автоматически (см. переобучение).

> 🎒 **На пальцах.** Разница Пирсона и Спирмена на примере. Пирсон спрашивает: «во сколько раз?» Спирмен: «кто на каком месте?» Если каждый следующий ученик выше предыдущего, но на разную величину, Спирмен даст ровно 1 (порядок идеальный), а Пирсон меньше единицы (шаги неровные).

### Covariance Matrix

Ковариация двух переменных измеряет, насколько согласованно они меняются:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Для d признаков ковариационная матрица C имеет размер d x d, где C[i][j] = Cov(признак_i, признак_j). Диагональные элементы C[i][i] — дисперсии каждого признака.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Connection to PCA.** PCA раскладывает ковариационную матрицу по собственным векторам. Собственные векторы — главные компоненты (направления максимальной дисперсии). Собственные значения говорят, сколько дисперсии захватывает каждая компонента. Именно это разбиралось в уроке 10, но теперь видно, почему раскладывать надо ковариационную матрицу: она кодирует все попарные линейные связи в данных.

**Connection to correlation.** Матрица корреляций — это ковариационная матрица стандартизованных переменных (каждая поделена на своё стандартное отклонение). Корреляция нормирует ковариацию, загоняя все значения в [-1, 1].

> 🎒 **На пальцах.** Ковариационная матрица — это таблица «кто с кем дружит» среди признаков. Для трёх признаков это сетка 3x3: по диагонали каждый признак сам с собой (его собственная дисперсия), вне диагонали — попарные связи. Таблица симметрична, потому что «рост связан с весом» и «вес связан с ростом» — одно и то же число. Для 100 признаков получится сетка 100x100, то есть 10 000 клеток, но уникальных чисел там только 5050 — половина плюс диагональ.

### Hypothesis Testing

Проверка гипотез — каркас для принятия решений в условиях неопределённости. Вы начинаете с утверждения, собираете данные и определяете, согласуются ли данные с утверждением.

**The setup:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**The p-value** — вероятность увидеть данные настолько же экстремальные, как наблюдаемые, при условии что H0 верна. Это НЕ вероятность того, что H0 верна. Это самое частое непонимание во всей статистике.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence intervals** дают диапазон правдоподобных значений параметра:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Ширина доверительного интервала говорит о точности оценки. Широкий интервал — высокая неопределённость. Узкий — оценка точна (но не обязательно верна, если данные смещены).

> 🎒 **На пальцах.** Про p-значение по-человечески. Вы подозреваете, что монета кривая. p-значение отвечает не на вопрос «кривая ли она», а на другой: «если бы монета была честной, насколько странным был бы такой результат?» Выпало 9 орлов из 10 — очень странно, p маленькое. Выпало 6 из 10 — совсем не странно, p большое. Это разные вопросы, и путают их постоянно.

### The t-test

t-тест сравнивает средние. У него несколько разновидностей.

**One-sample t-test:** отличается ли среднее генеральной совокупности от предполагаемого значения?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independent):** различаются ли средние двух групп?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** когда измерения приходят парами (одна и та же модель на тех же разбиениях данных):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

В ML парный t-тест обычное дело: вы прогоняете обе модели на одних и тех же 10 фолдах кросс-валидации и сравниваете их результаты попарно.

> 🎒 **На пальцах.** Парный тест — как сравнивать двух учеников по одним и тем же контрольным, а не по разным. Если Вася писал лёгкий вариант, а Петя сложный, сравнивать баллы бессмысленно. Дайте обоим одинаковые задания — и разница станет осмысленной. То же и с моделями: один и тот же тестовый набор для обеих.

### Chi-squared Test

Тест хи-квадрат проверяет, совпадают ли наблюдаемые частоты с ожидаемыми. Полезен для категориальных данных.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for ML Models

A/B-тестирование в ML — не то же самое, что веб-A/B-тесты. У сравнения моделей свои сложности:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**The procedure:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

> 🎒 **На пальцах.** Разберём процедуру на числах. Прогнали обе модели на одних и тех же 10 фолдах, получили 10 пар оценок. Разницы вышли такие: девять раз примерно +0.01 в пользу B и один раз −0.03. Среднее положительное, но разброс огромный — парный t-тест на таких данных, скорее всего, скажет «доказательств недостаточно». Без шагов 5-7 вы бы просто посмотрели на среднее, объявили B победителем и получили ту самую историю из начала урока.

### Statistical Significance vs Practical Significance

Результат может быть статистически значимым и практически бессмысленным. При достаточном количестве данных даже ничтожная разница становится статистически значимой.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** количественно выражает величину разницы независимо от размера выборки:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Всегда сообщайте и p-значение, и размер эффекта. P-значение говорит, реальна ли разница. Размер эффекта — важна ли она.

> 🎒 **На пальцах.** Таблетка достоверно снижает температуру на 0.01 градуса. Статистика говорит «эффект есть, p = 0.001». Здравый смысл говорит «и что?». Разница между «эффект существует» и «эффект имеет значение» — это и есть разница между p-значением и размером эффекта.

### Multiple Comparison Problem

Когда вы проверяете много гипотез, часть окажется «значимой» по случайности. Проверив 20 вещей при alpha = 0.05, ожидайте 1 ложное срабатывание, даже если ничего настоящего нет.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction:** поделить alpha на число тестов.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

В ML это важно, когда вы сравниваете модель по нескольким метрикам, перебираете множество конфигураций гиперпараметров или проверяете на нескольких наборах данных.

> 🎒 **На пальцах.** Купите 20 лотерейных билетов — шанс выиграть хоть на одном заметно выше, чем на одном билете. Здесь то же самое: проверили 20 гипотез — почти наверняка одна «выстрелит» случайно. И если рассказать только о ней, получится очень убедительная, но ложная история.

### Bootstrap Methods

Бутстрэп оценивает выборочное распределение статистики, переотбирая ваши данные с возвращением. Никаких предположений о лежащем в основе распределении не требуется.

**The algorithm:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval (percentile method):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Why bootstrap matters for ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap for model comparison:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Это надёжнее парного t-теста, потому что не делает предположений о распределении.

> 🎒 **На пальцах.** Бутстрэп — как вытаскивать шарики из мешка, каждый раз возвращая шарик обратно. Проделали это тысячу раз, каждый раз считали свою метрику — получили тысячу немного разных ответов. Разброс этой тысячи и показывает, насколько случайна ваша единственная настоящая цифра. Гениально просто и требует только компьютера.

### Parametric vs Non-parametric Tests

**Parametric tests** предполагают конкретное распределение (обычно нормальное):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests** не делают предположений о распределении:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**When to use non-parametric:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**When to use parametric:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

В экспериментах ML у вас обычно малое n (5 или 10 фолдов кросс-валидации), поэтому непараметрические тесты вроде критерия Уилкоксона часто уместнее t-тестов.

### Central Limit Theorem: Practical Implications

ЦПТ говорит: распределение выборочных средних приближается к нормальному с ростом n, независимо от распределения генеральной совокупности.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Why this matters for ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**What CLT does NOT do:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

> 🎒 **На пальцах.** Обратите внимание на формулу: дисперсия среднего равна sigma² / n. Значит разброс падает не пропорционально n, а как корень из n. Хотите вдвое более точную оценку — нужно вчетверо больше данных. Отсюда и все разговоры про «нужно больше данных» — они дорожают квадратично.

### Common Statistical Mistakes in ML Papers

1. **Testing on the training set.** Гарантированное переобучение. Всегда держите отложенные данные, которых модель не видела при обучении.

2. **No confidence intervals.** Одна цифра точности без указания неопределённости делает результат невоспроизводимым и непроверяемым.

3. **Ignoring multiple comparisons.** Проверить 50 конфигураций и отчитаться о лучшей без поправки — значит раздуть долю ложных срабатываний.

4. **Confusing statistical and practical significance.** p = 0.001 на улучшении точности в 0.01% ничего не значит.

5. **Using accuracy on imbalanced data.** 99% точности на наборе, где 99% отрицательного класса, означает, что модель не выучила ничего. Используйте precision, recall, F1 или AUC.

6. **Cherry-picking metrics.** Отчёт только по той метрике, где ваша модель выигрывает. Честная оценка показывает все релевантные метрики.

7. **Leaking information across train/test splits.** Нормализация до разбиения или использование будущих данных для предсказания прошлого.

8. **Small test sets with no variance estimates.** Оценка на 100 примерах с заявлением об улучшении на 2% — это шум, а не сигнал.

9. **Assuming independence when data is not independent.** Медицинские снимки одного пациента, несколько предложений из одного документа. Наблюдения внутри группы скоррелированы.

10. **P-hacking.** Перебирать тесты, подвыборки и критерии исключения, пока не получится p < 0.05. Результат — артефакт перебора.

> 🎒 **На пальцах.** Пятая ошибка самая коварная. Представьте детектор редкой болезни, который всегда отвечает «здоров». На тысяче человек, где болен один, он будет прав в 99.9% случаев. Точность отличная, польза нулевая. Всегда спрашивайте, что было бы, если бы модель просто всегда отвечала одно и то же.

## Building It

Вы реализуете:

1. **Descriptive statistics from scratch** (среднее, медиана, мода, стандартное отклонение, процентили, IQR)
2. **Correlation functions** (Пирсон и Спирмен, вместе с ковариационной матрицей)
3. **Hypothesis tests** (одновыборочный t-тест, двухвыборочный t-тест, хи-квадрат)
4. **Bootstrap confidence intervals** (для любой статистики, без предположений)
5. **A/B test simulator** (сгенерировать данные, проверить, посчитать ошибки первого и второго рода)
6. **Statistical vs practical significance demo** (показать, что при большом n «значимым» становится всё)

Всё с нуля, только `math` и `random`. Ни numpy, ни scipy.

```figure
f3-bootstrap-resample
```

## Key Terms

| Term | Definition |
|---|---|
| Mean | Сумма значений, делённая на количество. Чувствительно к выбросам. |
| Median | Среднее значение отсортированных данных. Устойчива к выбросам. |
| Standard deviation | Корень из дисперсии. Меряет разброс в исходных единицах. |
| Percentile | Значение, ниже которого лежит заданный процент данных. |
| IQR | Межквартильный размах. Q3 минус Q1. Разброс средних 50%. |
| Pearson correlation | Меряет линейную связь двух переменных. Диапазон [-1, 1]. |
| Spearman correlation | Меряет монотонную связь по рангам. |
| Covariance matrix | Матрица попарных ковариаций всех признаков. |
| Null hypothesis | Предположение по умолчанию: эффекта или разницы нет. |
| p-value | Вероятность увидеть такие экстремальные данные при верной нулевой гипотезе. |
| Confidence interval | Диапазон правдоподобных значений параметра при заданном уровне доверия. |
| t-test | Проверяет, значимо ли различаются средние. Использует распределение Стьюдента. |
| Chi-squared test | Проверяет, отличаются ли наблюдаемые частоты от ожидаемых. |
| Effect size | Величина разницы независимо от размера выборки. Часто используют d Коэна. |
| Bonferroni correction | Делит порог значимости на число тестов для контроля ложных срабатываний. |
| Bootstrap | Переотбор с возвращением для оценки выборочных распределений. |
| Type I error | Ложное срабатывание. Отвергли H0, когда она верна. |
| Type II error | Пропуск. Не отвергли H0, когда она неверна. |
| Statistical power | Вероятность правильно отвергнуть ложную H0. Мощность = 1 минус доля ошибок второго рода. |
| Central limit theorem | Выборочные средние сходятся к нормальному распределению с ростом выборки. |
| Parametric test | Предполагает конкретное распределение данных (обычно нормальное). |
| Non-parametric test | Не делает предположений о распределении. Работает с рангами или знаками. |
