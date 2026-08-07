"""
Few-shot, chain-of-thought и голосование

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p11-l02-few-shot-cot
Разбор:  /check-code p11-l02-few-shot-cot
"""

import re
from collections import Counter

COT_SYSTEM = (
    "You are a math problem solver. "
    "For each problem, show your step-by-step reasoning, "
    "then give the final numerical answer on the last line "
    "in the format: 'The answer is [number]'."
)


def format_example(example, with_reasoning=True):
    """Один few-shot пример в текстовом виде.

    example — dict с ключами question, reasoning, answer.

    format_example({"question": "2+2?", "reasoning": "2 and 2 make 4.",
                    "answer": "4"})
        ->  "Q: 2+2?\\nA: 2 and 2 make 4. The answer is 4."
    тот же пример с with_reasoning=False
        ->  "Q: 2+2?\\nA: The answer is 4."

    with_reasoning=False даёт обычный few-shot, True — few-shot CoT. Разница
    между ними на GSM8K это несколько процентных пунктов точности.
    """
    raise NotImplementedError


def build_cot_prompt(question, examples, num_examples=3, with_reasoning=True):
    """Собрать (system, user) для few-shot CoT промпта.

    Примеры идут первыми, разделённые пустой строкой, затем целевой вопрос
    и оборванная строка "A:" — модель продолжит с неё.

    build_cot_prompt("5+5?", examples, num_examples=1)[1]
        ->  "Q: 2+2?\\nA: 2 and 2 make 4. The answer is 4.\\n\\nQ: 5+5?\\nA:"
    build_cot_prompt("5+5?", [])[1]  ->  "Q: 5+5?\\nA:"   (zero-shot)

    system — это COT_SYSTEM. Берётся ровно num_examples первых примеров;
    если их меньше, берутся все, что есть.

    Соответствует FewShotPromptTemplate из LangChain.
    """
    raise NotImplementedError


def extract_answer(text):
    """Достать число из строки "The answer is X". Вернуть float или None.

    extract_answer("... The answer is 18.")        ->  18.0
    extract_answer("The answer is -3.5")           ->  -3.5
    extract_answer("I have no idea")               ->  None

    Если фраза встречается несколько раз (модель порассуждала и передумала),
    берём ПОСЛЕДНЮЮ: рассуждение идёт сверху вниз, вывод внизу.

    Числа с запятыми ("The answer is 1,200") встречаются постоянно —
    запятые надо выкинуть, иначе получишь 1.0 вместо 1200.0.
    """
    raise NotImplementedError


def select_examples(query, examples, k):
    """k примеров, наиболее похожих на запрос по пересечению слов.

    Похожесть — коэффициент Жаккара по множествам слов в нижнем регистре:
    |общие| / |объединение|.

    select_examples("apples cost", exs, 2)  ->  два примера про яблоки

    При равной похожести побеждает тот, кто раньше в списке — иначе выбор
    зависит от порядка словаря и результат перестаёт воспроизводиться.

    Соответствует SemanticSimilarityExampleSelector, только вместо
    эмбеддингов — пересечение слов.
    """
    raise NotImplementedError


def select_diverse_examples(examples, k):
    """k примеров с максимальным покрытием разных ответов (label diversity).

    Сначала берём по одному примеру на каждый ещё не встреченный answer,
    в порядке появления. Если набралось меньше k — добираем оставшимися
    примерами, тоже по порядку.

    select_diverse_examples([A(ans=1), B(ans=1), C(ans=2)], 2)
        ->  [A, C]   (а не [A, B])

    Зачем: если все примеры показывают один класс, модель решит, что ответ
    всегда такой. Для классификации нужен хотя бы один пример на метку.
    """
    raise NotImplementedError


def majority_vote(answers):
    """Победитель голосования и его доля. Возвращает (answer, confidence).

    majority_vote([24.0, 24.0, 27.0])  ->  (24.0, 0.6666666666666666)
    majority_vote([])                  ->  (None, 0.0)

    При равенстве голосов выигрывает тот, кто встретился РАНЬШЕ, — иначе
    два прогона одного и того же набора дадут разные ответы.

    confidence — доля голосов победителя, а не абсолютное число: по ней
    удобно решать, эскалировать ли задачу к дорогому методу.
    """
    raise NotImplementedError


def self_consistency(samples):
    """Self-consistency: N текстов рассуждений -> (answer, confidence, votes).

    samples — список ответов модели, полученных при temperature > 0.

    self_consistency(["...The answer is 24.", "...The answer is 24.",
                      "...The answer is 27."])
        ->  (24.0, 0.666..., Counter({24.0: 2, 27.0: 1}))

    Тексты, из которых ответ не достаётся, в голосовании не участвуют
    вообще — и в знаменателе confidence их тоже быть не должно.

    Соответствует dspy.majority.
    """
    raise NotImplementedError


def tree_of_thought(root, expand, evaluate, breadth=3, depth=2, beam=2):
    """Поиск в дереве рассуждений. Возвращает (лучший путь, его оценка).

    root      — начальный узел (например, текст задачи);
    expand    — expand(path) -> список узлов-продолжений;
    evaluate  — evaluate(path) -> float, чем больше, тем перспективнее;
    breadth   — сколько продолжений брать от одного узла;
    depth     — сколько уровней раскрывать;
    beam      — сколько лучших путей нести на следующий уровень.

    path — это список узлов от корня. Начинаем с [ [root] ], на каждом
    уровне раскрываем beam лучших путей, каждый на breadth продолжений,
    оцениваем и снова оставляем beam лучших.

    tree_of_thought("", lambda p: ["a", "b"], lambda p: len("".join(p)))
        ->  путь из depth+1 элементов, самый длинный по сумме

    depth=0 значит «не раскрывать»: вернётся ([root], evaluate([root])).

    Дорого: узлов до beam * breadth * depth, и каждый в проде — вызов LLM.
    Поэтому ToT берут только там, где линейный CoT проигрывает.
    """
    raise NotImplementedError
