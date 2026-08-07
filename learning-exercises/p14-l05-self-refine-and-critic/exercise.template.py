"""
Self-Refine и CRITIC

Реализуй функции ниже. Заготовки бросают NotImplementedError — удали
строку raise и напиши код.

Правила:
  * сторонние библиотеки не использовать, только стандартная (math, random).
    Смысл упражнения — собрать руками.
  * файл test_exercise.py не трогай.
  * эталон лежит в solution.py — открывай ПОСЛЕ своих зелёных тестов.

Запуск:  ./learning-exercises/watch.sh p14-l05-self-refine-and-critic
Разбор:  /check-code p14-l05-self-refine-and-critic
"""


def format_issues(output, max_bullets=3, max_chars=60):
    """Список замечаний по форме черновика. Пустой список = форма в порядке.

    format_issues("- a\\n- b\\n- c")  ->  []
    format_issues("- a\\n- b")        ->  ['expected 3 bullets, got 2']

    Черновик считается правильным, если это ровно max_bullets непустых
    строк, каждая начинается с "- " и не длиннее max_chars символов.

    Ловушка: пустые строки в конце текста не должны считаться пунктами,
    иначе "- a\\n- b\\n- c\\n" внезапно окажется из четырёх пунктов.

    В AI это самый дешёвый вид верификатора: чистая функция без вызова
    модели. Именно такие проверки OpenAI Agents SDK вешает как output
    guardrail.
    """
    raise NotImplementedError


def self_feedback(output):
    """Self-Refine: модель критикует сама себя. Вернуть (critique, ok).

    self_feedback("- a\\n- b\\n- c")  ->  ('no issues', True)
    self_feedback("- a\\n- b")        ->  ('expected 3 bullets, got 2', False)

    Самокритик видит только форму: фактов он не проверяет, потому что
    сверять их ему не с чем. Это ровно та слабость, из-за которой
    появился CRITIC — уверенно звучащая галлюцинация проходит
    самопроверку.

    Несколько замечаний склеиваются через "; " в одну строку критики.
    """
    raise NotImplementedError


def external_verify(output, wrong_facts):
    """CRITIC: верификатор с внешним заземлением. Вернуть (critique, ok).

    wrong_facts — последовательность кортежей подстрок. Факт нарушен,
    когда ВСЕ подстроки кортежа встречаются в тексте (регистр не важен).

    external_verify("- a\\n- b\\n- c", [("paris", "germany")])
        ->  ('verifier: ok', True)
    external_verify("- Paris is in Germany\\n- b\\n- c", [("paris", "germany")])
        ->  ('verifier: contradicts reference: paris + germany', False)

    Проверка формы тоже остаётся: CRITIC не заменяет self-critique, а
    добавляет к нему заземление. Если внешних инструментов нет, CRITIC
    вырождается в Self-Refine — это прямо сказано в статье.

    Ловушка: сравнивай в нижнем регистре, иначе "Germany" в черновике не
    совпадёт с "germany" в справочнике.
    """
    raise NotImplementedError


def refine_prompt(task, history):
    """Промпт шага refine: задача плюс ВСЯ история попыток и критики.

    history — список словарей с ключами iteration, output, critique.

    refine_prompt("bullets", [])
        ->  'TASK: bullets\\nWrite the first draft.'

    refine_prompt("bullets", [{"iteration": 1, "output": "- a",
                               "critique": "too short"}])
        ->  'TASK: bullets\\nATTEMPT 1:\\n- a\\nCRITIQUE 1: too short\\n'
            'Write an improved version that fixes every critique above.'

    История здесь — не украшение. Абляция в статье Self-Refine показывает:
    выкинь прошлые попытки из промпта, и модель начинает ходить по кругу,
    повторяя уже отвергнутый вариант.
    """
    raise NotImplementedError


def scripted_generate(prompt, script):
    """Заглушка модели: детерминированная функция от промпта.

    script — последовательность пар (keyword, draft). Возвращается draft
    ПЕРВОЙ пары, чей keyword встретился в промпте (регистр не важен).
    Пустой keyword совпадает всегда, поэтому его ставят последним как
    вариант по умолчанию.

    s = (("germany", "fixed"), ("", "first draft"))
    scripted_generate("TASK: facts", s)              ->  'first draft'
    scripted_generate("CRITIQUE 1: germany", s)      ->  'fixed'

    Пустой script — это ValueError: сгенерировать нечего.

    Настоящий вызов модели тут был бы client.messages.create(...). Нам
    нужна воспроизводимость: тест обязан проверять петлю, а не то, угадала
    ли модель ответ.
    """
    raise NotImplementedError


def should_stop(iteration, verified, repeated, max_iterations):
    """Составное условие останова. Вернуть (stop, reason).

    should_stop(1, True, False, 4)   ->  (True, 'verified')
    should_stop(2, False, True, 4)   ->  (True, 'stalled')
    should_stop(4, False, False, 4)  ->  (True, 'budget')
    should_stop(1, False, False, 4)  ->  (False, 'continue')

    Порядок проверок важен: пройденная верификация бьёт исчерпанный
    бюджет, иначе последняя удачная итерация будет отмечена как 'budget'.

    repeated значит «модель повторила уже отвергнутый черновик». Без этой
    ветки петля будет крутиться до упора бюджета, каждый раз получая ту же
    критику. Одноусловный останов — самая частая ошибка в проде.
    """
    raise NotImplementedError


def refine_loop(task, script, verify, max_iterations=4):
    """Петля Self-Refine/CRITIC целиком. Вернуть список попыток.

    verify — функция output -> (critique, ok). Подставь self_feedback для
    Self-Refine или лямбду вокруг external_verify для CRITIC.

    Каждая попытка — словарь с ключами:
      iteration, prompt, output, critique, verified, stop_reason.
    У всех попыток кроме последней stop_reason == 'continue'.

    s = (("", "- a\\n- b\\n- c"),)
    len(refine_loop("t", s, self_feedback))  ->  1

    Порядок шагов: собрать промпт по истории, сгенерировать, проверить,
    записать попытку, решить про останов. Черновик, который уже был
    отвергнут раньше, обязан завершить петлю как 'stalled'.
    """
    raise NotImplementedError


def loop_report(history):
    """Сводка по завершённой петле — то, что уходит в лог и в дашборд.

    Вернуть словарь с ключами iterations, converged, reason, final_output,
    critiques.

    Пустая история — ValueError: отчитываться не о чем.

    h = refine_loop("t", (("", "- a\\n- b\\n- c"),), self_feedback)
    loop_report(h)["converged"]  ->  True
    loop_report(h)["iterations"] ->  1

    converged True значит, что верификатор принял последний черновик, а не
    что петля просто закончилась. Различать эти два исхода обязательно:
    'budget' и 'stalled' — это провалы, которые нужно эскалировать.
    """
    raise NotImplementedError
