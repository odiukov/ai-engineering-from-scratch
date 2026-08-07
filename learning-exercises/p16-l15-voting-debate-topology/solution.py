"""
Голосование и топология дебатов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

from itertools import permutations


def candidates(ballots):
    """Общий список кандидатов выборов, отсортированный. Заодно проверка бюллетеней.

    candidates([["A", "B"], ["B", "A"]])  ->  ("A", "B")

    Три условия, каждое ловит реальную поломку данных:
      * выборы без бюллетеней бессмысленны              -> ValueError;
      * один кандидат дважды в одном бюллетене           -> ValueError;
      * бюллетени ранжируют РАЗНЫЕ наборы кандидатов     -> ValueError.

    Последнее особенно коварно: если один агент забыл упомянуть кандидата,
    Борда молча начислит ему ноль, и результат выборов будет зависеть не от
    предпочтений, а от того, кто поленился дописать строку.
    """
    if not ballots:
        raise ValueError("нет ни одного бюллетеня")
    first = None
    for ballot in ballots:
        names = set(ballot)
        if len(names) != len(ballot):
            raise ValueError(f"кандидат повторяется в бюллетене: {ballot}")
        if first is None:
            first = names
        elif names != first:
            raise ValueError("бюллетени ранжируют разные наборы кандидатов")
    return tuple(sorted(first))


def plurality_winner(ballots):
    """Простое большинство: считаем только первые строчки бюллетеней.

    plurality_winner([["A", "B"], ["A", "B"], ["B", "A"]])  ->  "A"

    Это и есть self-consistency из урока: N агентов, у каждого один голос,
    берём самый частый ответ. Вся остальная часть бюллетеня отбрасывается —
    отсюда и главная слабость метода.

    Ничья разрешается лексикографически меньшим именем: агрегатор обязан быть
    детерминированным, иначе один и тот же прогон даст разные протоколы.
    """
    names = candidates(ballots)
    counts = dict.fromkeys(names, 0)
    for ballot in ballots:
        counts[ballot[0]] += 1
    return min(names, key=lambda c: (-counts[c], c))


def borda_scores(ballots):
    """Подсчёт по Борда: первое место даёт m - 1 очков, последнее — 0.

    borda_scores([["A", "B", "C"], ["B", "C", "A"]])
      ->  {"A": 2, "B": 3, "C": 1}

    Разбор: A получает 2 + 0, B получает 1 + 2, C получает 0 + 1.

    В отличие от большинства, Борда читает бюллетень целиком, поэтому
    кандидат, которого все ставят вторым, обгоняет того, кого половина ставит
    первым, а половина — последним. Для ансамбля агентов это «широко
    приемлемый ответ» против «ответа, который поляризует».
    """
    names = candidates(ballots)
    m = len(names)
    scores = dict.fromkeys(names, 0)
    for ballot in ballots:
        for position, name in enumerate(ballot):
            scores[name] += m - 1 - position
    return scores


def borda_winner(ballots):
    """Победитель по Борда: наибольшая сумма очков.

    borda_winner([["A", "B", "C"], ["B", "C", "A"]])  ->  "B"

    Ничья — снова лексикографически меньшее имя, тот же контракт, что у
    plurality_winner.
    """
    scores = borda_scores(ballots)
    return min(scores, key=lambda c: (-scores[c], c))


def approval_winner(approvals):
    """Одобрительное голосование: каждый отмечает любое число приемлемых.

    approval_winner([["A", "B"], ["B", "C"], ["B"]])  ->  "B"
    approval_winner([[], []])                         ->  None

    approvals — список наборов одобренных имён, по одному набору на голосующего.
    Ранжирования здесь нет вообще: голосующий не обязан упорядочивать, он
    только отделяет приемлемое от неприемлемого.

    Повтор внутри одного набора не даёт второго голоса — сначала приведи набор
    к множеству, иначе один агент сможет накрутить счёт.
    """
    counts = {}
    for approved in approvals:
        for name in set(approved):
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda c: (-counts[c], c))


def pairwise_margin(ballots, a, b):
    """Перевес a над b в парном сравнении: (за a) - (за b).

    pairwise_margin([["A", "B"], ["A", "B"], ["B", "A"]], "A", "B")  ->  1
    pairwise_margin(те же, "B", "A")                                 ->  -1
    pairwise_margin(те же, "A", "A")                                 ->  0

    Голосующий «за a», если поставил a выше b в своём бюллетене. Величина
    антисимметрична: margin(a, b) = -margin(b, a). Из этих перевесов и
    складывается общественное предпочтение — а оно, в отличие от каждого
    отдельного бюллетеня, вовсе не обязано быть транзитивным.

    Неизвестное имя — ValueError: тихий ноль тут хуже падения.
    """
    names = candidates(ballots)
    for name in (a, b):
        if name not in names:
            raise ValueError(f"нет такого кандидата: {name}")
    if a == b:
        return 0
    margin = 0
    for ballot in ballots:
        margin += 1 if ballot.index(a) < ballot.index(b) else -1
    return margin


def condorcet_winner(ballots):
    """Кандидат, обыгрывающий каждого остальных один на один. Или None.

    condorcet_winner([["A", "B"], ["A", "B"], ["B", "A"]])  ->  "A"
    condorcet_winner(профиль парадокса)                     ->  None

    Победителя Кондорсе многие считают «правильным» ответом выборов: если он
    есть, ему проигрывает каждый по отдельности. Беда в том, что его может не
    быть вовсе — и это не редкий вырожденный случай, а обычная ситуация.
    """
    names = candidates(ballots)
    for c in names:
        if all(pairwise_margin(ballots, c, other) > 0 for other in names if other != c):
            return c
    return None


def condorcet_cycle(ballots):
    """Тройка (x, y, z), где общество предпочитает x > y > z > x. Или None.

    condorcet_cycle([["A","B","C"], ["B","C","A"], ["C","A","B"]])
      ->  ("A", "B", "C")

    Это парадокс Кондорсе: каждый отдельный бюллетень — строгий транзитивный
    порядок, а общественное предпочтение зацикливается. Никакой ошибки в
    подсчёте нет, так устроено само агрегирование (теорема Эрроу — про то же
    самое). Для дебатов агентов вывод практический: «продолжаем спорить, пока
    не сойдёмся» может не сойтись никогда, поэтому раунды ограничивают.

    Перебираем упорядоченные тройки в лексикографическом порядке и возвращаем
    первую циклическую — чтобы ответ не зависел от порядка перебора. Меньше
    трёх кандидатов — цикла быть не может.
    """
    names = candidates(ballots)
    for x, y, z in permutations(names, 3):
        if (
            pairwise_margin(ballots, x, y) > 0
            and pairwise_margin(ballots, y, z) > 0
            and pairwise_margin(ballots, z, x) > 0
        ):
            return (x, y, z)
    return None
