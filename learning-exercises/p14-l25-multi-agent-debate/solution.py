"""
Multi-agent debate: дебаты и коллаборация агентов — эталон.

Открывай ПОСЛЕ своих зелёных тестов.
"""

# Топологии из урока. Full mesh — все читают всех, star — хаб и спицы,
# ring — кольцо из двух соседей.
TOPOLOGIES = ("full_mesh", "star", "ring")

# Хаб звезды. Урок отдельно предупреждает: плохой хаб портит всех, поэтому
# его индекс вынесен в константу — так его хотя бы видно.
HUB = 0


def majority_answer(proposals):
    """Итог голосования: самый частый ответ, ничья — по алфавиту.

    majority_answer(["A", "B", "A"])  ->  "A"
    majority_answer(["B", "A"])       ->  "A"   (по голосу у каждого, берём меньший)
    majority_answer([])               ->  ValueError

    Ловушка: `Counter.most_common()` при равных счётчиках возвращает порядок
    вставки. Тогда итог дебатов начнёт зависеть от того, в каком порядке
    агенты попали в список, — а в проде этот порядок случайный. Сортируй по
    паре (-голоса, ответ), тогда перестановка агентов ничего не меняет.
    """
    if not proposals:
        raise ValueError("голосовать не за что: пустой список предложений")
    votes = {}
    for answer in proposals:
        votes[answer] = votes.get(answer, 0) + 1
    # ключ сортировки, а не max(): нужен детерминированный разрыв ничьей
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def weighted_answer(proposals):
    """Голосование с весами: список пар (ответ, уверенность). Побеждает сумма.

    weighted_answer([("A", 0.25), ("A", 0.25), ("B", 0.75)])  ->  "B"
    weighted_answer([("A", 1.0), ("B", 1.0)])                 ->  "A"   (ничья -> алфавит)
    weighted_answer([("A", -1.0)])                            ->  ValueError

    Уверенность отрицательной быть не может: «минус уверен» — это голос
    против, а протокол дебатов такого хода не предусматривает.

    Из упражнения 2 урока: помогает, когда один агент действительно знает
    ответ, а остальные угадывают. Вредит, когда модель откалибрована плохо
    и уверенность не связана с правотой.
    """
    if not proposals:
        raise ValueError("голосовать не за что: пустой список предложений")
    weights = {}
    for answer, confidence in proposals:
        if confidence < 0:
            raise ValueError(f"отрицательная уверенность у {answer!r}: {confidence}")
        weights[answer] = weights.get(answer, 0.0) + confidence
    return sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def topology_peers(kind, n):
    """Кто чьи предложения читает: {индекс агента: отсортированные соседи}.

    topology_peers("full_mesh", 3)  ->  {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    topology_peers("star", 3)       ->  {0: [1, 2], 1: [0], 2: [0]}
    topology_peers("ring", 4)       ->  {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [0, 2]}

    Звезда: хаб (агент HUB) читает всех, спицы читают только хаб. Кольцо:
    каждый читает двух соседей по кругу. На n = 3 кольцо совпадает с полной
    сеткой — экономия начинается с четырёх агентов.

    n < 2 -> ValueError (дебаты в одиночку — это Self-Refine из урока 05).
    Неизвестная топология -> ValueError.
    """
    if kind not in TOPOLOGIES:
        raise ValueError(f"неизвестная топология: {kind!r}, доступны {TOPOLOGIES}")
    if n < 2:
        raise ValueError(f"дебаты требуют минимум двух агентов, дано {n}")

    if kind == "full_mesh":
        return {i: [j for j in range(n) if j != i] for i in range(n)}
    if kind == "star":
        peers = {HUB: [j for j in range(n) if j != HUB]}
        peers.update({i: [HUB] for i in range(n) if i != HUB})
        return peers
    # ring: set() убирает дубль на n = 2, где оба соседа — один и тот же агент
    return {i: sorted({(i - 1) % n, (i + 1) % n} - {i}) for i in range(n)}


def critique_ops(peers_map, rounds):
    """Стоимость прогона в операциях критики: rounds * число рёбер.

    critique_ops(topology_peers("full_mesh", 5), 3)  ->  60
    critique_ops(topology_peers("star", 5), 3)       ->  24
    critique_ops(topology_peers("full_mesh", 5), 0)  ->  0

    Одна операция — «агент i прочитал предложение агента j». Отрицательное
    число раундов -> ValueError.

    Оговорка к уроку: там для звезды N=5, R=3 названо 12 операций — это счёт
    только в одну сторону, спицы читают хаб. Хаб, который сам никого не
    читает, ничему не научится и останется при своём первом ответе, поэтому
    здесь рёбра считаются в обе стороны: 24. Всё равно втрое дешевле полной
    сетки, вывод урока не меняется.
    """
    if rounds < 0:
        raise ValueError(f"раундов не может быть меньше нуля: {rounds}")
    return rounds * sum(len(peers) for peers in peers_map.values())


def debate_round(opinions, peers_map, stubborn=()):
    """Один раунд дебатов: каждый агент пересматривает ответ по себе и соседям.

    debate_round(["A", "B", "B"], topology_peers("full_mesh", 3))
      ->  ["B", "B", "B"]
    debate_round(["A", "B", "B"], topology_peers("full_mesh", 3), stubborn=(0,))
      ->  ["A", "B", "B"]

    Агент считает majority_answer по списку [своё мнение] + мнения соседей.
    Индексы из stubborn не меняют мнение вообще — это «forced disagreement»
    из упражнения 1 урока, страховка от convergence collapse.

    ГЛАВНАЯ ЛОВУШКА: обновление ОДНОВРЕМЕННОЕ. Все читают СТАРЫЕ мнения,
    новый список собирается отдельно. Если править список на месте, агент 1
    увидит уже обновлённого агента 0, и результат раунда станет зависеть от
    порядка обхода — на звезде это видно сразу.
    """
    updated = []
    for i, own in enumerate(opinions):
        if i in stubborn:
            updated.append(own)
            continue
        # [own] + соседи: агент участвует в собственном голосовании,
        # иначе одинокий носитель правды мгновенно потеряет свой ответ
        updated.append(majority_answer([own] + [opinions[j] for j in peers_map[i]]))
    return updated


def run_debate(opinions, peers_map, rounds, stubborn=()):
    """Прогон дебатов до единогласия или до исчерпания раундов.

    Вернуть dict с ключами:
      "answer"      — итоговый ответ (majority_answer по последнему раунду);
      "converged"   — все ли согласились;
      "rounds_used" — сколько раундов реально отработали;
      "ops"         — стоимость по critique_ops для отработанных раундов;
      "history"     — список снимков мнений, history[0] — стартовые.

    run_debate(["A", "A", "B"], topology_peers("full_mesh", 3), 3)["answer"]
      ->  "A"
    run_debate(["A", "A"], topology_peers("full_mesh", 2), 5)["rounds_used"]
      ->  0    (стартовое единогласие — спорить не о чем, платить не за что)

    Ранний выход — не мелочь: именно на нём держится вся экономия из урока.
    """
    current = list(opinions)
    history = [list(current)]
    used = 0
    for _ in range(rounds):
        if len(set(current)) == 1:
            break
        current = debate_round(current, peers_map, stubborn)
        history.append(list(current))
        used += 1
    return {
        "answer": majority_answer(current),
        "converged": len(set(current)) == 1,
        "rounds_used": used,
        "ops": critique_ops(peers_map, used),
        "history": history,
    }


def collapsed_early(history):
    """Признак convergence collapse: разногласие исчезло за один раунд.

    collapsed_early([["A", "B", "B"], ["B", "B", "B"]])  ->  True
    collapsed_early([["A", "A", "B"], ["A", "A", "B"]])  ->  False
    collapsed_early([["A", "A"], ["A", "A"]])            ->  False

    Стартовое единогласие коллапсом не считается: спорить было не о чем.

    Это сигнал, а не приговор. Быстрое схождение бывает и у правильного
    ответа. Но урок предупреждает ровно об этом сценарии: все повторили
    первое озвученное мнение, критики не случилось, ошибка не вскрылась.
    """
    if len(history) < 2:
        return False
    return len(set(history[0])) > 1 and len(set(history[1])) == 1


def compare_topologies(opinions, rounds, truth, kinds=TOPOLOGIES):
    """Стоимость против точности: прогнать одни и те же мнения по топологиям.

    Вернуть {топология: {"answer", "correct", "ops", "converged", "rounds_used"}}.

    compare_topologies(["A", "A", "A", "B", "B"], 3, "A")["star"]["ops"]  ->  8
    compare_topologies(["A", "A", "A", "B", "B"], 3, "A")["full_mesh"]["ops"]  ->  20

    Это упражнение 4 урока в одну функцию: если разреженная топология даёт
    тот же ответ дешевле, полная сетка не нужна.
    """
    report = {}
    for kind in kinds:
        peers = topology_peers(kind, len(opinions))
        result = run_debate(opinions, peers, rounds)
        report[kind] = {
            "answer": result["answer"],
            "correct": result["answer"] == truth,
            "ops": result["ops"],
            "converged": result["converged"],
            "rounds_used": result["rounds_used"],
        }
    return report
