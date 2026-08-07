"""Тесты к уроку «Команды агентов по ролям — роли, задачи, процессы». Правь exercise.py."""

import random

import pytest

from exercise import (
    MAX_BACKSTORY_WORDS,
    crew_prompt,
    make_agent,
    make_task,
    recall_context,
    remember,
    run_flow,
    run_hierarchical,
    run_sequential,
)

SPECIALIST_OUTPUTS = {
    "researcher": "3 sources found",
    "writer": "draft in 3 paragraphs",
    "editor": "final brief, 800 words",
}

ROLE_ORDER = ("researcher", "writer", "editor")


def search(query):
    """Return top results for the query."""
    return f"src1, src2, src3 for {query}"


def _role_of(prompt):
    for line in prompt.splitlines():
        if line.startswith("ROLE: "):
            return line[len("ROLE: ") :]
    return ""


def _done_line(prompt):
    for line in prompt.splitlines():
        if "DONE:" in line:
            return line.split("DONE:", 1)[1]
    return ""


def scripted_agent(prompt):
    """Детерминированная заглушка модели: ответ зависит только от промпта.

    Менеджер называет первую роль, которой ещё нет в строке DONE, и
    отвечает "done", когда сделаны все три.
    """
    role = _role_of(prompt)
    if role == "manager":
        done = _done_line(prompt)
        for candidate in ROLE_ORDER:
            if candidate not in done:
                return candidate
        return "done"
    return SPECIALIST_OUTPUTS.get(role, f"output from {role}")


def stubborn_agent(prompt):
    """Менеджер, залипший на одной роли: проверка защиты от бесконечной петли."""
    if _role_of(prompt) == "manager":
        return "researcher"
    return scripted_agent(prompt)


def hallucinating_agent(prompt):
    """Менеджер, который называет несуществующую роль."""
    if _role_of(prompt) == "manager":
        return "designer"
    return scripted_agent(prompt)


def noisy_agent(seed):
    """Заглушка модели с разбросом: случайность приходит через seed, не глобально."""
    rng = random.Random(seed)

    def run(prompt):
        return f"{_role_of(prompt)} draft #{rng.randrange(10 ** 6)}"

    return run


def brief_crew():
    """Три агента и три задачи: исследователь -> писатель -> редактор."""
    researcher = make_agent(
        "researcher", "find 3 credible sources", "former librarian, terse", tools=(search,)
    )
    writer = make_agent("writer", "turn sources into a draft", "editorial voice")
    editor = make_agent("editor", "tighten the draft", "cuts adjectives")
    return [
        make_task("research the topic", "3 sources", researcher),
        make_task("write a draft", "3 paragraphs", writer),
        make_task("edit to final brief", "800 words", editor),
    ]


def manager_task():
    manager = make_agent("manager", "pick the next specialist", "PM background, routes by gap")
    return make_task("route the crew", "a role name or done", manager)


# ---------------------------------------------------------------- make_agent
def test_make_agent_keeps_all_four_fields():
    agent = make_agent("writer", "turn sources into a draft", "editorial voice", tools=(search,))
    assert agent["role"] == "writer"
    assert agent["goal"] == "turn sources into a draft"
    assert agent["backstory"] == "editorial voice"
    assert agent["tools"] == (search,)


def test_make_agent_requires_a_backstory():
    """Backstory задаёт тон и момент останова, поэтому пустым он быть не может."""
    with pytest.raises(ValueError):
        make_agent("writer", "draft it", "   ")


def test_make_agent_rejects_a_backstory_that_bloats_the_prompt():
    """Раздутая backstory уезжает в модель на каждом шаге каждого агента."""
    fat = " ".join(["word"] * (MAX_BACKSTORY_WORDS + 1))
    with pytest.raises(ValueError):
        make_agent("writer", "draft it", fat)
    ok = " ".join(["word"] * MAX_BACKSTORY_WORDS)
    assert make_agent("writer", "draft it", ok)["role"] == "writer"


def test_make_agent_rejects_a_tool_without_a_docstring():
    """Docstring — это описание, по которому модель выбирает инструмент."""

    def mystery(query):
        return query

    with pytest.raises(ValueError):
        make_agent("researcher", "find sources", "terse", tools=(mystery,))


# ----------------------------------------------------------------- make_task
def test_make_task_keeps_the_contract_and_the_assignee():
    agent = make_agent("writer", "draft it", "terse voice")
    task = make_task("write a draft", "3 paragraphs", agent)
    assert task["expected_output"] == "3 paragraphs"
    assert task["agent"] is agent
    assert task["context"] == ()


def test_make_task_refuses_a_task_without_expected_output():
    """Без контракта crew отработает, а аудит провалится."""
    agent = make_agent("writer", "draft it", "terse voice")
    with pytest.raises(ValueError):
        make_task("write a draft", "", agent)


def test_make_task_refuses_an_unassigned_task():
    with pytest.raises(ValueError):
        make_task("write a draft", "3 paragraphs", "writer")



# --------------------------------------------------------------- crew_prompt
def test_crew_prompt_carries_role_goal_and_backstory():
    agent = make_agent("writer", "turn sources into a draft", "editorial voice")
    task = make_task("write a draft", "3 paragraphs", agent)
    prompt = crew_prompt(agent, task)
    assert "ROLE: writer" in prompt
    assert "GOAL: turn sources into a draft" in prompt
    assert "BACKSTORY: editorial voice" in prompt


def test_crew_prompt_states_the_expected_output_contract():
    agent = make_agent("writer", "draft it", "terse voice")
    task = make_task("write a draft", "3 paragraphs", agent)
    assert "EXPECTED OUTPUT: 3 paragraphs" in crew_prompt(agent, task)


def test_crew_prompt_numbers_context_in_the_order_given():
    agent = make_agent("editor", "tighten it", "cuts adjectives")
    task = make_task("edit", "800 words", agent)
    prompt = crew_prompt(agent, task, ("3 sources", "draft in 3 paragraphs"))
    assert "CONTEXT 1: 3 sources" in prompt
    assert "CONTEXT 2: draft in 3 paragraphs" in prompt


def test_crew_prompt_shows_tools_with_their_descriptions():
    agent = make_agent("researcher", "find sources", "terse", tools=(search,))
    task = make_task("research", "3 sources", agent)
    prompt = crew_prompt(agent, task)
    assert "TOOLS: search — Return top results for the query." in prompt



# ------------------------------------------------------------ run_sequential
def test_run_sequential_runs_tasks_in_declaration_order():
    trace = run_sequential(brief_crew(), "agent engineering 2026", scripted_agent)
    assert [step["role"] for step in trace] == list(ROLE_ORDER)


def test_run_sequential_threads_each_output_into_the_next_prompt():
    """Хендофф обязан донести результат предыдущего шага до следующего."""
    trace = run_sequential(brief_crew(), "agent engineering 2026", scripted_agent)
    assert "CONTEXT 1: 3 sources found" in trace[1]["prompt"]
    assert "CONTEXT 1: draft in 3 paragraphs" in trace[2]["prompt"]


def test_run_sequential_does_not_lose_the_original_topic():
    trace = run_sequential(brief_crew(), "agent engineering 2026", scripted_agent)
    assert "CONTEXT 1: agent engineering 2026" in trace[0]["prompt"]


def test_run_sequential_prefers_declared_context_over_the_previous_task():
    """Объявленный context важнее соседства в списке задач."""
    tasks = brief_crew()
    tasks[2] = make_task(
        "edit to final brief", "800 words", tasks[2]["agent"], context=(tasks[0],)
    )
    trace = run_sequential(tasks, "agent engineering 2026", scripted_agent)
    assert "CONTEXT 1: 3 sources found" in trace[2]["prompt"]
    assert "draft in 3 paragraphs" not in trace[2]["prompt"]


def test_run_sequential_rejects_context_pointing_forward():
    tasks = brief_crew()
    tasks[0] = make_task("research", "3 sources", tasks[0]["agent"], context=(tasks[2],))
    with pytest.raises(ValueError):
        run_sequential(tasks, "agent engineering 2026", scripted_agent)



# ---------------------------------------------------------- run_hierarchical
def test_run_hierarchical_routes_through_every_specialist():
    result = run_hierarchical(manager_task(), brief_crew(), "agents", scripted_agent)
    assert result["done"] == list(ROLE_ORDER)
    assert result["stop_reason"] == "done"
    assert result["final"] == SPECIALIST_OUTPUTS["editor"]


def test_run_hierarchical_pays_the_manager_llm_tax():
    """Один лишний вызов модели на каждый раунд плюс закрывающий 'done'."""
    result = run_hierarchical(manager_task(), brief_crew(), "agents", scripted_agent)
    assert result["llm_calls"] == 2 * len(result["done"]) + 1
    assert result["llm_calls"] > len(run_sequential(brief_crew(), "agents", scripted_agent))


def test_run_hierarchical_hands_the_latest_output_to_the_next_specialist():
    """Контекст передаётся вперёд, а исходный запрос не теряется на первом шаге."""
    result = run_hierarchical(manager_task(), brief_crew(), "agents", scripted_agent)
    specialists = [step for step in result["trace"] if step["pick"] is None]
    assert "CONTEXT 1: agents" in specialists[0]["prompt"]
    assert f"CONTEXT 1: {SPECIALIST_OUTPUTS['researcher']}" in specialists[1]["prompt"]
    assert f"CONTEXT 1: {SPECIALIST_OUTPUTS['writer']}" in specialists[2]["prompt"]



def test_run_hierarchical_stops_on_a_role_that_does_not_exist():
    result = run_hierarchical(manager_task(), brief_crew(), "agents", hallucinating_agent)
    assert result["stop_reason"] == "unknown pick 'designer'"
    assert result["done"] == []
    assert result["llm_calls"] == 1


def test_run_hierarchical_stops_when_the_manager_repeats_a_finished_role():
    """Залипший менеджер иначе платит за каждый раунд до конца бюджета."""
    result = run_hierarchical(
        manager_task(), brief_crew(), "agents", stubborn_agent, max_rounds=50
    )
    assert result["stop_reason"] == "repeated pick 'researcher'"
    assert result["llm_calls"] == 3


def test_run_hierarchical_respects_the_round_budget():
    result = run_hierarchical(manager_task(), brief_crew(), "agents", scripted_agent, max_rounds=2)
    assert result["stop_reason"] == "budget"
    assert result["done"] == ["researcher", "writer"]



# ------------------------------------------------------------------ run_flow
def _brief_flow():
    def start(topic):
        return ("researched", f"3 sources on {topic}")

    def on_researched(prior):
        return ("drafted", f"draft from {prior}")

    def on_drafted(prior):
        return ("edited", f"final {prior}")

    return start, {"researched": on_researched, "drafted": on_drafted}


def test_run_flow_records_explicit_topics_in_order():
    start, listeners = _brief_flow()
    trace = run_flow(start, listeners, "agents")
    assert [topic for _, topic, _ in trace] == ["researched", "drafted", "edited"]
    assert trace[0][0] == "start"


def test_run_flow_is_replayable():
    """Одинаковый вход — побайтово одинаковая трасса: это и есть аудит."""
    start, listeners = _brief_flow()
    assert run_flow(start, listeners, "agents") == run_flow(start, listeners, "agents")


def test_run_flow_ends_when_a_listener_returns_none():
    def start(topic):
        return ("researched", topic)

    def on_researched(prior):
        return None

    trace = run_flow(start, {"researched": on_researched}, "agents")
    assert len(trace) == 1


def test_run_flow_refuses_to_spin_forever_on_a_cycle():
    def start(payload):
        return ("a", payload)

    def on_a(value):
        return ("b", value)

    def on_b(value):
        return ("a", value)

    with pytest.raises(ValueError):
        run_flow(start, {"a": on_a, "b": on_b}, "x", max_steps=8)


def test_flow_shape_is_owned_by_code_while_crew_output_is_owned_by_the_model():
    """Разброс модели меняет тексты в обоих, но темы Flow остаются те же."""
    crew_a = [step["output"] for step in run_sequential(brief_crew(), "t", noisy_agent(1))]
    crew_b = [step["output"] for step in run_sequential(brief_crew(), "t", noisy_agent(2))]
    assert crew_a != crew_b

    def flow_for(agent):
        def start(topic):
            return ("researched", agent("ROLE: researcher"))

        def on_researched(prior):
            return ("drafted", agent("ROLE: writer"))

        return start, {"researched": on_researched}

    trace_a = run_flow(*flow_for(noisy_agent(1)), "t")
    trace_b = run_flow(*flow_for(noisy_agent(2)), "t")
    assert trace_a != trace_b
    assert [topic for _, topic, _ in trace_a] == [topic for _, topic, _ in trace_b]


# ------------------------------------------------------------------ remember
def test_remember_appends_to_the_named_store():
    memory = remember({}, "long_term", "crew shipped the brief")
    assert memory["long_term"] == ["crew shipped the brief"]


def test_remember_keys_entity_facts_by_entity():
    memory = {}
    remember(memory, "entity", "on the enterprise plan", key="customer-7")
    remember(memory, "entity", "prefers email", key="customer-7")
    remember(memory, "entity", "on the free plan", key="customer-9")
    assert memory["entity"]["customer-7"] == ["on the enterprise plan", "prefers email"]
    assert memory["entity"]["customer-9"] == ["on the free plan"]


def test_remember_rejects_an_unknown_store_and_a_keyless_entity():
    with pytest.raises(ValueError):
        remember({}, "medium_term", "oops")
    with pytest.raises(ValueError):
        remember({}, "entity", "on the enterprise plan")


# ------------------------------------------------------------ recall_context
def test_recall_context_returns_long_term_hits_by_word_overlap():
    memory = {}
    remember(memory, "long_term", "brief on agent engineering")
    remember(memory, "long_term", "notes about audio codecs")
    assert recall_context(memory, "agent brief")["long_term"] == ["brief on agent engineering"]


def test_recall_context_drops_irrelevant_long_term_entries():
    """Always-on память без фильтра превращает выдачу в шум."""
    memory = {}
    for fact in ("notes about audio codecs", "gpu autoscaling checklist"):
        remember(memory, "long_term", fact)
    assert recall_context(memory, "agent brief")["long_term"] == []


def test_recall_context_caps_long_term_at_k():
    memory = {}
    for i in range(5):
        remember(memory, "long_term", f"brief number {i}")
    assert len(recall_context(memory, "brief", k=2)["long_term"]) == 2


def test_recall_context_finds_entity_facts_that_share_no_words_with_the_query():
    """Entity-память достаётся по ключу, а не по похожести."""
    memory = {}
    remember(memory, "entity", "on the enterprise plan", key="customer-7")
    recalled = recall_context(memory, "agent brief", entity_id="customer-7")
    assert recalled["entity"] == ["on the enterprise plan"]
    assert recall_context(memory, "agent brief")["entity"] == []


def test_recall_context_survives_a_wiped_short_term_buffer():
    """Long-term живёт между kickoff-ами, short-term стирается в конце прогона."""
    memory = {}
    remember(memory, "short_term", "this run: 3 sources found")
    remember(memory, "long_term", "brief on agent engineering")
    memory["short_term"] = []
    recalled = recall_context(memory, "agent brief")
    assert recalled["short_term"] == []
    assert recalled["long_term"] == ["brief on agent engineering"]


