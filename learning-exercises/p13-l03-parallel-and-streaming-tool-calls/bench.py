"""Входные данные для замера скорости."""

import itertools
import json
import random

random.seed(0)

_LATENCIES = [random.randint(50, 900) for _ in range(200)]

# Двести параллельных вызовов. Чанки идут по одному проводу вперемешку, но
# внутри каждого id порядок сохраняется — как в настоящем стриме.
_IDS = [f"call_{i:03d}" for i in range(200)]
_PAYLOADS = {
    cid: json.dumps({"city": random.choice(["Tokyo", "Zurich", "Lagos"]), "n": i})
    for i, cid in enumerate(_IDS)
}
_PER_CALL = [
    [
        {"type": "args_delta", "id": cid, "chunk": text[i : i + 4]}
        for i in range(0, len(text), 4)
    ]
    for cid, text in _PAYLOADS.items()
]

_EVENTS = [{"type": "call_start", "id": cid, "name": "get_weather"} for cid in _IDS]
# zip_longest по вызовам = чередование чанков разных id при сохранении
# внутреннего порядка каждого
_EVENTS += [
    event
    for wave in itertools.zip_longest(*_PER_CALL)
    for event in wave
    if event is not None
]
_EVENTS += [{"type": "call_stop", "id": cid} for cid in _IDS]

_CALLS = [{"id": cid, "name": "get_weather", "arguments": {}} for cid in _IDS]
_RESULTS = [{"tool_call_id": cid, "content": "ok"} for cid in _IDS]
random.shuffle(_RESULTS)

_DEPENDS = {_IDS[i]: [_IDS[i - 1]] for i in range(1, 200, 3)}

BENCH = {
    "sequential_duration": (_LATENCIES,),
    "parallel_duration": (_LATENCIES,),
    "speedup": (_LATENCIES,),
    "try_parse_arguments": (_PAYLOADS[_IDS[0]],),
    "accumulate_stream": (_EVENTS,),
    "stream_completion_order": (_EVENTS,),
    "correlate_results": (_CALLS, _RESULTS),
    "parallel_batches": (_CALLS, _DEPENDS),
}
