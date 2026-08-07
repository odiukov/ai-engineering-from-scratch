"""Входные данные для замера скорости."""

import json
import random

random.seed(0)  # обязательно: замер должен быть воспроизводим

_ONTOLOGY = {
    "cuisine": {
        "italian": ["italian", "pasta", "pizza"],
        "chinese": ["chinese", "dim sum"],
        "indian": ["indian", "curry", "tandoori"],
        "thai": ["thai", "pad thai"],
        "any": ["any food", "any cuisine"],
    },
    "area": {
        "north": ["north"], "south": ["south"], "east": ["east"],
        "west": ["west"], "center": ["center", "centre"],
    },
    "price": {
        "cheap": ["cheap", "budget", "inexpensive"],
        "moderate": ["moderate", "mid-range", "medium"],
        "expensive": ["expensive", "fancy", "upscale"],
    },
}
_CUES = ["never mind", "forget about", "don't worry about"]
_SCHEMA = {
    "cuisine": ["italian", "chinese", "indian", "thai", "any"],
    "area": ["north", "south", "east", "west", "center"],
    "price": ["cheap", "moderate", "expensive"],
    "name": None,
}

_TEMPLATES = [
    "I want a {price} {cuisine} restaurant in the {area}",
    "actually make it {price}",
    "never mind the cuisine, any food is fine",
    "somewhere in the {area} please",
    "{cuisine} food would be nice",
]
_FILL = {"price": ["cheap", "moderate", "expensive"],
         "cuisine": ["italian", "chinese", "indian", "thai"],
         "area": ["north", "south", "east", "west", "center"]}


def _utterance():
    tpl = random.choice(_TEMPLATES)
    return tpl.format(**{k: random.choice(v) for k, v in _FILL.items()})


_UTTERANCE = _utterance()
_TURNS = [_utterance() for _ in range(2000)]

_STATES = [
    {"cuisine": random.choice(_FILL["cuisine"]),
     "area": random.choice(_FILL["area"]),
     "price": random.choice(_FILL["price"])}
    for _ in range(5000)
]
_GOLD = [
    {"cuisine": random.choice(_FILL["cuisine"]),
     "area": random.choice(_FILL["area"]),
     "price": random.choice(_FILL["price"])}
    for _ in range(5000)
]

_RAW_STATE = {"cuisine": "ITALIAN", "area": " north ", "price": "cheap",
              "name": "The Copper Kettle", "vibe": "cosy"}


def _llm(prompt):
    return json.dumps({"cuisine": "italian", "area": "north", "price": "cheap"})


BENCH = {
    "extract_slots": (_UTTERANCE, _ONTOLOGY),
    "is_negated": (_UTTERANCE, "cuisine", _CUES),
    "update_state": (_STATES[0], _UTTERANCE, _ONTOLOGY, _CUES),
    "track_dialogue": (_TURNS, _ONTOLOGY, _CUES),
    "validate_state": (_RAW_STATE, _SCHEMA),
    "llm_dst": (_TURNS[:200], _llm, _SCHEMA),
    "joint_goal_accuracy": (_STATES, _GOLD),
    "slot_accuracy": (_STATES, _GOLD),
}
