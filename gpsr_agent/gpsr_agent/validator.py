from .schemas import Plan
from .context import load_facts

ARG_KEYS = {
    "navigate_to": {"location"}, "detect_object": {"object", "location", "color", "attributes"},
    "detect_person": {"person", "location", "gesture", "clothing_color", "attributes"},
    "grasp": {"object"}, "release": {"object", "target", "location"},
    "follow_person": {"person", "destination"}, "say": {"text"}, "listen": {"prompt"},
    "count_objects": {"category", "location"},
}
REQUIRED = {
    "navigate_to": {"location"}, "detect_object": {"object"}, "detect_person": set(),
    "grasp": {"object"}, "release": {"object"}, "follow_person": {"person"},
    "say": {"text"}, "listen": set(),"count_objects": {"category", "location"},
}

def entity_key(value: str) -> str:
    """Match existing underscore IDs against competition display names."""
    return " ".join(value.replace("_", " ").casefold().split())

def known_entities() -> dict[str, set[str]]:
    # The operator is the established command-giver reference, not a named person.
    entities = {"object": set(), "person": {"operator"}, "location": {"operator"}}
    types = {"is_object": "object", "is_person": "person",
             "is_room": "location", "is_location": "location"}
    for fact in load_facts():
        subject, relation, value = fact["subject"], fact["relation"], fact["object"]
        if relation in types and value is True:
            entities[types[relation]].add(entity_key(subject))
        elif relation == "inside" and isinstance(value, str):
            entities["location"].update((entity_key(subject), entity_key(value)))
        elif relation == "usually_found_at" and isinstance(value, str):
            entities["object"].add(entity_key(subject))
            entities["location"].add(entity_key(value))
        elif relation == "instruction_point" and isinstance(value, str):
            entities["location"].add(entity_key(value))
    return entities

def validate_plan(data: dict) -> dict:
    try:
        plan = Plan.model_validate(data)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}
    errors = []
    entities = known_entities()
    for i, step in enumerate(plan.steps):
        missing = REQUIRED[step.action] - step.args.keys()
        extra = step.args.keys() - ARG_KEYS[step.action]
        if missing:
            errors.append(f"step {i}: missing arguments {sorted(missing)}")
        if extra:
            errors.append(f"step {i}: unsupported arguments {sorted(extra)}")
        for key, value in step.args.items():
            if key in {"object", "person"}:
                allowed = entities[key]
            elif key in {"location", "destination", "target"}:
                allowed = entities["location"] | entities["person"]
            else:
                continue
            if not isinstance(value, str) or entity_key(value) not in allowed:
                errors.append(f"step {i}: unknown {key} {value!r}")
    return {"valid": not errors, "errors": errors, "plan": plan.model_dump()}
