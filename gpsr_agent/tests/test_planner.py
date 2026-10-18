from gpsr_agent.validator import validate_plan
from gpsr_agent.context import query_facts


def test_merged_facts_deduplicate_without_losing_distinct_facts(tmp_path):
    import json
    from gpsr_agent.context import load_facts

    original = {"subject": "milk", "relation": "usually_found_at",
                "object": "pantry", "kind": "default", "source": "seed"}
    duplicate = {**original, "source": "competition_template"}
    observed = {**original, "kind": "observation"}
    other_location = {**original, "object": "refrigerator"}
    for name, facts in [("facts.json", [original, original]),
                        ("competition_facts.json", [duplicate, observed, other_location])]:
        (tmp_path / name).write_text(json.dumps({"facts": facts}))
    assert load_facts(tmp_path) == [original, observed, other_location]


def test_get_facts_includes_both_sources():
    import json
    from gpsr_agent.agent import get_facts

    assert json.loads(get_facts.invoke({"subject": "robot"}))[0]["relation"] == "instruction_point"
    assert json.loads(get_facts.invoke({"subject": "milk", "relation": "is_object"}))[0]["object"] is True


def test_validator_uses_competition_and_local_entities():
    result = validate_plan({"command": "bring milk", "steps": [
        {"action": "navigate_to", "args": {"location": "pantry"}},
        {"action": "detect_object", "args": {"object": "milk"}},
        {"action": "detect_person", "args": {"person": "Adel"}},
        {"action": "navigate_to", "args": {"location": "gpsr_instruction_point"}},
    ]})
    assert result["valid"], result["errors"]


def test_validator_matches_space_and_underscore_names():
    for name in ("orange juice", "orange_juice"):
        assert validate_plan({"command": "find juice", "steps": [
            {"action": "detect_object", "args": {"object": name}},
        ]})["valid"]


def test_validator_rejects_unknown_entities_and_wrong_types():
    for action, args in [
        ("navigate_to", {"location": "invented_room"}),
        ("grasp", {"object": "invented_object"}),
        ("follow_person", {"person": "invented_person"}),
        ("grasp", {"object": "kitchen"}),
        ("navigate_to", {"location": ["pantry"]}),
    ]:
        assert not validate_plan({"command": "x", "steps": [
            {"action": action, "args": args},
        ]})["valid"]

def test_known_fact():
    assert query_facts("strawberry_jello", "usually_found_at")[0]["object"] == "bedside_table"

def test_valid_sequence():
    result = validate_plan({"command": "bring sponge", "steps": [
        {"action": "navigate_to", "args": {"location": "kitchen_table"}},
        {"action": "detect_object", "args": {"object": "sponge"}},
        {"action": "grasp", "args": {"object": "sponge"}},
        {"action": "navigate_to", "args": {"location": "operator"}},
        {"action": "release", "args": {"object": "sponge", "target": "operator"}},
    ]})
    assert result["valid"]

def test_unknown_action_rejected():
    result = validate_plan({"command": "x", "steps": [{"action": "teleport", "args": {}}]})
    assert not result["valid"]

def test_missing_arguments_rejected():
    result = validate_plan({"command": "x", "steps": [{"action": "navigate_to", "args": {}}]})
    assert not result["valid"]
