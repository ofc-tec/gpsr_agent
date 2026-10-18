import json
from pathlib import Path

DEFAULT_KNOWLEDGE = Path(__file__).resolve().parent.parent / "knowledge"

def load_json(name: str, folder: Path = DEFAULT_KNOWLEDGE) -> dict:
    return json.loads((folder / name).read_text(encoding="utf-8"))

def load_facts(folder: Path = DEFAULT_KNOWLEDGE) -> list[dict]:
    """Merge both sources, keeping the first copy of each semantic fact.

    Source metadata does not make a new fact. Kind remains part of identity
    so defaults and observations are not collapsed into each other.
    """
    facts = []
    seen = set()
    for name in ("facts.json", "competition_facts.json"):
        for fact in load_json(name, folder)["facts"]:
            key = json.dumps(
                [fact["subject"], fact["relation"], fact["object"], fact.get("kind")],
                sort_keys=True,
            )
            if key not in seen:
                seen.add(key)
                facts.append(fact)
    return facts

def query_facts(subject: str | None = None, relation: str | None = None) -> list[dict]:
    facts = load_facts()
    return [f for f in facts if (subject is None or f["subject"] == subject)
            and (relation is None or f["relation"] == relation)]

def query_precepts() -> list[dict]:
    return load_json("precepts.json")["precepts"]
