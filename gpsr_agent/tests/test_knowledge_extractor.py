import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from gpsr_agent import agent
from gpsr_agent.knowledge_extractor import (
    Proposal, Proposals, read_evaluations, review_proposals,
)


@pytest.fixture
def knowledge(tmp_path):
    folder = tmp_path / "knowledge"
    folder.mkdir()
    files = {
        "facts.json": {"schema_version": 1, "note": "preserve this", "facts": [
            {"subject": "table", "relation": "inside", "object": "kitchen", "source": "seed"},
        ]},
        "competition_facts.json": {"facts": [
            {"subject": "cup", "relation": "is_object", "object": True, "source": "competition"},
        ]},
        "precepts.json": {"schema_version": 1, "precepts": [
            {"id": "existing", "instruction": "Detect before grasping."},
            {"name": "legacy", "when": "reporting", "then": "Return to the operator", "priority": "mandatory"},
        ]},
    }
    for name, data in files.items():
        (folder / name).write_text(json.dumps(data))
    return folder


def proposal(kind="precept", **kwargs):
    return Proposal(type=kind, description="Monitor the guided person's proximity and verify arrival.",
                    rationale="Guidance must verify the person reaches the destination.",
                    evidence=["evaluation_001"], **kwargs)


EVALUATIONS = [{"id": "evaluation_001", "accepted": False,
                "feedback": "You must check that the person is following."}]


@pytest.mark.parametrize("answer", ["", "n"])
def test_feedback_persistence_and_flag(knowledge, monkeypatch, answer):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", knowledge)
    plan = {"command": "say hello", "steps": [{"action": "say", "args": {"text": "hello"}}]}
    fake = SimpleNamespace(invoke=lambda request: {
        "messages": [AIMessage(content=json.dumps(plan))]
    })
    with patch("sys.argv", ["gpsr-agent", "--feedback", "say hello"]):
        with patch("builtins.input", side_effect=[answer, "Wrong greeting"]):
            agent.run_cli(fake, "test:model")
    path = knowledge.parent / "evaluations.jsonl"
    saved = json.loads(path.read_text())
    assert saved["instruction"] == "say hello"
    assert saved["plan"] == plan
    assert saved["validation"]["valid"] is True
    assert saved["accepted"] is (answer == "")
    assert saved["feedback"] == ("" if answer == "" else "Wrong greeting")
    assert saved["timestamp"] and saved["id"].startswith("evaluation_")
    assert saved["model"] == "test:model"
    assert len(read_evaluations(path)) == (0 if answer == "" else 1)


def test_normal_run_does_not_prompt_or_save(knowledge, monkeypatch):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", knowledge)
    plan = {"command": "hello", "steps": []}
    fake = SimpleNamespace(invoke=lambda _: {"messages": [AIMessage(content=json.dumps(plan))]})
    with patch("sys.argv", ["gpsr-agent", "hello"]), patch("builtins.input") as prompt:
        agent.run_cli(fake)
    prompt.assert_not_called()
    assert not (knowledge.parent / "evaluations.jsonl").exists()


def test_feedback_includes_invalid_final_result(knowledge, monkeypatch):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", knowledge)
    plan = {"command": "x", "steps": [{"action": "teleport", "args": {}}]}
    fake = SimpleNamespace(invoke=lambda _: {"messages": [AIMessage(content=json.dumps(plan))]})
    with patch("sys.argv", ["gpsr-agent", "x", "--feedback"]):
        with patch("builtins.input", side_effect=["n", "Unsupported action"]):
            agent.run_cli(fake, "test:model")
    saved = json.loads((knowledge.parent / "evaluations.jsonl").read_text())
    assert saved["plan"] == plan
    assert saved["validation"]["valid"] is False


def test_proposal_validation():
    valid = proposal()
    assert valid.status == "proposed"
    for changes in ({"type": "code"}, {"status": "approved"}, {"evidence": []},
                    {"evidence": [""]}, {"description": " "}, {"type": "fact"}):
        with pytest.raises(ValidationError):
            Proposal.model_validate({**valid.model_dump(), **changes})


def test_approval_preserves_existing_knowledge_and_backups(knowledge):
    before = {path.name: path.read_bytes() for path in knowledge.glob("*.json")}
    fact = proposal("fact", fact={"subject": "shelf", "relation": "inside", "object": "office"})
    with patch("builtins.input", return_value="y"):
        review_proposals(Proposals(proposals=[proposal(), fact]), EVALUATIONS, knowledge)
    for name in ("facts.json", "precepts.json"):
        original = json.loads(before[name])
        updated = json.loads((knowledge / name).read_text())
        key = "facts" if name == "facts.json" else "precepts"
        assert updated[key][:-1] == original[key]
        assert {k: v for k, v in updated.items() if k != key} == {k: v for k, v in original.items() if k != key}
        assert next(knowledge.glob(name + ".*.bak")).read_bytes() == before[name]
    assert (knowledge / "competition_facts.json").read_bytes() == before["competition_facts.json"]


def test_rejection_and_development_suggestions_do_not_change_knowledge(knowledge):
    before = {p.name: p.read_bytes() for p in knowledge.glob("*.json")}
    proposals = Proposals(proposals=[proposal(), proposal("skill"), proposal("validator")])
    with patch("builtins.input", return_value="") as prompt:
        review_proposals(proposals, EVALUATIONS, knowledge)
    assert prompt.call_count == 1
    assert {p.name: p.read_bytes() for p in knowledge.glob("*.json")} == before
    records = [json.loads(line) for line in (knowledge.parent / "knowledge_proposals.jsonl").read_text().splitlines()]
    assert [r["review"] for r in records] == ["rejected", "development", "development"]


def test_duplicates_conflicts_and_invalid_evidence_are_blocked(knowledge):
    duplicate = proposal("fact", fact={"subject": "cup", "relation": "is_object", "object": True})
    conflict = proposal("fact", fact={"subject": "table", "relation": "inside", "object": "office"})
    precept = Proposal.model_validate({**proposal().model_dump(), "description": "detect before grasping!"})
    bad_evidence = Proposal.model_validate({**proposal().model_dump(), "evidence": ["unknown"]})
    before = {p.name: p.read_bytes() for p in knowledge.glob("*.json")}
    with patch("builtins.input", return_value="y") as prompt:
        review_proposals(Proposals(proposals=[duplicate, conflict, precept, bad_evidence]), EVALUATIONS, knowledge)
    prompt.assert_not_called()
    assert {p.name: p.read_bytes() for p in knowledge.glob("*.json")} == before


def test_extractor_uses_read_only_tools_and_structured_proposals(monkeypatch):
    from gpsr_agent import knowledge_extractor as extractor

    monkeypatch.setattr(extractor, "load_dotenv", lambda: None)
    monkeypatch.setenv("GPSR_MODEL", "test:model")
    with patch.object(extractor, "init_chat_model") as model:
        with patch.object(extractor, "create_agent") as create:
            extractor.build_agent()
    model.assert_called_once_with("test:model", temperature=0)
    kwargs = create.call_args.kwargs
    assert [tool.name for tool in kwargs["tools"]] == ["get_facts", "get_precepts", "get_capabilities"]
    assert kwargs["response_format"].schema is Proposals


def test_development_suggestions_are_not_duplicated(knowledge):
    batch = Proposals(proposals=[proposal("skill"), proposal("skill")])
    with patch("builtins.input") as prompt:
        review_proposals(batch, EVALUATIONS, knowledge)
        review_proposals(batch, EVALUATIONS, knowledge)
    prompt.assert_not_called()
    records = [json.loads(line) for line in (knowledge.parent / "knowledge_proposals.jsonl").read_text().splitlines()]
    assert sum(record["review"] == "development" for record in records) == 1
