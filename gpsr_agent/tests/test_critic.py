import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from gpsr_agent import agent, critic
from gpsr_agent.knowledge_extractor import read_evaluations


@pytest.mark.parametrize("accepted,override", [(True, None), (False, "google_genai:test")])
def test_independent_structured_critic(accepted, override, monkeypatch):
    monkeypatch.setattr(critic, "load_dotenv", lambda: None)
    monkeypatch.setenv("GPSR_MODEL", "openai:planner")
    if override:
        monkeypatch.setenv("GPSR_CRITIC_MODEL", override)
    else:
        monkeypatch.delenv("GPSR_CRITIC_MODEL", raising=False)
    payload = {
        "instruction": "Find a sponge", "plan": {"steps": [{"action": "detect_object", "args": {"object": "sponge"}}]},
        "validation": {"valid": True}, "facts": [{"subject": "sponge"}],
        "precepts": [{"instruction": "Detect before grasping"}],
        "capabilities": {"detect_object": ["object"]},
    }
    original = deepcopy(payload)
    critique = {"accepted": accepted, "issues": [] if accepted else ["Missing verification"],
                "recommendation": "No change" if accepted else "Verify the result",
                "feedback": "Plan fulfills the task." if accepted else "Missing verification."}
    with patch.object(critic, "init_chat_model") as model:
        with patch.object(critic, "create_agent") as create:
            create.return_value.invoke.return_value = {"structured_response": critique}
            result = critic.evaluate_plan(**payload)
    assert result.model_dump() == critique
    assert payload == original
    model.assert_called_once_with(override or critic.DEFAULT_MODEL, temperature=0)
    assert create.call_args.kwargs["tools"] == []
    strategy = create.call_args.kwargs["response_format"]
    assert strategy.schema is critic.Critique
    assert strategy.handle_errors is False
    request = create.return_value.invoke.call_args.args[0]
    assert json.loads(request["messages"][0]["content"]) == payload


def test_without_feedback_never_invokes_critic(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", tmp_path / "knowledge")
    monkeypatch.setattr("sys.argv", ["gpsr-agent", "say hello"])
    plan = {"command": "say hello", "steps": []}
    planner = SimpleNamespace(invoke=lambda _: {"messages": [AIMessage(content=json.dumps(plan))]})
    with patch.object(agent, "evaluate_plan") as evaluate, patch("builtins.input") as prompt:
        agent.run_cli(planner)
    evaluate.assert_not_called()
    prompt.assert_not_called()
    assert not (tmp_path / "evaluations.jsonl").exists()


@pytest.mark.parametrize("plan,valid", [
    ({"command": "say hello", "steps": [{"action": "say", "args": {"text": "hello"}}], "assumptions": []}, True),
    ({"command": "say hello", "steps": [{"action": "teleport", "args": {}}], "assumptions": []}, False),
    ("malformed final response", False),
])
def test_feedback_reviews_full_plan_before_human(tmp_path, monkeypatch, capsys, plan, valid):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", tmp_path / "knowledge")
    monkeypatch.setattr("sys.argv", ["gpsr-agent", "say hello", "--feedback"])
    monkeypatch.setenv("GPSR_CRITIC_MODEL", "google_genai:critic-test")
    facts, precepts = [{"subject": "sponge", "relation": "is_object", "object": True}], [{"instruction": "Be truthful"}]
    monkeypatch.setattr(agent, "query_facts", lambda: facts)
    monkeypatch.setattr(agent, "query_precepts", lambda: precepts)
    text = plan if isinstance(plan, str) else json.dumps(plan)
    planner = SimpleNamespace(invoke=lambda _: {"messages": [AIMessage(content=text)]})

    def human(prompt):
        if prompt.startswith("Review Gemini"):
            output = capsys.readouterr().out
            assert "Critic: rejected" in output
            assert "Missing verification" in output
            assert "Recommendation:" in output
            assert "Proposed feedback:" in output
            return "reject"
        if prompt.startswith("Accept plan?"):
            return "y"  # Humans can disagree with a rejecting Critic.
        return "The plan is acceptable."

    with patch.object(agent, "evaluate_plan", return_value=critic.Critique(
        accepted=False, issues=["Missing verification"], recommendation="Verify success",
        feedback="The plan lacks arrival verification."
    )) as evaluate, patch("builtins.input", side_effect=human):
        agent.run_cli(planner, "openai:planner")
    instruction, actual_plan, validation, actual_facts, actual_precepts, capabilities = evaluate.call_args.args
    assert instruction == "say hello"
    assert actual_plan == plan
    assert validation["valid"] is valid
    assert actual_facts == facts and actual_precepts == precepts
    assert "say" in capabilities
    saved = json.loads((tmp_path / "evaluations.jsonl").read_text())
    assert saved["accepted"] is True
    assert saved["feedback"] == "The plan is acceptable."
    assert saved["critic"]["accepted"] is False
    assert saved["critic_model"] == "google_genai:critic-test"
    assert saved["model"] == "openai:planner"
    assert saved["plan"] == plan
    assert saved["validation"] == validation
    # A machine rejection is not a human rejection for the Knowledge Extractor.
    assert read_evaluations(tmp_path / "evaluations.jsonl") == []


def test_api_failure_allows_human_feedback_and_extractor(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", tmp_path / "knowledge")
    monkeypatch.setattr("sys.argv", ["gpsr-agent", "hello", "--feedback"])
    planner = SimpleNamespace(invoke=lambda _: {"messages": [AIMessage(content='{"command":"hello","steps":[]}')]})

    def human(prompt):
        if prompt.startswith("Accept plan?"):
            assert "Critic unavailable: API quota exhausted" in capsys.readouterr().out
            return "n"
        return "The requested delivery is missing."

    with patch.object(agent, "evaluate_plan", side_effect=RuntimeError("API quota exhausted")):
        with patch("builtins.input", side_effect=human):
            agent.run_cli(planner, "openai:planner")
    saved = json.loads((tmp_path / "evaluations.jsonl").read_text())
    assert saved["critic"] is None
    assert saved["critic_error"] == "API quota exhausted"
    assert saved["accepted"] is False
    assert saved["feedback"] == "The requested delivery is missing."
    assert saved["review_action"] == "manual"
    assert saved["feedback_source"] == "human"
    assert saved["feedback_reviewed"] is True
    assert read_evaluations(tmp_path / "evaluations.jsonl") == [saved]


@pytest.mark.parametrize("critic_verdict,answers,verdict,text,source,action", [
    (False, ["accept"], False, "Missing verification.", "critic", "accept"),
    (True, ["a"], True, "Missing verification.", "critic", "accept"),
    (False, ["edit", "y", "Verification is unnecessary here."], True,
     "Verification is unnecessary here.", "human_edited", "edit"),
    (True, ["e", "n", "Delivery is missing."], False, "Delivery is missing.", "human_edited", "edit"),
    (False, ["edit", "", ""], False, "Missing verification.", "human_edited", "edit"),
    (False, ["reject", "y", "The instruction is satisfied."], True,
     "The instruction is satisfied.", "human", "reject"),
    (True, ["r", "n", "The instruction is not satisfied."], False,
     "The instruction is not satisfied.", "human", "reject"),
])
def test_accept_edit_reject_evaluations(tmp_path, monkeypatch, critic_verdict, answers, verdict, text, source, action):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", tmp_path / "knowledge")
    original = critic.Critique(accepted=critic_verdict, issues=["Missing verification"],
                              recommendation="Verify arrival", feedback="Missing verification.")
    snapshot = original.model_dump()
    with patch.object(agent, "evaluate_plan", return_value=original):
        with patch("builtins.input", side_effect=answers) as prompt:
            agent.save_evaluation("guide someone", {"steps": []}, {"valid": True}, "test:planner")
    assert prompt.call_count == len(answers)
    saved = json.loads((tmp_path / "evaluations.jsonl").read_text())
    assert saved["accepted"] is verdict
    assert saved["feedback"] == text
    assert saved["critic"] == snapshot == original.model_dump()
    assert saved["feedback_source"] == source
    assert saved["review_action"] == action
    assert saved["feedback_reviewed"] is True
    assert read_evaluations(tmp_path / "evaluations.jsonl") == ([] if verdict else [saved])


def test_review_requires_explicit_action_verdict_and_rejection_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "DEFAULT_KNOWLEDGE", tmp_path / "knowledge")
    review = critic.Critique(accepted=True, issues=[], recommendation="None", feedback="Correct plan.")
    answers = ["", "invalid", "reject", "", "n", "", "Human rejection reason"]
    with patch.object(agent, "evaluate_plan", return_value=review):
        with patch("builtins.input", side_effect=answers) as prompt:
            agent.save_evaluation("task", {}, {}, "test:planner")
    assert prompt.call_count == len(answers)
    saved = json.loads((tmp_path / "evaluations.jsonl").read_text())
    assert saved["accepted"] is False
    assert saved["feedback"] == "Human rejection reason"


def test_extractor_excludes_unreviewed_machine_feedback_but_keeps_legacy(tmp_path):
    base = {"accepted": False, "feedback": "Missing verification."}
    records = [
        {**base, "id": "legacy"},
        {**base, "id": "unreviewed", "feedback_reviewed": False},
        {**base, "id": "machine_only", "feedback_source": "critic"},
        {**base, "id": "adopted", "feedback_source": "critic", "feedback_reviewed": True, "review_action": "accept"},
    ]
    path = tmp_path / "evaluations.jsonl"
    path.write_text("".join(json.dumps(item) + "\n" for item in records))
    assert [item["id"] for item in read_evaluations(path)] == ["legacy", "adopted"]
