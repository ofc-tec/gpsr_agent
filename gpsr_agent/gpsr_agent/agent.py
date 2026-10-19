"""Tool-calling GPSR planning agent; no robot execution."""
import json
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from .context import DEFAULT_KNOWLEDGE, query_facts, query_precepts
from .validator import validate_plan
from .critic import DEFAULT_MODEL as DEFAULT_CRITIC_MODEL, evaluate_plan

@tool
def get_facts(subject: str = "", relation: str = "") -> str:
    """Query merged local and competition facts. Empty filters mean all facts."""
    return json.dumps(query_facts(subject or None, relation or None))

@tool
def get_precepts() -> str:
    """Retrieve rules governing GPSR planning and uncertainty."""
    return json.dumps(query_precepts())

@tool
def get_capabilities() -> str:
    """Get the approved GPSR atomic action vocabulary and parameter keys."""
    from .validator import ARG_KEYS
    return json.dumps({k: sorted(v) for k, v in ARG_KEYS.items()})

@tool
def check_plan(plan_json: str) -> str:
    """Validate a proposed plan represented as a JSON object encoded in a string."""
    try:
        result = validate_plan(json.loads(plan_json))
        return json.dumps(result)
    except ValueError as exc:
        return json.dumps({"valid": False, "errors": [str(exc)]})

SYSTEM = """You are a GPSR task PLANNER, not an executor. Use tools to inspect facts,
precepts and capabilities. Never invent robot skills. Determine a feasible sequence
of atomic actions, preserving the difference between expected and observed locations.
Do not claim robot actions happened. Call check_plan on your draft before finalizing.
Your final reply must be ONLY a JSON object with keys command, steps and assumptions;
steps contain action and args. If you cannot ground a command, return an empty steps
list and explain why in assumptions. For counting tasks, use count_objects(category, location) when available.
count_objects counts observed object instances, not object types.
It returns an integer count at execution time.
Do not invent counts or assume unsupported result binding.

The command field must contain the original user command. assumptions must be a
list of strings, never an object. Example plan structure:
{"command": "Find a sponge", "steps": [{"action": "detect_object", "args": {"object": "sponge"}}], "assumptions": []}
Query get_precepts and get_capabilities before drafting. For fact discovery use
get_facts with empty filters, or query a subject with an empty relation first.
Known relations include usually_found_at, inside, instruction_point, is_object,
is_location, is_room, and is_person. Do not invent relation names.
check_plan.plan_json must be a JSON-encoded string containing the ENTIRE proposed
plan object, not placeholder text. If check_plan returns valid=false, correct
the reported errors and check again. Finalize only the checked plan.
"""

def build_agent(model=None):
    load_dotenv()
    if model is None:
        #model_name = os.getenv("GPSR_MODEL", "google_genai:gemini-3.8-flash")
        model_name = os.getenv("GPSR_MODEL", "openai:gpt-5-mini")
        model = init_chat_model(model_name, temperature=0)
    return create_agent(model=model, tools=[get_facts, get_precepts, get_capabilities, check_plan], system_prompt=SYSTEM)

def save_evaluation(command, plan, validation, model):
    critique = None
    critic_error = None
    try:
        critique = evaluate_plan(command, plan, validation, query_facts(),
                                 query_precepts(), json.loads(get_capabilities.invoke({})))
        print(f"Critic: {'accepted' if critique.accepted else 'rejected'}")
        for issue in critique.issues:
            print(f"  - {issue}")
        print(f"Recommendation: {critique.recommendation}")
        print(f"Proposed feedback: {critique.feedback}")
    except Exception as exc:
        # Provider/output failures must not prevent the independent human review.
        critic_error = str(exc) or type(exc).__name__
        print(f"Critic unavailable: {critic_error}")
        print("No Critic decision was made. Human evaluation can continue.")
    review_action = "manual"
    if critique is not None:
        choices = {"a": "accept", "accept": "accept", "e": "edit", "edit": "edit",
                   "r": "reject", "reject": "reject"}
        choice = input("Review Gemini evaluation: Accept / Edit / Reject [a/e/r]: ").strip().lower()
        while choice not in choices:
            choice = input("Choose Accept, Edit, or Reject [a/e/r]: ").strip().lower()
        review_action = choices[choice]
    if review_action == "accept":
        # Explicit review adopts both fields, not just the machine's verdict.
        accepted, feedback = critique.accepted, critique.feedback
        feedback_source = "critic"
    else:
        editing = review_action == "edit"
        default = critique.accepted if editing else None
        suffix = ("Y/n" if default else "y/N") if editing else "y/n"
        answer = input(f"Accept plan? [{suffix}]: ").strip().lower()
        while answer not in {"y", "yes", "n", "no"} and not (editing and answer == ""):
            answer = input(f"Accept plan? [{suffix}]: ").strip().lower()
        accepted = default if answer == "" else answer in {"y", "yes"}
        prompt = "Feedback [Enter to keep Gemini's text]: " if editing else "Feedback: "
        feedback = input(prompt).strip()
        if editing and not feedback:
            feedback = critique.feedback
        while not accepted and not feedback:
            feedback = input("Feedback is required for a rejected plan: ").strip()
        feedback_source = "human_edited" if editing else "human"
    evaluation = {
        "id": f"evaluation_{uuid4().hex}",
        "instruction": command,
        "plan": plan,
        "validation": validation,
        "accepted": accepted,
        "feedback": feedback,
        "feedback_reviewed": True,
        "feedback_source": feedback_source,
        "review_action": review_action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model or os.getenv("GPSR_MODEL", "openai:gpt-5-mini"),
        "critic": critique.model_dump() if critique is not None else None,
        "critic_model": os.getenv("GPSR_CRITIC_MODEL", DEFAULT_CRITIC_MODEL),
    }
    if critic_error is not None:
        evaluation["critic_error"] = critic_error
    with (DEFAULT_KNOWLEDGE.parent / "evaluations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
    print(f"Saved evaluation: {evaluation['id']}")

def run_cli(agent, model_identifier=None):
    feedback = "--feedback" in sys.argv[1:]
    command = " ".join(arg for arg in sys.argv[1:] if arg != "--feedback").strip() or input("GPSR> ").strip()
    result = agent.invoke({"messages": [{"role": "user", "content": command}]})
    for msg in result["messages"]:
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                print(f"Tool: {call['name']}", file=sys.stderr)
        elif getattr(msg, "type", None) == "tool":
            if getattr(msg, "status", None) == "error":
                print(f"Tool error: {msg.text}", file=sys.stderr)
            elif msg.name == "check_plan":
                try:
                    checked = json.loads(msg.text)
                    status = "passed" if checked["valid"] else "failed"
                    print(f"Validation: {status}", file=sys.stderr)
                    for error in checked.get("errors", []):
                        print(f"  - {error}", file=sys.stderr)
                except (ValueError, KeyError, TypeError):
                    print("Validation error: unreadable tool result", file=sys.stderr)

    # Keep the model/tool protocol as JSON; format only the terminal display.
    text = result["messages"][-1].text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1])
    try:
        generated = json.loads(text)
        checked = validate_plan(generated)
    except ValueError as exc:
        print(f"Plan error: expected a JSON plan ({exc})", file=sys.stderr)
        if feedback:
            save_evaluation(command, text, {"valid": False, "errors": [str(exc)]}, model_identifier)
        return
    if not checked["valid"]:
        print("Final validation: failed", file=sys.stderr)
        for error in checked["errors"]:
            print(f"  - {error}", file=sys.stderr)
        if feedback:
            print(f"Generated plan: {text}")
            save_evaluation(command, generated, checked, model_identifier)
        return

    plan = checked["plan"]
    print("Final validation: passed", file=sys.stderr)
    print(f"Plan: {plan['command']}")
    if not plan["steps"]:
        print("No actions proposed.")
    for number, step in enumerate(plan["steps"], 1):
        args = ", ".join(f"{key}={value!r}" for key, value in step["args"].items())
        print(f"{number}. {step['action']}({args})")
    for assumption in plan["assumptions"]:
        print(f"Assumption: {assumption}")
    if feedback:
        save_evaluation(command, generated, checked, model_identifier)

def main():
    agent = build_agent()
    run_cli(agent, os.getenv("GPSR_MODEL", "openai:gpt-5-mini"))

if __name__ == "__main__":
    main()
