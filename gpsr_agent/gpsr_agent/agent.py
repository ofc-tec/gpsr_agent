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
list and explain why in assumptions. Count commands can use detect_person with
attributes and say to report the result; never invent a count_people skill.

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
    answer = input("Accept plan? [Y/n]: ").strip().lower()
    while answer not in {"", "y", "yes", "n", "no"}:
        answer = input("Accept plan? [Y/n]: ").strip().lower()
    accepted = answer in {"", "y", "yes"}
    evaluation = {
        "id": f"evaluation_{uuid4().hex}",
        "instruction": command,
        "plan": plan,
        "validation": validation,
        "accepted": accepted,
        "feedback": "" if accepted else input("Feedback: ").strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model or os.getenv("GPSR_MODEL", "openai:gpt-5-mini"),
    }
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
