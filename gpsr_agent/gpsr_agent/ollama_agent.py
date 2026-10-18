"""Local Ollama entry point using the shared GPSR planner and tools."""
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langsmith import tracing_context
from ollama import ResponseError

from .agent import build_agent as build_shared_agent, run_cli


def build_agent():
    load_dotenv()
    options = {"model": os.getenv("GPSR_OLLAMA_MODEL", "llama3.1:latest"),
               "temperature": 0}
    if os.getenv("GPSR_OLLAMA_BASE_URL"):
        options["base_url"] = os.environ["GPSR_OLLAMA_BASE_URL"]
    model = ChatOllama(**options)
    with tracing_context(enabled=False):
        return build_shared_agent(model=model)


def main():
    try:
        with tracing_context(enabled=False):
            agent = build_agent()
            run_cli(agent, "ollama:" + os.getenv("GPSR_OLLAMA_MODEL", "llama3.1:latest"))
    except ResponseError as exc:
        if "does not support tools" in str(exc).lower():
            raise SystemExit(
                "This Ollama model does not support the planner's tool calls. "
                "Run ollama pull llama3.1:latest, then set "
                "GPSR_OLLAMA_MODEL=llama3.1:latest and retry."
            ) from None
        raise


if __name__ == "__main__":
    main()
