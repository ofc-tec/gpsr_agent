# GPSR Agent

Standalone language-to-action planning prototype for General Purpose Service
Robots, supporting the ICRA 2027 workshop *From Language to Action: Agentic AI
and Behavior Trees*.

The planner uses facts and precepts to propose atomic robot actions and checks
plans with Pydantic and deterministic validation. A separate Knowledge Extractor
proposes reusable improvements from human feedback, with human approval before
knowledge updates. ROS 2 execution and Behavior Tree generation are not included.

## Quick start

Requires Python 3.12+ and uv.

```bash
uv sync --all-packages --all-groups
cp gpsr_agent/.env.example .env
# Configure provider credentials and GPSR_MODEL in .env.
uv run python -m gpsr_agent.agent
uv run python -m gpsr_agent.agent --feedback
uv run python -m gpsr_agent.knowledge_extractor
```

Local Ollama entry point:

```bash
ollama pull llama3.1:latest
uv run python -m gpsr_agent.ollama_agent
```

See [project documentation](gpsr_agent/README.md) for configuration, knowledge
files, validation limitations, and approval behavior. The workspace root manages
the environment and lockfile; application code lives in `gpsr_agent/gpsr_agent/`.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest gpsr_agent/tests -q
```

Credentials, local environments, evaluations, proposal history, and knowledge
backups are excluded from Git.
