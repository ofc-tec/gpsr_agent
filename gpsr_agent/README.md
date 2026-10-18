# GPSR Agent v0

ROS-independent, tool-using planning agent for the ICRA 2027 workshop.

## Setup (Ubuntu)

```bash
cd ~/gpsr_agent
uv sync --all-packages --all-groups
# Keep credentials in ~/gpsr_agent/.env; never commit .env.
uv run --package gpsr-agent pytest gpsr_agent/tests -q
uv run --package gpsr-agent gpsr-agent 'Bring me a strawberry jello from the bedside table.'
```

The repository root is a uv workspace; `gpsr_agent/` is its Python project.
Both use the root `.venv` and `uv.lock`. Activate only the root environment
(`source ~/gpsr_agent/.venv/bin/activate`) if using shell activation.

## Local Ollama planner

Uses `ChatOllama` with the same prompt, fact queries, precepts, validation,
and tool-call output as the Gemini entry point. No Gemini credentials are needed.

```bash
cd ~/gpsr_agent
ollama pull llama3.1:latest
uv run --package gpsr-agent gpsr-agent-ollama 'Bring me a sponge from the kitchen table.'
# Equivalent module invocation:
uv run --package gpsr-agent python -m gpsr_agent.ollama_agent 'Bring me a sponge from the kitchen table.'
```

The server defaults to `http://localhost:11434`; start `ollama serve` if needed.
Override with `GPSR_OLLAMA_BASE_URL` and `GPSR_OLLAMA_MODEL`.
The model must support native tool calls. The default is `llama3.1:latest`; to try another model,
use a tool-capable model instead:

```bash
ollama pull qwen3:8b
GPSR_OLLAMA_MODEL=qwen3:8b uv run --package gpsr-agent gpsr-agent-ollama 'Bring me a sponge from the kitchen table.'
```

## Human feedback and knowledge extraction

From `~/gpsr_agent` (the uv workspace root):

```bash
# Normal planning behaves as before.
uv run python -m gpsr_agent.agent
# After the final result, accept or reject it and explain a rejection.
uv run python -m gpsr_agent.agent --feedback
# Review reusable improvements inferred from rejected evaluations.
uv run python -m gpsr_agent.knowledge_extractor
```

Ollama also accepts `--feedback` through the shared terminal interface.
Evaluations are appended to `gpsr_agent/evaluations.jsonl`, with original
instructions, generated plans, validation results, decisions, feedback, UTC
timestamps, model identifiers and evaluation IDs. Invalid final responses can
also be evaluated. Normal planning without the flag creates no evaluations.

The independent extractor uses the same `GPSR_MODEL` configuration as the main
planner, existing retrieval tools, and Pydantic structured proposals. It reads
both fact sources, precepts and capabilities. It has no write tools. Proposal
evidence must refer to rejected evaluations containing feedback.

Only explicit approval writes facts/precepts. Approving a fact confirms that
the reviewer independently verified it; feedback alone is not established world
knowledge. Existing entries and top-level JSON fields are preserved. Each write
creates a `.bak` snapshot and replaces the updated file atomically. Competition
facts are read-only. Exact duplicates and conflicting subject/relation facts are
blocked conservatively; paraphrases and semantic rule conflicts require human
review against the displayed existing precepts. No existing rules are replaced.

`gpsr_agent/knowledge_proposals.jsonl` records proposals and review decisions.
Skills and validators remain development suggestions there; the extractor never
implements code, capabilities, ROS changes, or Behavior Trees. No new dependencies
are required. A live extraction invokes the configured LLM.

## Design

- Eight atomic actions from the October 6 workshop design: `navigate_to`, `detect_object`, `detect_person`, `grasp`, `release`, `follow_person`, `say`, `listen`.
- JSON fact/precept knowledge layer. Seed facts are **examples**, not imported official CompetitionTemplate data.
- LangChain tool-calling agent using Gemini by default, with observable tool call names on stderr.
- Pydantic + deterministic argument-schema validator; currently **syntax and allowed-argument checks only**. Real-world grounding, precondition checks, and state tracking are TODO.
- No ROS or BT generator yet. Keep execution separate.

## Repo initialization

```bash
git init -b main
git add .
git commit -m 'Initialize GPSR agent planning core'
# If GitHub CLI is configured:
gh repo create gpsr-agent --private --source=. --remote=origin --push
```

## Next milestones

1. Import the actual CommandGenerator `CompetitionTemplate` ontology and archived command batch.
2. Enforce entity grounding and action preconditions; add explicit unknown/uncertain facts.
3. Add controlled observation events / proposed facts, rather than arbitrary LLM file writes.
4. Evaluate plans on generator commands; then add BT translation and a thin ROS 2 node.
