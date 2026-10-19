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
# Review Gemini's proposed verdict and written feedback.
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

### Optional semantic Critic

`--feedback` runs an independent Gemini Critic after deterministic validation and
before human evaluation, including for invalid final plans. It receives the full
instruction, plan, validation, both fact sources, precepts and capabilities. It
checks task completion and execution-time mechanisms, rather than requiring
future perception results to be known during planning. It does not repair plans.

```dotenv
GPSR_MODEL=openai:gpt-5-mini
GPSR_CRITIC_MODEL=google_genai:gemini-3.8-flash
```

Use the existing Gemini credentials in `.env`, alongside the planner credentials.

```bash
uv run python -m gpsr_agent.agent --feedback
```

After displaying Gemini's proposed plan verdict and written feedback, the terminal
offers **Accept / Edit / Reject**. Accept adopts both fields as human-reviewed
feedback; accepting a rejecting evaluation still means rejecting the plan. Edit
allows changing the verdict and text (Enter keeps the proposed value). Reject
declines the assessment and requires your own explicit plan verdict and feedback.
Written feedback is required for rejected plans; no evaluation is adopted by default.

The evaluation log preserves the original `critic` and `critic_model` separately
from the final human-reviewed `accepted` and `feedback` fields. It also records
`review_action` (accept/edit/reject/manual), `feedback_source`
(critic/human_edited/human), and `feedback_reviewed`. The Critic cannot review its
own feedback on behalf of the human. If it fails, the error is displayed and saved
as `critic_error`, with `critic` set to null, and manual human evaluation continues.
The Knowledge Extractor processes rejected plans with human-reviewed feedback,
including explicitly adopted Gemini feedback. Unreviewed machine feedback is
excluded; legacy entries containing direct human feedback remain supported.
Knowledge proposals still require their separate approval before any fact/precept
update. Reviewing an evaluation never automatically establishes world knowledge.
Without `--feedback`, the Critic is never invoked. The Critic has no write tools
and cannot modify knowledge, capabilities, or plans.

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
