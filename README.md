# Research Agent Pro — HITL Research Agent

## Quick Start

```bash
# 1. Install uv (if not already installed)
curl -Lsf https://astral.sh/uv/install.sh | sh

# 2. Set up environment
cp .env.example .env          # edit MODEL= as needed

# 3. Install dependencies
uv sync

# 4. Pull your Ollama model
ollama pull qwen2.5:7b        # or any model you set in .env

# 5. Run
uv run python research_agent_pro.py
```

## HITL Controls

At every tool call the agent wants to make, you'll see:

```
━━━ HITL GATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Tool   : web_search
  Args   : {"query": "latest LLM benchmarks 2025"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Decision [a]pprove / [e]dit / [r]eject :
```

| Key | Action |
|-----|--------|
| `a` (or Enter) | Approve — run the tool as-is |
| `e` | Edit — modify query args before running |
| `r` | Reject — skip this tool call (agent is told why) |

## Architecture

```
User query
    │
    ▼
┌─────────┐   tool_calls?   ┌──────────────────┐
│  agent  │ ──────────────► │ human_approval   │
│  (LLM)  │ ◄────────────── │  INTERRUPT here  │
└─────────┘  ToolMessages   └──────────────────┘
    │                              │
    │ no tool_calls                │ Command(resume={action,args,reason})
    ▼                              │
   END                       SqliteSaver checkpoint
```

State is persisted to `research_agent.db` (SQLite) so every interaction
is recoverable.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen2.5:7b` | Ollama model name |
| `TEMPERATURE` | `0.7` | LLM temperature |
| `DB_PATH` | `research_agent.db` | SQLite checkpoint file |

