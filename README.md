# castor-server

**Self-hosted, model-agnostic Anthropic Managed Agents API.**

Drop-in replacement for Anthropic's Managed Agents service that you run on
your own machine or VPC. Bring your own model (Anthropic, OpenRouter, OpenAI,
local LLMs via LiteLLM). Wire-compatible with the official `anthropic` SDK,
plus Castor extensions for budget control, speculative review, time-travel
fork, and HITL modify.

## 30-second offline demo

No API key required — uses the built-in `mock` model.

```bash
# Terminal 1
uv tool install castor-server   # or: pip install castor-server
castor-server run               # binds 0.0.0.0:8080
```

```bash
# Terminal 2
curl -s -X POST http://localhost:8080/v1/agents \
  -H 'content-type: application/json' \
  -d '{"name":"hello","model":"mock"}' \
  | python3 -m json.tool
```

```python
# Or from Python — works with any HTTP client
import httpx, json

base = "http://localhost:8080"
agent = httpx.post(f"{base}/v1/agents", json={"name": "hello", "model": "mock"}).json()
session = httpx.post(f"{base}/v1/sessions", json={"agent": agent["id"]}).json()
httpx.post(
    f"{base}/v1/sessions/{session['id']}/events",
    json={"events": [{"type": "user.message", "content": [{"type": "text", "text": "ping"}]}]},
)

# Stream the response
with httpx.stream("GET", f"{base}/v1/sessions/{session['id']}/events/stream") as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            print(json.loads(line[5:]))
```

You'll see `session.status_running → agent.message → session.status_idle` —
the entire pipeline working with zero external dependencies.

## Real models

Set one of these env vars before running an agent with a real model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # default model_map points here
export OPENROUTER_API_KEY=sk-or-...      # for OpenRouter routing
export OPENAI_API_KEY=sk-...             # for OpenAI
```

`castor-server run` prints which keys it detected on startup.

## Drop-in replacement for `anthropic-python`

If you already have code written against the Anthropic Managed Agents API,
just point your client at castor-server:

```python
from anthropic import Anthropic

# This is the only line you change.
client = Anthropic(base_url="http://localhost:8080", api_key="local")

# Everything else uses the standard anthropic SDK.
env = client.beta.environments.create(name="my-env")
agent = client.beta.agents.create(
    name="my-helper",
    model="claude-sonnet-4-6",
    system="You are a helpful assistant.",
    tools=[{"type": "agent_toolset_20260401"}],
)
session = client.beta.sessions.create(agent=agent.id, environment_id=env.id)
client.beta.sessions.events.send(
    session_id=session.id,
    events=[{
        "type": "user.message",
        "content": [{"type": "text", "text": "Hello!"}],
    }],
)
history = client.beta.sessions.events.list(session_id=session.id)
```

**Verified compatible with `anthropic-python` 0.93.0** for all CRUD operations
(`agents.*`, `environments.*`, `sessions.*`, `sessions.events.send/.list`).
See `scripts/sdk_check.py` for the full validation.

### Streaming events (SDK workaround required)

The `client.beta.sessions.events.stream()` method in `anthropic-python` 0.93.0
is broken — its internal `Stream` class is hardcoded for the Messages API
event names (`message_start`, `content_block_*`, etc.) and silently drops
managed agents events (`session.status_*`, `agent.message`, etc.).
This affects every server, including `api.anthropic.com`. Track upstream
fix: anthropics/anthropic-sdk-python.

**Workaround #1 — built-in helper (recommended):**

```python
from castor_server.client import stream_events  # also: astream_events

for event in stream_events(
    base_url="http://localhost:8080",
    session_id=session.id,
    api_key="local",
):
    print(event["type"], event)
    if event["type"] == "session.status_idle":
        break
```

**Workaround #2 — raw httpx (zero dependency):**

```python
import httpx, json

with httpx.stream(
    "GET",
    f"http://localhost:8080/v1/sessions/{session.id}/events/stream",
    headers={"x-api-key": "local"},
    timeout=60,
) as r:
    for line in r.iter_lines():
        if line.startswith("data:"):
            event = json.loads(line[5:])
            print(event)
            if event["type"] == "session.status_idle":
                break
```

**Workaround #3 — polling:** `client.beta.sessions.events.list(session_id=...)`
works fine via the SDK and returns the full event history. Poll on a timer
if you don't need real-time updates.

## Why use castor-server

| | Anthropic Managed Agents | castor-server |
|---|---|---|
| Hosting | Anthropic | Yourself (laptop / VPC / k8s) |
| Models | Claude only | Any (LiteLLM) |
| Pricing | $0.08 / active session-hour + token cost | Token cost only |
| Data residency | Anthropic's infra | Your network |
| Replay & deterministic re-execution | ❌ | ✅ |
| Budget control per session | ❌ | ✅ |
| Speculative scan (`/sessions/{id}/scan`) | ❌ | ✅ |
| Time-travel fork (`/sessions/{id}/fork`) | ❌ | ✅ |
| HITL approve / reject / modify | partial | ✅ |
| Wire-compatible with `anthropic-python` | — | ✅ |

## API surface

**Phase 1 — 100% Anthropic-compatible:**
- `POST/GET /v1/agents` — agent CRUD with versioning
- `POST/GET /v1/sessions` — session lifecycle
- `POST /v1/sessions/{id}/events` — `user.message`, `user.tool_confirmation`,
  `user.interrupt`, `user.custom_tool_result`
- `GET /v1/sessions/{id}/events/stream` — SSE stream of `agent.message`,
  `agent.tool_use`, `session.status_*`, `span.model_request_*`
- `POST/GET /v1/environments` — sandbox config

**Phase 2 — Castor-only extensions:**
- `POST /v1/sessions/{id}/scan` — speculative review of an agent run
- `POST /v1/sessions/{id}/fork` — fork from a previous step
- `GET /v1/sessions/{id}/budget` — real-time budget usage
- `user.tool_confirmation` with `result: "modify"` — HITL modify

**Built-in tools** (matches `agent_toolset_20260401`):
`bash`, `read`, `write`, `edit`, `glob`, `grep`, `web_fetch`, `web_search`.
All tool execution is sandboxed via [Roche](https://github.com/substratum-labs/roche).

## Authentication

- **Local dev (default):** no auth.
- **Production:** set `CASTOR_API_KEY=<your-key>` and clients must send
  `Authorization: Bearer <your-key>`.

## Configuration

All settings via environment variables (prefix `CASTOR_`):

| Var | Default | Description |
|---|---|---|
| `CASTOR_HOST` | `0.0.0.0` | Bind host |
| `CASTOR_PORT` | `8080` | Bind port |
| `CASTOR_DATABASE_URL` | `sqlite+aiosqlite:///castor_server.db` | SQLAlchemy URL |
| `CASTOR_API_KEY` | _(unset)_ | Global bearer token (none = open) |
| `CASTOR_ENABLE_BUDGETS` | `false` | Enforce per-session budgets |
| `CASTOR_DEBUG` | `false` | Debug logging |

## Development

```bash
uv sync                          # Install deps
uv run pytest                    # Run tests
uv run ruff check src/ tests/    # Lint
uv run ruff format src/ tests/   # Format
uv run python scripts/wire_check.py   # End-to-end wire format check
```

## Project layout

```
src/castor_server/
├── api/         # FastAPI routes
├── core/        # Session manager, kernel adapter, event bus, LLM adapter
├── models/      # Pydantic schemas (Anthropic-compatible)
├── store/       # SQLAlchemy persistence
└── tools/       # Built-in toolset
```

## License

Apache 2.0
