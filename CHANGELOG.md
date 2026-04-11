# Changelog

All notable changes to Castor Server are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — 0.1.0 (public preview)

First public release. 27 commits, 149 tests passing.

### Wire compatibility

- 100% wire-compatible with Anthropic Managed Agents API. Verified against
  `anthropic-python` 0.93.0 — change one line (`base_url=`) and existing
  Anthropic agent code runs against Castor Server.
- Endpoints implemented: `/v1/agents`, `/v1/sessions`, `/v1/sessions/{id}/events`,
  `/v1/sessions/{id}/events/stream` (SSE), `/v1/environments`, `/v1/models`,
  `/v1/files`.
- Event flow: `user.message`, `user.interrupt`, `user.tool_confirmation`,
  `user.custom_tool_result` inbound; `agent.message`, `agent.tool_use`,
  `agent.tool_result`, `agent.custom_tool_use`, `session.status_*`,
  `span.model_request_*` outbound.
- Built-in toolset matching `agent_toolset_20260401`: `bash`, `read`, `write`,
  `edit`, `glob`, `grep`, `web_fetch`, `web_search`.
- MCP toolset runtime: discover and call tools on remote MCP servers.
- Custom tools via protocol layering — server handles the full Anthropic
  semantics while the kernel sees a simplified `external_input` syscall, so
  replay, budget, and scan all cover custom-tool calls too.

### Castor extensions (Anthropic Managed Agents can't do these)

- `GET /v1/sessions/{id}/budget` — real-time per-resource budget usage from
  the kernel's `Capability` table.
- `POST /v1/sessions/{id}/scan` — speculative execution review; dry-runs the
  agent and returns the syscalls it _would_ make, flagged for risk.
- `POST /v1/sessions/{id}/fork` — time-travel fork. Rewind a session to any
  step and start a new timeline from that point.
- `user.tool_confirmation` with `result: "modify"` — HITL modify flow: user
  tells the agent "do Y instead of X", agent sees the feedback on replay
  and re-plans.

### Runtime

- Built on [castor-kernel](https://pypi.org/project/castor-kernel/) — a
  deterministic, checkpoint-based agent microkernel. All LLM calls and
  tool invocations route through `proxy.syscall()` into a journal, enabling
  replay, fork, scan, and budget enforcement as first-class primitives.
- `session_manager` decomposed by Ring: pure Ring 2 (API/app layer)
  delegating execution to Ring 0/1 in the kernel. Refactored from 880 lines
  to ~400 lines in the rewrite.
- Agent function (`agent_fn.py`) runs inside the kernel and implements the
  standard ReAct loop. It's a convenience helper, not a kernel feature —
  it doesn't belong to any Ring and users can replace it with their own
  agent logic.
- SSE events emitted from the agent function (Ring 3), not via a kernel
  callback hook (which would violate the Ring model and break replay
  determinism).

### Sandbox isolation (optional)

- Install with `pip install 'castor-server[sandbox]'` to enable Roche-backed
  sandbox isolation. Without it, tools run directly on the host
  (development/demo mode).
- Tools use a context variable to detect the active sandbox, so the same
  `bash`, `read`, `write`, `edit`, `glob`, `grep`, `web_fetch` implementations
  execute inside a Roche container when a session has an `environment_id`.
- `POST /v1/environments` CRUD for declaring sandbox configurations (image,
  memory, cpus, network, writable, network_allowlist).
- Resource mounting: `github_repository` resources are `git clone`'d into
  the sandbox at session creation, with optional `authorization_token` and
  `checkout` ref. `file` resources copy uploaded blobs from the Files API.
- `HITL approve` flow now sets the sandbox context **before** calling
  `kernel.approve()`, ensuring approved destructive tools execute inside
  the sandbox rather than on the host.

### Database

- SQLite by default (`castor_server.db`). Install with `pip install
  'castor-server[postgres]'` and set `CASTOR_DATABASE_URL` to
  `postgresql+asyncpg://...` for production.
- Verified: 149/149 tests pass against both SQLite in-memory and
  PostgreSQL 16. Switch backends with a single env var — no schema changes.
- Background tasks dispatched by `session_manager` now get a fresh DB
  session via the module-level factory. Reusing the request's session
  caused deadlocks on PostgreSQL (SQLite in-memory hid the bug).

### Authentication

- Optional global API key via `CASTOR_API_KEY` environment variable. When
  set, every endpoint except `/health` requires `Authorization: Bearer <key>`.
  When unset, no auth is required (development mode).
- Intentionally _not_ multi-tenant — a single global key. Real multi-tenancy
  with isolated tenants is on the roadmap.

### Developer experience

- 30-second offline demo with `model="mock"` — runs the entire agent
  pipeline (agents, sessions, events, kernel, SSE) with zero external
  dependencies. No API key required.
- Startup logs which LLM provider keys were detected, or nudges you toward
  `model="mock"` if none are set.
- CLI: `castor-server run` starts the server, `castor-server deploy` prints
  a Docker-compose template.
- `castor_server.client.stream_events` — a 5-line httpx-based SSE helper
  that works around a bug in `anthropic-python` 0.93.0's `Stream` class
  (see below).
- Dockerfile + `docker-compose.yml` for one-command deployment.

### Upstream bug reported

- Discovered that `anthropic-python` 0.93.0's `client.beta.sessions.events.stream()`
  silently drops all Managed Agents events. Root cause: `Stream.__stream__` is
  hardcoded for Messages API event names (`message_start`, `content_block_*`,
  etc.), so all `session.status_*` / `agent.message` / `agent.tool_*` events
  fall through the if-chain and are discarded. Affects every Managed Agents
  server, including `api.anthropic.com`. Minimal repro in
  `scripts/sdk_stream_proof.py`; reported to anthropics/anthropic-sdk-python.
- `castor_server.client.stream_events` provides a drop-in workaround that
  yields the full event stream correctly.

### Gaps (not yet implemented)

- Vault API
- Some Skills features
- Web search tool (stub — needs backend configuration)
- Horizontal scaling (Redis EventBus, distributed locks, worker pool)
- OpenTelemetry / Prometheus metrics
- GitHub Actions CI
