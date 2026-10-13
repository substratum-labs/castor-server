# T-306-C Physical Management Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the T-306-C simulated control plane with live per-session `castord` processes, frozen AISA envelopes, and hostile tests that observe durable kernel facts.

**Architecture:** `castord` owns the durable Core and exposes agent `ipc.sock` plus host-only `control.sock`. `castor-server` only validates tenancy/RBAC, supervises the child, and sends management requests over the control channel; it never proxies Agent turn operations.

**Tech Stack:** Rust `castor-kernel`, Unix domain sockets, framed JSON AISA v0.1, Python asyncio/FastAPI, pytest.

**Spec:** `../substratum-internal/design/castor/architecture/next/l5_castor_server_decision_management_rfc.md`

## Global Constraints

- One physical `castord` child and one isolated storage root per active tenant/session.
- Probe `.c01-writer.lock` with advisory flock without retaining the lock.
- Agent requests use `ipc.sock`; privileged management operations use `control.sock`.
- Wire messages are exactly `SyscallRequest { request_id, op, payload }` and `SyscallResponse { request_id, status, outcome, error }`, framed by bounded u32 big-endian JSON.
- Persist a fence successfully before any termination or idle harvest signal.
- Retain the 216 legacy mock-track tests; M1–M18 must use real filesystem, subprocess, sockets, or journal evidence where the contract claims such evidence.

---

### Task 1: Validate the kernel control-channel contract

**Files:**
- Modify: `../castor/kernel/tests/host_aisa_gateway_contract.rs`
- Modify: `../castor/kernel/tests/governed_turn_composition_contract.rs`
- Modify: `../castor/kernel/src/bin/castord.rs`

**Interfaces:**
- Consumes: `SyscallRequest`, `SyscallResponse`, `D1GovernedTurnAuthority`.
- Produces: channel-separated `GrantCapability`, privileged `RevokeCapability`, `ResolveQuarantinedDispute`, `InspectJournal`, and `GetProjectionSummary` behavior.

- [ ] **Step 1: Write/strengthen failing Rust contract assertions** for a control request that persists the requested Core fact, an Agent request rejected with `UnauthorizedOpcode`, and a quarantined scope that remains locked after process recovery until a real resolve request.
- [ ] **Step 2: Run the focused Rust tests** with `cargo test --test host_aisa_gateway_contract --test governed_turn_composition_contract` and confirm failures name the missing behavior.
- [ ] **Step 3: Implement the minimum kernel dispatch/state changes** in `castord.rs` and composition authority needed for those requests, preserving the frozen framed schema.
- [ ] **Step 4: Re-run the focused Rust tests**, then `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings`.

### Task 2: Replace the Python UDS simulation with a physical supervisor

**Files:**
- Modify: `src/castor_server/core/aisa_client.py`
- Modify: `src/castor_server/core/castord_manager.py`
- Test: `tests/test_decision_management_contract.py`

**Interfaces:**
- Consumes: absolute path to the built `castord` binary and `ManagedCastordSession` socket/root paths.
- Produces: `AisaResponse(request_id, status, outcome, error_code)`, `provision`, `restart`, `fence_and_reap`, and actual child PID state.

- [ ] **Step 1: Change one hostile test to assert that a provisioned session has a live child PID, regular socket inodes, and a valid framed `SyscallResponse`; run it and confirm it fails against the asyncio simulation.**
- [ ] **Step 2: Implement request-id generation and exact request/response serialization in `AisaClient`; map the gateway `error.code` without introducing a parallel envelope.**
- [ ] **Step 3: Implement child spawn with `--storage-root`, `--socket`, and `--control-socket`, readiness polling, non-retained `.c01-writer.lock` flock probe, restart, and fence-before-TERM/KILL/waitpid reaping.**
- [ ] **Step 4: Run the changed hostile test and the full contract file; fix only the observed protocol/lifecycle mismatches.**

### Task 3: Route management intents and inspection through live control sockets

**Files:**
- Modify: `src/castor_server/api/management.py`
- Modify: `src/castor_server/app.py`
- Test: `tests/test_decision_management_contract.py`

**Interfaces:**
- Consumes: tenant/session manager lookup and `AisaClient(control_socket, CONTROL)`.
- Produces: HTTP 201 only for Core `EntryPersisted`, HTTP 200 for Core idempotence, journal-derived inspection/audit responses, and lifecycle responses observed after durable fences.

- [ ] **Step 1: Replace the in-memory-decision assertions with a failing test that creates a session over HTTP, submits a control operation, and observes its matching `core-journal.log`/inspection response.**
- [ ] **Step 2: Make session creation provision the physical child; map grant/revoke/resolve requests into the frozen control opcode payloads and record audit identifiers only after Core response.**
- [ ] **Step 3: Make inspection and journal endpoints request live `GetProjectionSummary`/`InspectJournal`; convert socket/integrity faults to fail-closed HTTP errors.**
- [ ] **Step 4: Make terminate/harvest/delete call the manager’s fence-and-reap path and prove postconditions with PID and journal evidence.**
- [ ] **Step 5: Run M1–M18 and preserve the existing RBAC, tenancy, and payload-bound checks.**

### Task 4: Complete evidence and regression verification

**Files:**
- Modify: `tests/test_decision_management_contract.py`

**Interfaces:**
- Consumes: real session fixtures and all routes above.
- Produces: non-tautological M1–M18 tests for CRC bitrot, crash recovery, direct Agent socket use, server outage, and durable fence ordering.

- [ ] **Step 1: Make M4 flip a byte in an actual framed `core-journal.log` record and assert live inspection fails closed.**
- [ ] **Step 2: Make M9 create a real quarantined dispute, kill/restart the child, and prove a same-scope action remains rejected until control resolution.**
- [ ] **Step 3: Make M7/M17 send valid Agent requests directly to `ipc.sock` while no management request counter or FastAPI listener participates.**
- [ ] **Step 4: Make M14 compare the live projection summary after journal replay and M15/M18 inspect the persisted fence before reaping.**
- [ ] **Step 5: Run `uv run pytest`, scoped Ruff check/format, `git diff --check`, and the full kernel test suite before committing each repository’s scoped changes.**
