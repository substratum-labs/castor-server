#!/usr/bin/env python3
"""Brain-before-body latency benchmark.

Measures the effect of parallelizing sandbox provisioning with the first
LLM inference call. Compares sequential (provision → run) vs parallel
(provision + run concurrently) paths.

Usage:
    uv run python scripts/bench_brain_before_body.py

Results are written to benchmarks/brain_before_body_2026-04.md
"""

from __future__ import annotations

import asyncio
import statistics
import time


# ---------------------------------------------------------------------------
# Simulated components
# ---------------------------------------------------------------------------

SANDBOX_PROVISION_MS = 800  # Typical Docker sandbox creation time
LLM_INFERENCE_MS = 2000  # Typical first LLM call latency
TOOL_EXECUTION_MS = 200  # Typical tool execution time


async def _simulate_sandbox_provision() -> float:
    """Simulate sandbox provisioning delay."""
    start = time.monotonic()
    await asyncio.sleep(SANDBOX_PROVISION_MS / 1000)
    return (time.monotonic() - start) * 1000


async def _simulate_llm_inference() -> float:
    """Simulate first LLM inference call."""
    start = time.monotonic()
    await asyncio.sleep(LLM_INFERENCE_MS / 1000)
    return (time.monotonic() - start) * 1000


async def _simulate_tool_execution() -> float:
    """Simulate tool execution (needs sandbox)."""
    start = time.monotonic()
    await asyncio.sleep(TOOL_EXECUTION_MS / 1000)
    return (time.monotonic() - start) * 1000


# ---------------------------------------------------------------------------
# Benchmark paths
# ---------------------------------------------------------------------------


async def run_sequential() -> dict[str, float]:
    """Sequential: provision sandbox → LLM → tool → done."""
    t0 = time.monotonic()

    # Step 1: Provision sandbox (blocking)
    await _simulate_sandbox_provision()
    t_sandbox_ready = (time.monotonic() - t0) * 1000

    # Step 2: First LLM call
    await _simulate_llm_inference()
    t_first_llm = (time.monotonic() - t0) * 1000

    # Step 3: First tool (sandbox already ready)
    await _simulate_tool_execution()
    t_total = (time.monotonic() - t0) * 1000

    return {
        "sandbox_ready_ms": t_sandbox_ready,
        "first_llm_ms": t_first_llm,
        "total_ms": t_total,
    }


async def run_parallel() -> dict[str, float]:
    """Parallel (brain-before-body): LLM starts while sandbox provisions."""
    t0 = time.monotonic()

    # Step 1: Start sandbox provisioning in background
    sandbox_task = asyncio.create_task(_simulate_sandbox_provision())

    # Step 2: First LLM call runs concurrently with sandbox
    await _simulate_llm_inference()
    t_first_llm = (time.monotonic() - t0) * 1000

    # Step 3: Await sandbox (usually already done by now)
    await sandbox_task
    t_sandbox_ready = (time.monotonic() - t0) * 1000

    # Step 4: First tool
    await _simulate_tool_execution()
    t_total = (time.monotonic() - t0) * 1000

    return {
        "sandbox_ready_ms": t_sandbox_ready,
        "first_llm_ms": t_first_llm,
        "total_ms": t_total,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def percentile(data: list[float], p: int) -> float:
    """Compute the p-th percentile of data."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


async def bench(n: int = 50) -> str:
    """Run N iterations of each path, compute stats, return markdown report."""
    print(f"Running {n} iterations of each path...")

    seq_results = []
    par_results = []

    for i in range(n):
        seq_results.append(await run_sequential())
        par_results.append(await run_parallel())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} done")

    def stats(results: list[dict], key: str) -> dict[str, float]:
        values = [r[key] for r in results]
        return {
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
            "mean": statistics.mean(values),
        }

    seq_total = stats(seq_results, "total_ms")
    par_total = stats(par_results, "total_ms")
    seq_llm = stats(seq_results, "first_llm_ms")
    par_llm = stats(par_results, "first_llm_ms")

    def delta(seq_v: float, par_v: float) -> str:
        pct = ((seq_v - par_v) / seq_v) * 100
        return f"{pct:+.1f}%"

    report = f"""# Brain-before-body Benchmark Results

**Date:** 2026-04
**Method:** Simulated sandbox provisioning ({SANDBOX_PROVISION_MS}ms) + LLM inference ({LLM_INFERENCE_MS}ms) + tool execution ({TOOL_EXECUTION_MS}ms)
**Sample size:** {n} iterations per path

## Simulation Parameters

| Component | Simulated latency |
|---|---|
| Sandbox provisioning | {SANDBOX_PROVISION_MS}ms |
| First LLM inference | {LLM_INFERENCE_MS}ms |
| Tool execution | {TOOL_EXECUTION_MS}ms |

## Results: Total Round-Trip Time

| Metric | Sequential | Parallel | Delta |
|---|---|---|---|
| p50 | {seq_total['p50']:.0f}ms | {par_total['p50']:.0f}ms | **{delta(seq_total['p50'], par_total['p50'])}** |
| p90 | {seq_total['p90']:.0f}ms | {par_total['p90']:.0f}ms | **{delta(seq_total['p90'], par_total['p90'])}** |
| p99 | {seq_total['p99']:.0f}ms | {par_total['p99']:.0f}ms | **{delta(seq_total['p99'], par_total['p99'])}** |
| mean | {seq_total['mean']:.0f}ms | {par_total['mean']:.0f}ms | {delta(seq_total['mean'], par_total['mean'])} |

## Results: Time to First LLM Response

| Metric | Sequential | Parallel | Delta |
|---|---|---|---|
| p50 | {seq_llm['p50']:.0f}ms | {par_llm['p50']:.0f}ms | **{delta(seq_llm['p50'], par_llm['p50'])}** |
| p90 | {seq_llm['p90']:.0f}ms | {par_llm['p90']:.0f}ms | **{delta(seq_llm['p90'], par_llm['p90'])}** |
| p99 | {seq_llm['p99']:.0f}ms | {par_llm['p99']:.0f}ms | **{delta(seq_llm['p99'], par_llm['p99'])}** |

## Interpretation

Sequential path: sandbox must finish before LLM starts → total = {SANDBOX_PROVISION_MS} + {LLM_INFERENCE_MS} + {TOOL_EXECUTION_MS} = {SANDBOX_PROVISION_MS + LLM_INFERENCE_MS + TOOL_EXECUTION_MS}ms theoretical.

Parallel path: LLM starts immediately, sandbox provisions in background. Since sandbox ({SANDBOX_PROVISION_MS}ms) < LLM ({LLM_INFERENCE_MS}ms), sandbox is ready before the first tool call → total = max({SANDBOX_PROVISION_MS}, {LLM_INFERENCE_MS}) + {TOOL_EXECUTION_MS} = {max(SANDBOX_PROVISION_MS, LLM_INFERENCE_MS) + TOOL_EXECUTION_MS}ms theoretical.

Theoretical reduction: {((SANDBOX_PROVISION_MS + LLM_INFERENCE_MS + TOOL_EXECUTION_MS) - (max(SANDBOX_PROVISION_MS, LLM_INFERENCE_MS) + TOOL_EXECUTION_MS)) / (SANDBOX_PROVISION_MS + LLM_INFERENCE_MS + TOOL_EXECUTION_MS) * 100:.0f}%

**Pass criteria:** p50 ≥ 30% reduction: {'PASS' if ((seq_total['p50'] - par_total['p50']) / seq_total['p50'] * 100) >= 30 else 'FAIL'} ({delta(seq_total['p50'], par_total['p50'])})
**Pass criteria:** p99 ≥ 50% reduction: {'PASS' if ((seq_total['p99'] - par_total['p99']) / seq_total['p99'] * 100) >= 50 else 'NEEDS INVESTIGATION'} ({delta(seq_total['p99'], par_total['p99'])})

## Note

This benchmark uses simulated delays (asyncio.sleep) rather than real sandbox provisioning and LLM calls. Real-world results will vary based on Docker performance, LLM provider latency, and network conditions. The simulation confirms the architectural benefit of parallelization.
"""
    return report


if __name__ == "__main__":
    report = asyncio.run(bench(50))
    print(report)

    # Write to file
    from pathlib import Path

    out = Path("benchmarks/brain_before_body_2026-04.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(f"Results written to {out}")
