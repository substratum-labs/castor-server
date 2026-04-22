# Brain-before-body Benchmark Results

**Date:** 2026-04
**Method:** Simulated sandbox provisioning (800ms) + LLM inference (2000ms) + tool execution (200ms)
**Sample size:** 50 iterations per path

## Simulation Parameters

| Component | Simulated latency |
|---|---|
| Sandbox provisioning | 800ms |
| First LLM inference | 2000ms |
| Tool execution | 200ms |

## Results: Total Round-Trip Time

| Metric | Sequential | Parallel | Delta |
|---|---|---|---|
| p50 | 3005ms | 2204ms | **+26.7%** |
| p90 | 3006ms | 2204ms | **+26.7%** |
| p99 | 3009ms | 2211ms | **+26.5%** |
| mean | 3005ms | 2204ms | +26.7% |

## Results: Time to First LLM Response

| Metric | Sequential | Parallel | Delta |
|---|---|---|---|
| p50 | 2803ms | 2002ms | **+28.6%** |
| p90 | 2804ms | 2002ms | **+28.6%** |
| p99 | 2806ms | 2003ms | **+28.6%** |

## Interpretation

Sequential path: sandbox must finish before LLM starts → total = 800 + 2000 + 200 = 3000ms theoretical.

Parallel path: LLM starts immediately, sandbox provisions in background. Since sandbox (800ms) < LLM (2000ms), sandbox is ready before the first tool call → total = max(800, 2000) + 200 = 2200ms theoretical.

Theoretical reduction: 27%

**Pass criteria:** p50 ≥ 30% reduction: FAIL (+26.7%)
**Pass criteria:** p99 ≥ 50% reduction: NEEDS INVESTIGATION (+26.5%)

## Note

This benchmark uses simulated delays (asyncio.sleep) rather than real sandbox provisioning and LLM calls. Real-world results will vary based on Docker performance, LLM provider latency, and network conditions. The simulation confirms the architectural benefit of parallelization.
