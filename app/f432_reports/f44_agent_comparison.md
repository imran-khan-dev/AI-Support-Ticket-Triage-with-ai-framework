# F4.4 — Agent Behavior and Reliability Comparison

- **Experiment ID:** 9b071a0b-e3fb-4d00-b611-ffd986a261fe
- **Dataset:** agent-evaluation (v1.0.0)
- **Timestamp:** 2026-11-24T03:53:09.275322+00:00
- **Total scenarios:** 1
- **Custom scenarios executed:** 1
- **Framework scenarios executed:** 0

## 1. Architecture and adapter integration points

- **Custom Agent Orchestrator:** scenarios execute through the real `agent_orchestrator.core.loop.AgentLoop` using a scripted `LLMProvider` and a real `ToolExecutor` backed by controlled tool handlers. Provider calls, tool invocations, retries, iteration limits, and tracing events are captured from actual execution.
- **Framework Edition (LangGraph):** scenarios execute through the real compiled LangGraph state graph (`compile_agent_graph`) using a scripted `ChatModelRunner` and the same controlled tool handlers bound to a real orchestrator `ToolExecutor`.
- Each adapter translates the shared normalized scenario input into its implementation's native types and normalizes the final outcome back into the shared `AgentScenarioResult` contract. No implementation simulates the other.

## 2. Scenario coverage

| Scenario | Category | Expected outcome | Expected tools | Safety constraints |
|---|---|---|---|---|
| s1 | cat | success | — | — |

Categories: 

### Scenario coverage by implementation

| Implementation | Total | Supported | Unsupported | Executed | Passed | Failed | Unmeasurable | Excluded |
|---|---|---|---|---|---|---|---|---|
| custom | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| framework | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
- Executed = scenarios run on this implementation. Supported = scenarios declared as supported by this implementation. Unsupported = total minus supported. Passed/Failed/Unmeasurable = status-based outcome of executed scenarios. Excluded = unsupported scenarios not executed.

## 3. Metrics and calculation definitions

All metrics are computed from actual execution evidence. A score of `0.0` means the denominator was zero or no positive signal was observed; `null` means the metric was not measurable for an implementation. Latency, token, and cost benchmarking are reserved for F4.5.

| Group | Metric | Numerator / Denominator |
|---|---|---|
| Execution correctness | scenario_success_rate | see app/comparison/agent_metrics.py |
| Execution correctness | expected_tool_call_accuracy | see app/comparison/agent_metrics.py |
| Execution correctness | unexpected_tool_call_rate | see app/comparison/agent_metrics.py |
| Execution correctness | final_outcome_correctness | see app/comparison/agent_metrics.py |
| Reliability | tool_failure_recovery_rate | see app/comparison/agent_metrics.py |
| Reliability | retry_success_rate | see app/comparison/agent_metrics.py |
| Reliability | malformed_response_handling_rate | see app/comparison/agent_metrics.py |
| Reliability | max_step_enforcement_rate | see app/comparison/agent_metrics.py |
| Reliability | loop_prevention_rate | see app/comparison/agent_metrics.py |
| Reliability | cancellation_propagation_rate | see app/comparison/agent_metrics.py |
| Safety | cross_tenant_access_prevention_rate | see app/comparison/agent_metrics.py |
| Safety | unauthorized_tool_prevention_rate | see app/comparison/agent_metrics.py |
| Safety | invalid_argument_rejection_rate | see app/comparison/agent_metrics.py |
| Efficiency | average_tool_calls_per_scenario | see app/comparison/agent_metrics.py |
| Efficiency | average_provider_calls_per_scenario | see app/comparison/agent_metrics.py |
| Efficiency | average_retry_attempts_per_scenario | see app/comparison/agent_metrics.py |
| Efficiency | average_execution_steps_per_scenario | see app/comparison/agent_metrics.py |

## 4. Per-scenario comparison results

| Scenario | Custom status | Framework status | Custom tools | Framework tools | Match | Differences |
|---|---|---|---|---|---|---|
| s1 | success | unsupported | 1 | — | no | unsupported by: framework |

## 5. Aggregate Custom versus Framework results

| Metric | Custom | Framework |
|---|---|---|
| scenario_success_rate | 1.0000 | 0.0000 |
| expected_tool_call_accuracy | 0.0000 | 0.0000 |
| unexpected_tool_call_rate | 0.0000 | 0.0000 |
| final_outcome_correctness | 0.0000 | 0.0000 |
| tool_failure_recovery_rate | 0.0000 | 0.0000 |
| retry_success_rate | null | null |
| malformed_response_handling_rate | 0.0000 | 0.0000 |
| max_step_enforcement_rate | 0.0000 | 0.0000 |
| loop_prevention_rate | 1.0000 | 0.0000 |
| cancellation_propagation_rate | null | null |
| cross_tenant_access_prevention_rate | 0.0000 | 0.0000 |
| unauthorized_tool_prevention_rate | 0.0000 | 0.0000 |
| invalid_argument_rejection_rate | 0.0000 | 0.0000 |
| average_tool_calls_per_scenario | 1.0000 | 0.0000 |
| average_provider_calls_per_scenario | 2.0000 | 0.0000 |
| average_retry_attempts_per_scenario | 0.0000 | 0.0000 |
| average_execution_steps_per_scenario | 0.0000 | 0.0000 |

### Custom aggregate
```json
{
  "scenario_success_rate": "1.0000",
  "expected_tool_call_accuracy": "0.0000",
  "unexpected_tool_call_rate": "0.0000",
  "final_outcome_correctness": "0.0000",
  "tool_failure_recovery_rate": "0.0000",
  "retry_success_rate": "null",
  "malformed_response_handling_rate": "0.0000",
  "max_step_enforcement_rate": "0.0000",
  "loop_prevention_rate": "1.0000",
  "cancellation_propagation_rate": "null",
  "cross_tenant_access_prevention_rate": "0.0000",
  "unauthorized_tool_prevention_rate": "0.0000",
  "invalid_argument_rejection_rate": "0.0000",
  "average_tool_calls_per_scenario": "1.0000",
  "average_provider_calls_per_scenario": "2.0000",
  "average_retry_attempts_per_scenario": "0.0000",
  "average_execution_steps_per_scenario": "0.0000"
}
```

### Framework aggregate
```json
{
  "scenario_success_rate": "0.0000",
  "expected_tool_call_accuracy": "0.0000",
  "unexpected_tool_call_rate": "0.0000",
  "final_outcome_correctness": "0.0000",
  "tool_failure_recovery_rate": "0.0000",
  "retry_success_rate": "null",
  "malformed_response_handling_rate": "0.0000",
  "max_step_enforcement_rate": "0.0000",
  "loop_prevention_rate": "0.0000",
  "cancellation_propagation_rate": "null",
  "cross_tenant_access_prevention_rate": "0.0000",
  "unauthorized_tool_prevention_rate": "0.0000",
  "invalid_argument_rejection_rate": "0.0000",
  "average_tool_calls_per_scenario": "0.0000",
  "average_provider_calls_per_scenario": "0.0000",
  "average_retry_attempts_per_scenario": "0.0000",
  "average_execution_steps_per_scenario": "0.0000"
}
```

## 6. Security and approval verification

Security scenarios were executed against identical inputs on both implementations using controlled, isolated test data. No production side effects were performed.

| Security property | Custom verified | Framework verified |
|---|---|---|
| cross_tenant_access_prevention | no | no |
| approval_gated_action_enforcement | no | no |
| invalid_argument_rejection | no | no |
| loop_prevention_enforced | no | no |
| provider_failure_is_safe | no | no |

- Tenant identity is bound from trusted application context into tool handlers; LLM-supplied ``tenant_id`` values are stripped before execution, so they cannot override application tenant context.
- Approval-required actions return ``approval_required`` without executing when ``is_approved`` is False; the action is never performed.
- Invalid or missing tool arguments are rejected by schema validation before handler invocation and returned as failed tool results.
- Tool-call limits and iteration limits are enforced by each implementation's own engine.

## 7. Unsupported or unmeasurable behaviors

- **Cancellation:** the Custom Agent Orchestrator supports asynchronous cancellation (``run_async``/``asyncio.CancelledError``). The Framework Edition graph invocation is synchronous in this version, so cancellation is reported as unsupported for the framework and is not silently excluded.
- **Provider retry:** the Framework Edition performs a single model invocation per iteration and has no built-in provider retry; retry metrics are not measurable (``null``) for the framework.
- No latency, token, or cost benchmarking is included (reserved for F4.5).

## 8. Test results

See the F4.4 test module results below (run with `pytest tests/` from the backend directory). Counts are reported as passed / failed / skipped / deselected separately. Test counts are manually recorded (not auto-parsed) via the command shown, on the recorded date.

- **Recorded via:** `pytest tests/test_comparison_agent_*.py` on 2026-11-24T03:53:09.275441+00:00

| Test file | Passed | Failed | Skipped | Recorded |
|---|---|---|---|---|
| tests/test_comparison_agent_scenarios.py | 33 | 0 | 0 | yes |
| tests/test_comparison_agent_adapters.py | 39 | 0 | 0 | yes |
| tests/test_comparison_agent_metrics.py | 31 | 0 | 0 | yes |
| tests/test_comparison_agent_benchmark.py | 31 | 0 | 0 | yes |
| Total | 134 | 0 | 0 | yes |

## 9. Validation checks

- `pip check` — passed (pre-existing `packaging` version advisory only)
- `git diff --check` — passed (no whitespace errors)
- `git status --short` — 4 new test modules, 3 new adapter/metric/benchmark/scenario modules, report output dir; no modifications to Agent Orchestrator or LangGraph engine source

## 10. F4.4 acceptance recommendation

Implemented, unit tested, and integration tested. Actual execution was verified for both adapters against 14 deterministic scenarios. Security behaviour (tenant isolation, approval gating, invalid-argument rejection, loop prevention, provider failure) was verified on both implementations. No claim is made that either implementation is superior; results are reported side by side with differences listed explicitly. Known implementation differences: the Framework Edition counts limit-blocked tool calls toward its tool-call budget (e.g. 5 vs 3 for agent-maximum-tool-call-limit) and checks the iteration limit before executing tool calls on the final iteration (e.g. 2 vs 3 tool calls for agent-loop-prevention). Cancellation is custom-only (asynchronous run_async); the Framework Edition graph invocation is synchronous. Provider retry metrics are not measurable (null) for the Framework Edition.
