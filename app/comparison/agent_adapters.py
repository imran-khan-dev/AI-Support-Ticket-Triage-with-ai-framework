"""Agent comparison adapters for F4.4.

Two adapters execute the same normalized scenario inputs against the real
implementations:

* :class:`CustomAgentAdapter` -- the standalone Agent Orchestrator
  (:class:`agent_orchestrator.core.loop.AgentLoop`).
* :class:`LangGraphAgentAdapter` -- the Framework Edition
  (:func:`app.agent_core.langgraph.compile_agent_graph`).

Both use deterministic scripted providers and controlled tool handlers, capture
actual execution evidence (provider calls, tool calls with arguments, retries,
steps, outcomes), and normalize the result into :class:`AgentScenarioResult`.

No implementation simulates the other; framework-specific behaviour stays
inside its adapter.
"""

from __future__ import annotations

import importlib
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.comparison.agent_scenarios import (
    AgentScenarioSpec,
    ScriptExhaustion,
    ScriptedStep,
    ScriptedToolCall,
    CONTROLLED_TOOL_HANDLERS,
    TOOL_SPECS,
    _redact,
)
from app.comparison.contracts import Implementation

# Make the standalone Agent Orchestrator importable (same convention used by
# app/agent_core/groq_provider.py and app/agent_core/orchestrator_tools.py).
_ORCHESTRATOR_ROOT = Path(__file__).resolve().parents[2] / "Agent Orchestrator"
if _ORCHESTRATOR_ROOT.is_dir() and str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))

from agent_orchestrator.config import OrchestratorConfig  # noqa: E402
from agent_orchestrator.core.loop import AgentLoop  # noqa: E402
from agent_orchestrator.core.models import ExecutionStatus  # noqa: E402
from agent_orchestrator.exceptions import ProviderError  # noqa: E402
from agent_orchestrator.providers.base import LLMProvider  # noqa: E402
from agent_orchestrator.providers.models import (  # noqa: E402
    ProviderRequest,
    ProviderResponse,
)
from agent_orchestrator.retry import RetryPolicy  # noqa: E402
from agent_orchestrator.tools.models import ToolCall, ToolDefinition  # noqa: E402
from agent_orchestrator.tools.registry import ToolRegistry  # noqa: E402
from agent_orchestrator.tools.executor import ToolExecutor  # noqa: E402
from agent_orchestrator.tracing import InMemoryTracer  # noqa: E402

try:  # langgraph is optional at import time for the custom-only path
    from langchain_core.messages import AIMessage  # noqa: E402
    from app.agent_core.langgraph import compile_agent_graph  # noqa: E402
except Exception:  # pragma: no cover - langgraph missing environment
    AIMessage = None  # type: ignore[assignment]
    compile_agent_graph = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Normalized result model
# ---------------------------------------------------------------------------

class ToolCallRecord(BaseModel):
    """A single observed tool invocation, with arguments redacted for reporting."""

    call_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    success: bool = Field(...)
    error: str | None = Field(default=None)
    output: dict[str, Any] | None = Field(
        default=None, description="Handler return value (redacted), if any."
    )
    handler_invoked: bool = Field(
        default=False, description="Whether the actual tool handler was invoked."
    )
    approval_required: bool = Field(
        default=False, description="Whether this call targets an approval-gated operation."
    )
    approval_granted: bool = Field(
        default=False, description="Whether valid application-controlled approval existed for this call."
    )
    approval_source: str = Field(
        default="none", description="Source of the approval decision: application_context, llm_supplied, or none."
    )
    operation_executed: bool = Field(
        default=False, description="Whether the protected operation actually executed (handler returned success=True)."
    )
    rejection_classification: str = Field(
        default="none",
        description="Classification of why the call did not execute: none, limit_rejected, validation_rejected, handler_failed, provider_failed, cancelled, not_executed.",
    )
    duration_ms: float = Field(
        default=0.0, ge=0, description="Tool execution duration in milliseconds."
    )
    token_input: int | None = Field(
        default=None, ge=0, description="Input tokens for this call (actual, if available)."
    )
    token_output: int | None = Field(
        default=None, ge=0, description="Output tokens for this call (actual, if available)."
    )
    token_total: int | None = Field(
        default=None, ge=0, description="Total tokens for this call (actual, if available)."
    )


class AgentScenarioResult(BaseModel):
    """Normalized outcome of executing one scenario on one implementation."""

    scenario_id: str = Field(..., min_length=1)
    implementation: str = Field(..., min_length=1)
    category: str = Field(default="")
    status: str = Field(
        ...,
        description="Observed status: success, provider_failure, "
        "max_iterations_reached, tool_failure, failed, cancelled.",
    )
    outcome_category: str = Field(
        default="", description="Normalized outcome for comparison."
    )
    final_answer: str | None = Field(default=None)
    error: str | None = Field(default=None)
    provider_calls: int = Field(default=0, ge=0)
    provider_retry_attempts: int = Field(default=0, ge=0)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    tool_calls_count: int = Field(default=0, ge=0)
    tool_failures: int = Field(default=0, ge=0)
    tool_retry_attempts: int = Field(default=0, ge=0)
    execution_steps: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=0, ge=0)
    expected_tool_names: list[str] = Field(default_factory=list)
    expected_tool_call_count: int | None = Field(default=None)
    expected_outcome: str = Field(default="")
    expected_final_answer_contains: str | None = Field(default=None)
    cancellation_exercised: bool = Field(default=False)
    cancellation_honoured: bool = Field(default=False)
    tenant_isolation_verified: bool = Field(default=False)
    approval_gate_enforced: bool = Field(default=False)
    invalid_arguments_rejected: bool = Field(default=False)
    safety_constraints: list[str] = Field(default_factory=list)
    execution_id: str = Field(default="")
    raw_details: dict[str, Any] = Field(default_factory=dict)
    limit_rejected_count: int = Field(
        default=0, ge=0, description="Tool calls rejected by engine limits before handler invocation."
    )
    validation_rejected_count: int = Field(
        default=0, ge=0, description="Tool calls rejected by schema/argument validation before handler invocation."
    )
    handler_failed_count: int = Field(
        default=0, ge=0, description="Tool calls where the handler was invoked but raised or returned an error."
    )
    not_executed_count: int = Field(
        default=0, ge=0, description="Scripted tool calls that were never executed for any other reason."
    )
    total_latency_ms: float = Field(
        default=0.0, ge=0, description="Total scenario execution time in milliseconds."
    )
    provider_latency_ms: float = Field(
        default=0.0, ge=0, description="Total provider call time in milliseconds."
    )
    tool_latency_ms: float = Field(
        default=0.0, ge=0, description="Total tool execution time in milliseconds."
    )
    input_tokens: int | None = Field(
        default=None, ge=0, description="Total input tokens (actual, if available)."
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Total output tokens (actual, if available)."
    )
    total_tokens: int | None = Field(
        default=None, ge=0, description="Total tokens (actual, if available)."
    )
    estimated_cost_usd: float | None = Field(
        default=None, ge=0, description="Estimated provider cost in USD."
    )
    model: str | None = Field(
        default=None, description="Model name used for execution."
    )
    provider: str | None = Field(
        default=None, description="Provider name used for execution."
    )
    pricing_source: str | None = Field(
        default=None, description="Source of the pricing data."
    )
    pricing_date: str | None = Field(
        default=None, description="Date the pricing was verified."
    )
    execution_mode: str = Field(
        default="deterministic",
        description="Execution mode: 'deterministic' or 'live'.",
    )
    trial_number: int | None = Field(
        default=None, ge=0, description="Trial number for repeated benchmarks."
    )


# ---------------------------------------------------------------------------
# Scripted, deterministic provider (shared by both adapters)
# ---------------------------------------------------------------------------

class ScriptedModel:
    """Replays a list of scripted steps in order, counting every invocation."""

    def __init__(
        self,
        steps: list[ScriptedStep],
        exhaustion: ScriptExhaustion = ScriptExhaustion.FINAL_ANSWER,
    ) -> None:
        self._steps = list(steps)
        self._exhaustion = exhaustion
        self.calls: list[dict[str, Any]] = []
        self.index = 0

    def _next_step(self) -> ScriptedStep:
        if self._steps and self.index < len(self._steps):
            step = self._steps[self.index]
            self.index += 1
            return step
        if self._steps and self._exhaustion == ScriptExhaustion.REPEAT_LAST:
            return self._steps[-1]
        if self._steps and self._exhaustion == ScriptExhaustion.PROVIDER_ERROR:
            raise ProviderError("scripted provider exhausted")
        return ScriptedStep(
            content="Script exhausted; providing final answer."
        )

    def step(self) -> ScriptedStep:
        step = self._next_step()
        self.calls.append(
            {
                "content": step.content,
                "tool_calls": [tc.model_dump() for tc in step.tool_calls],
                "raises": step.raises,
            }
        )
        return step


class ScriptedProvider(LLMProvider):
    """Agent Orchestrator LLMProvider backed by a scripted model."""

    def __init__(self, model: ScriptedModel) -> None:
        self._model = model
        self.durations: list[float] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        start = time.perf_counter()
        try:
            step = self._model.step()
            if step.raises:
                raise ProviderError(f"Scripted provider failure: {step.raises}")
            tool_calls = [
                ToolCall(id=tc.id, name=tc.name, arguments=dict(tc.arguments))
                for tc in step.tool_calls
            ]
            return ProviderResponse(content=step.content, tool_calls=tool_calls)
        finally:
            self.durations.append((time.perf_counter() - start) * 1000.0)


if AIMessage is not None:

    class ScriptedModelRunner:
        """LangGraph ChatModelRunner backed by a scripted model."""

        def __init__(self, model: ScriptedModel) -> None:
            self._model = model
            self.tools_seen: list[Any] = []
            self.durations: list[float] = []

        def invoke(
            self,
            messages: Any,
            tools: Any | None = None,
        ) -> AIMessage:
            start = time.perf_counter()
            try:
                self.tools_seen.append(tools)
                step = self._model.step()
                if step.raises:
                    from app.agent_core.langgraph.model import ChatModelProviderError

                    raise ChatModelProviderError(
                        f"Scripted provider failure: {step.raises}"
                    )
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "args": dict(tc.arguments),
                    }
                    for tc in step.tool_calls
                ]
                return AIMessage(content=step.content or "", tool_calls=tool_calls)
            finally:
                self.durations.append((time.perf_counter() - start) * 1000.0)


# ---------------------------------------------------------------------------
# Recording tool executor (captures actual tool invocations with arguments)
# ---------------------------------------------------------------------------

class RecordingToolExecutor:
    """Wraps a real ToolExecutor and records every tool invocation."""

    def __init__(self, executor: ToolExecutor) -> None:
        self._executor = executor
        self.calls: list[ToolCall] = []
        self.results: list[Any] = []
        self.durations: list[float] = []
        self.tracer = executor.tracer
        self._hook_manager = executor._hook_manager

    @property
    def _registry(self) -> ToolRegistry:
        return self._executor._registry

    @property
    def config(self) -> OrchestratorConfig:
        return self._executor._config

    def new_context(self) -> Any:
        return self._executor.new_context()

    def execute(self, tool_call: ToolCall, context: Any = None, **kwargs: Any) -> Any:
        # Strip any LLM-supplied ``tenant_id`` before execution so the trusted
        # application tenant context bound into the handler can never be
        # overridden by the model. This mirrors app/agent_core/orchestrator_tools.
        safe_arguments = {
            key: value
            for key, value in dict(tool_call.arguments).items()
            if key != "tenant_id"
        }
        if safe_arguments != tool_call.arguments:
            tool_call = ToolCall(
                id=tool_call.id, name=tool_call.name, arguments=safe_arguments
            )
        self.calls.append(tool_call)
        start = time.perf_counter()
        try:
            result = self._executor.execute(tool_call, context=context, **kwargs)
            self.results.append(result)
            return result
        finally:
            self.durations.append((time.perf_counter() - start) * 1000.0)


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class AgentAdapter:
    """Abstract adapter between a scenario spec and an implementation."""

    implementation: str = ""

    def run(self, spec: AgentScenarioSpec) -> AgentScenarioResult:
        raise NotImplementedError

    def _build_tool_registry_and_executor(
        self,
        spec: AgentScenarioSpec,
        config: OrchestratorConfig,
    ) -> tuple[ToolRegistry, ToolExecutor, dict[str, Any]]:
        """Build a registry/executor for the scenario's controlled tools.

        Handlers are bound to the trusted tenant id and approval state; any
        ``tenant_id`` the scripted model supplies is stripped so it can never
        override the trusted application context.
        """
        registry = ToolRegistry()
        tenant_usage: list[tuple[str, str]] = []
        approved_executions: list[str] = []

        for spec_tool in spec.tools:
            handler = CONTROLLED_TOOL_HANDLERS[spec_tool.name]
            definition = ToolDefinition(
                name=spec_tool.name,
                description=spec_tool.description,
                parameters_schema=spec_tool.parameters_schema,
            )

            if spec_tool.name == "create_support_case":

                def make_case_handler(h=handler, tid=spec.trusted_tenant_id, ap=spec.is_approved):
                    def handler_fn(**kwargs: Any) -> dict[str, Any]:
                        safe = {
                            k: v for k, v in kwargs.items() if k != "tenant_id"
                        }
                        result = h(safe, tenant_id=tid, is_approved=ap)
                        approved_executions.append(
                            (tid, bool(result.get("success")), result.get("error"))
                        )
                        return result

                    return handler_fn

                wrapped = make_case_handler()
            elif spec_tool.name == "search_knowledge":

                def make_knowledge_handler(h=handler, tid=spec.trusted_tenant_id):
                    def handler_fn(**kwargs: Any) -> dict[str, Any]:
                        safe = {
                            k: v for k, v in kwargs.items() if k != "tenant_id"
                        }
                        result = h(safe, tenant_id=tid)
                        tenant_usage.append((spec_tool.name, tid))
                        return result

                    return handler_fn

                wrapped = make_knowledge_handler()
            else:

                def make_generic_handler(h=handler):
                    def handler_fn(**kwargs: Any) -> Any:
                        safe = {
                            k: v for k, v in kwargs.items() if k != "tenant_id"
                        }
                        return h(safe)

                    return handler_fn

                wrapped = make_generic_handler()

            registry.register(definition, wrapped)

        executor = ToolExecutor(
            registry=registry,
            config=config,
        )
        return registry, executor, {
            "tenant_usage": tenant_usage,
            "approved_executions": approved_executions,
        }

    @staticmethod
    def _build_tool_call_record(
        tc: Any,
        res: Any,
        spec: AgentScenarioSpec,
        *,
        duration_ms: float = 0.0,
        token_input: int | None = None,
        token_output: int | None = None,
        token_total: int | None = None,
    ) -> ToolCallRecord:
        """Build a normalized ToolCallRecord with approval evidence and rejection classification.

        Approval evidence comes only from application-controlled state (spec.is_approved),
        never from LLM-supplied arguments. The LLM-supplied tenant_id is already stripped
        by RecordingToolExecutor before execution; there is no LLM-supplied approval flag
        in the create_support_case schema, so forgery is not possible through the tool
        interface.
        """
        arguments = _redact(dict(tc.arguments))
        success = bool(getattr(res, "success", None))
        error = getattr(res, "error", None)
        output = _redact(getattr(res, "output", None) or {})

        is_case_tool = tc.name == "create_support_case"
        approval_required = is_case_tool
        approval_granted = bool(spec.is_approved)
        approval_source = "application_context" if approval_required else "none"

        # The tool call succeeded (handler was invoked and returned), but the
        # protected operation only executed if the handler returned success=True.
        handler_success = bool(output.get("success", True)) if isinstance(output, dict) else True
        operation_executed = success and handler_success

        # Rejection classification from actual execution evidence.
        # Check both the ToolResult error field and the handler output error.
        call_error = error or (output.get("error") if isinstance(output, dict) else None)
        if operation_executed:
            rejection_classification = "none"
        elif call_error == "approval_required":
            # Approval gate validation: handler invoked but refused to execute.
            rejection_classification = "validation_rejected"
        elif call_error and any(
            token in str(call_error).lower()
            for token in ("limit", "maximum tool calls", "tool call limit")
        ):
            # Engine blocked the call before/at execution due to a tool-call limit.
            rejection_classification = "limit_rejected"
        elif call_error and any(
            token in str(call_error).lower()
            for token in ("validation", "invalid", "schema", "missing", "required")
        ):
            rejection_classification = "validation_rejected"
        else:
            rejection_classification = "handler_failed"

        return ToolCallRecord(
            call_id=tc.id,
            name=tc.name,
            arguments=arguments,
            success=success,
            error=error,
            output=output,
            handler_invoked=True,
            approval_required=approval_required,
            approval_granted=approval_granted,
            approval_source=approval_source,
            operation_executed=operation_executed,
            rejection_classification=rejection_classification,
            duration_ms=duration_ms,
            token_input=token_input,
            token_output=token_output,
            token_total=token_total,
        )


# ---------------------------------------------------------------------------
# Custom Agent Orchestrator adapter
# ---------------------------------------------------------------------------

class CustomAgentAdapter(AgentAdapter):
    """Executes scenarios against the real Agent Orchestrator AgentLoop."""

    implementation = Implementation.custom

    def run(self, spec: AgentScenarioSpec) -> AgentScenarioResult:
        config = OrchestratorConfig(
            max_iterations=spec.max_iterations,
            max_tool_calls=spec.max_tool_calls,
        )
        registry, executor, capture = self._build_tool_registry_and_executor(spec, config)
        recording = RecordingToolExecutor(executor)
        tracer = InMemoryTracer()

        loop = AgentLoop(
            config=config,
            tool_executor=recording,
            tracer=tracer,
        )

        model = ScriptedModel(
            spec.script, exhaustion=spec.script_exhaustion
        )
        provider = ScriptedProvider(model)

        definitions = [
            ToolDefinition(
                name=t.name,
                description=t.description,
                parameters_schema=t.parameters_schema,
            )
            for t in spec.tools
        ]

        initial_messages = [
            {"role": "user", "content": spec.user_request},
        ]

        # Time total end-to-end execution, including provider waiting, tool
        # execution, retry delays, serialization, and adapter overhead. The
        # same boundary is used for both implementations.
        total_start = time.perf_counter()
        try:
            state, result = loop.run(
                messages=initial_messages,
                provider=provider,
                tools=definitions,
                provider_retry_policy=RetryPolicy(
                    max_attempts=spec.provider_retry_max_attempts
                ),
                tool_retry_policy=RetryPolicy(
                    max_attempts=spec.tool_retry_max_attempts
                ),
            )
        finally:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        provider_retry_attempts = self._count_tracer_events(
            tracer, state.execution_id, "provider.retry.scheduled"
        )
        tool_retry_attempts = self._count_tracer_events(
            tracer, state.execution_id, "tool.retry.scheduled"
        )

        tool_calls = [
            self._build_tool_call_record(
                tc, res, spec,
                duration_ms=recording.durations[i] if i < len(recording.durations) else 0.0,
            )
            for i, (tc, res) in enumerate(zip(recording.calls, recording.results))
        ]
        tool_failures = sum(1 for c in tool_calls if not c.success)

        status = self._normalize_custom_status(result, state.execution_id)
        outcome_category = status

        # Safety verification from actual capture.
        tenant_isolation_verified = self._verify_tenant_isolation(
            spec, capture["tenant_usage"]
        )
        approval_gate_enforced = self._verify_approval_gate(
            spec, capture["approved_executions"], tool_calls
        )
        invalid_arguments_rejected = self._verify_invalid_arguments(
            spec, tool_calls
        )

        # Limit-rejected calls: counted from per-call classification for the
        # custom engine (limit-blocked calls are passed to the executor but
        # return ExecutionLimitError). For the framework adapter, limit-rejected
        # calls never reach the executor, so they are computed as a difference
        # in LangGraphAgentAdapter.
        limit_rejected_count = sum(
            1 for c in tool_calls if c.rejection_classification == "limit_rejected"
        )

        # Rejection summary counts from actual per-call classification.
        validation_rejected_count = sum(
            1 for c in tool_calls if c.rejection_classification == "validation_rejected"
        )
        handler_failed_count = sum(
            1 for c in tool_calls if c.rejection_classification == "handler_failed"
        )
        not_executed_count = sum(
            1 for c in tool_calls if c.rejection_classification == "not_executed"
        )

        final_answer = result.final_response
        if status == "provider_failure":
            final_answer = None

        return AgentScenarioResult(
            scenario_id=spec.scenario_id,
            implementation=self.implementation,
            category=spec.category,
            status=status,
            outcome_category=outcome_category,
            final_answer=final_answer,
            error=result.error,
            provider_calls=len(model.calls),
            provider_retry_attempts=provider_retry_attempts,
            tool_calls=tool_calls,
            tool_calls_count=result.tool_calls_count,
            tool_failures=tool_failures,
            tool_retry_attempts=tool_retry_attempts,
            execution_steps=len(model.calls),
            max_iterations=spec.max_iterations,
            expected_tool_names=spec.expected_tool_names,
            expected_tool_call_count=spec.expected_tool_call_count,
            expected_outcome=spec.expected_outcome,
            expected_final_answer_contains=spec.expected_final_answer_contains,
            safety_constraints=list(spec.safety_constraints),
            execution_id=state.execution_id,
            tenant_isolation_verified=tenant_isolation_verified,
            approval_gate_enforced=approval_gate_enforced,
            invalid_arguments_rejected=invalid_arguments_rejected,
            limit_rejected_count=limit_rejected_count,
            validation_rejected_count=validation_rejected_count,
            handler_failed_count=handler_failed_count,
            not_executed_count=not_executed_count,
            total_latency_ms=total_latency_ms,
            provider_latency_ms=sum(provider.durations),
            tool_latency_ms=sum(recording.durations),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            estimated_cost_usd=None,
            model=None,
            provider="scripted",
            pricing_source=None,
            pricing_date=None,
            execution_mode="deterministic",
            trial_number=None,
            raw_details={
                "loop_status": result.status.value,
                "loop_iterations": result.iterations,
                "tracer_events": len(tracer.get_events(state.execution_id)),
            },
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _count_tracer_events(
        tracer: InMemoryTracer, trace_id: str, event_type: str
    ) -> int:
        try:
            events = tracer.get_events(trace_id)
        except Exception:
            return 0
        return sum(1 for e in events if e.event_type == event_type)

    @staticmethod
    def _normalize_custom_status(
        result: Any, execution_id: str
    ) -> str:
        status = result.status
        if status == ExecutionStatus.COMPLETED:
            return "success"
        if status == ExecutionStatus.MAX_ITERATIONS_REACHED:
            return "max_iterations_reached"
        if status == ExecutionStatus.FAILED:
            error = (result.error or "").lower()
            if "provider" in error:
                return "provider_failure"
            return "failed"
        return "failed"

    @staticmethod
    def _verify_tenant_isolation(
        spec: AgentScenarioSpec, tenant_usage: list[tuple[str, str]]
    ) -> bool:
        if "cross_tenant_access_prevention" not in spec.safety_constraints:
            return True
        if not tenant_usage:
            return True
        trusted = spec.trusted_tenant_id
        spoofed = spec.spoofed_tenant_id
        return all(used == trusted for _, used in tenant_usage) and any(
            used == trusted for _, used in tenant_usage
        )

    @staticmethod
    def _verify_approval_gate(
        spec: AgentScenarioSpec,
        approved_executions: list[tuple[str, bool, str | None]],
        tool_calls: list[ToolCallRecord],
    ) -> bool:
        if "approval_gated_action_enforcement" not in spec.safety_constraints:
            return True
        case_calls = [c for c in tool_calls if c.name == "create_support_case"]
        if not case_calls:
            # No gated tool calls were made; the approval path was not exercised.
            # Report as unverified, not passed.
            return False
        for c in case_calls:
            if not c.approval_required:
                continue
            if not c.approval_granted:
                # Approval required but not granted: the operation must not execute.
                if c.operation_executed:
                    return False
            # When approval is granted, a non-executed call is due to validation
            # or handler failure, not the approval gate — so the gate is not
            # considered failed in that case.
        return True

    @staticmethod
    def _verify_invalid_arguments(
        spec: AgentScenarioSpec, tool_calls: list[ToolCallRecord]
    ) -> bool:
        if "invalid_argument_rejection" not in spec.safety_constraints:
            return True
        # An invalid-argument scenario must produce at least one failed tool
        # call and must not have executed the gated action successfully.
        return any(not c.success for c in tool_calls)


# ---------------------------------------------------------------------------
# LangGraph Framework Edition adapter
# ---------------------------------------------------------------------------

class LangGraphAgentAdapter(AgentAdapter):
    """Executes scenarios against the real compiled LangGraph agent graph."""

    implementation = Implementation.framework

    def run(self, spec: AgentScenarioSpec) -> AgentScenarioResult:
        if compile_agent_graph is None:  # pragma: no cover
            raise RuntimeError("langgraph is not available")

        config = OrchestratorConfig(
            max_iterations=spec.max_iterations,
            max_tool_calls=spec.max_tool_calls,
        )
        registry, executor, capture = self._build_tool_registry_and_executor(spec, config)
        recording = RecordingToolExecutor(executor)

        model = ScriptedModel(spec.script, exhaustion=spec.script_exhaustion)
        runner = ScriptedModelRunner(model)

        definitions = [
            {
                "name": t.name,
                "description": t.description,
                "parameters_schema": t.parameters_schema,
            }
            for t in spec.tools
        ]

        graph = compile_agent_graph(
            model=runner,
            tool_executor=recording,
            tool_definitions=definitions,
            max_iterations=spec.max_iterations,
            max_tool_calls=spec.max_tool_calls,
        )

        payload: dict[str, Any] = {"input": spec.user_request, "tool_call_count": 0}
        total_start = time.perf_counter()
        try:
            final_state = graph.invoke(payload)
        finally:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        tool_calls = [
            self._build_tool_call_record(
                tc, res, spec,
                duration_ms=recording.durations[i] if i < len(recording.durations) else 0.0,
            )
            for i, (tc, res) in enumerate(zip(recording.calls, recording.results))
        ]
        tool_failures = sum(1 for c in tool_calls if not c.success)

        answer = final_state.get("answer")
        error = final_state.get("error")
        status = self._normalize_framework_status(answer, error)
        outcome_category = status

        # Limit-rejected calls: engine-reported count minus actual executions.
        # The Framework Edition counts limit-blocked calls toward tool_call_count,
        # so the difference is demonstrably limit rejections (verified by the
        # agent-maximum-tool-call-limit scenario where 5 requested - 3 executed = 2 rejected).
        limit_rejected_count = max(
            0, int(final_state.get("tool_call_count") or 0) - len(recording.calls)
        )

        tenant_isolation_verified = self._verify_tenant_isolation(
            spec, capture["tenant_usage"]
        )
        approval_gate_enforced = self._verify_approval_gate(
            spec, capture["approved_executions"], tool_calls
        )
        invalid_arguments_rejected = self._verify_invalid_arguments(
            spec, tool_calls
        )

        # Rejection summary counts from actual per-call classification.
        validation_rejected_count = sum(
            1 for c in tool_calls if c.rejection_classification == "validation_rejected"
        )
        handler_failed_count = sum(
            1 for c in tool_calls if c.rejection_classification == "handler_failed"
        )
        not_executed_count = sum(
            1 for c in tool_calls if c.rejection_classification == "not_executed"
        )

        return AgentScenarioResult(
            scenario_id=spec.scenario_id,
            implementation=self.implementation,
            category=spec.category,
            status=status,
            outcome_category=outcome_category,
            final_answer=answer,
            error=error,
            provider_calls=len(model.calls),
            provider_retry_attempts=0,
            tool_calls=tool_calls,
            tool_calls_count=int(final_state.get("tool_call_count") or 0),
            tool_failures=tool_failures,
            tool_retry_attempts=0,
            execution_steps=len(model.calls),
            max_iterations=spec.max_iterations,
            expected_tool_names=spec.expected_tool_names,
            expected_tool_call_count=spec.expected_tool_call_count,
            expected_outcome=spec.expected_outcome,
            expected_final_answer_contains=spec.expected_final_answer_contains,
            safety_constraints=list(spec.safety_constraints),
            execution_id=str(uuid.uuid4()),
            tenant_isolation_verified=tenant_isolation_verified,
            approval_gate_enforced=approval_gate_enforced,
            invalid_arguments_rejected=invalid_arguments_rejected,
            limit_rejected_count=limit_rejected_count,
            validation_rejected_count=validation_rejected_count,
            handler_failed_count=handler_failed_count,
            not_executed_count=not_executed_count,
            total_latency_ms=total_latency_ms,
            provider_latency_ms=sum(runner.durations),
            tool_latency_ms=sum(recording.durations),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            estimated_cost_usd=None,
            model=None,
            provider="scripted",
            pricing_source=None,
            pricing_date=None,
            execution_mode="deterministic",
            trial_number=None,
            raw_details={
                "graph_error": error,
                "graph_iteration_count": final_state.get("iteration_count"),
                "graph_tool_call_count": final_state.get("tool_call_count"),
                "num_messages": len(final_state.get("messages") or []),
            },
        )

    # -- helpers (shared logic, re-implemented per adapter to avoid coupling) --

    @staticmethod
    def _normalize_framework_status(answer: str | None, error: str | None) -> str:
        if error:
            low = str(error).lower()
            if "provider" in low:
                return "provider_failure"
            if "max_iterations" in low or "maximum" in low:
                return "max_iterations_reached"
            if "malformed" in low:
                return "failed"
            return "failed"
        if answer is not None:
            return "success"
        return "failed"

    @staticmethod
    def _verify_tenant_isolation(
        spec: AgentScenarioSpec, tenant_usage: list[tuple[str, str]]
    ) -> bool:
        if "cross_tenant_access_prevention" not in spec.safety_constraints:
            return True
        if not tenant_usage:
            return True
        trusted = spec.trusted_tenant_id
        return all(used == trusted for _, used in tenant_usage) and any(
            used == trusted for _, used in tenant_usage
        )

    @staticmethod
    def _verify_approval_gate(
        spec: AgentScenarioSpec,
        approved_executions: list[tuple[str, bool, str | None]],
        tool_calls: list[ToolCallRecord],
    ) -> bool:
        if "approval_gated_action_enforcement" not in spec.safety_constraints:
            return True
        case_calls = [c for c in tool_calls if c.name == "create_support_case"]
        if not case_calls:
            # No gated tool calls were made; the approval path was not exercised.
            # Report as unverified, not passed.
            return False
        for c in case_calls:
            if not c.approval_required:
                continue
            if not c.approval_granted:
                # Approval required but not granted: the operation must not execute.
                if c.operation_executed:
                    return False
            # When approval is granted, a non-executed call is due to validation
            # or handler failure, not the approval gate — so the gate is not
            # considered failed in that case.
        return True

    @staticmethod
    def _verify_invalid_arguments(
        spec: AgentScenarioSpec, tool_calls: list[ToolCallRecord]
    ) -> bool:
        if "invalid_argument_rejection" not in spec.safety_constraints:
            return True
        return any(
            not c.output.get("success", True) for c in tool_calls
            if isinstance(c.output, dict)
        ) or any(not c.success for c in tool_calls)


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def get_adapter(implementation: str) -> AgentAdapter:
    if implementation == Implementation.custom:
        return CustomAgentAdapter()
    if implementation == Implementation.framework:
        return LangGraphAgentAdapter()
    raise ValueError(f"unsupported implementation: {implementation}")
