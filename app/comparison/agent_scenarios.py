"""Versioned, deterministic agent evaluation dataset for F4.4.

This module defines the shared agent comparison scenarios used to exercise the
Custom Agent Orchestrator and the Framework Edition (LangGraph) side by side.

Every scenario is implementation-agnostic: it supplies a normalized user
request, a scripted deterministic provider, controlled tool handlers, expected
behaviour, and safety constraints.  No external LLM or paid provider is
required; adapters replay the script against real execution code.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Scripted provider steps
# ---------------------------------------------------------------------------

class ScriptedToolCall(BaseModel):
    """A single tool call the scripted provider asks the agent to make."""

    id: str = Field(..., min_length=1, description="Unique call identifier.")
    name: str = Field(..., min_length=1, description="Tool name to invoke.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="JSON-compatible tool arguments."
    )


class ScriptedStep(BaseModel):
    """One deterministic provider response, replayed in order across iterations."""

    content: str | None = Field(
        default=None, description="Final-answer text produced by this step."
    )
    tool_calls: list[ScriptedToolCall] = Field(
        default_factory=list, description="Tool calls requested by this step."
    )
    raises: str | None = Field(
        default=None,
        description="If set, the scripted provider raises this exception class "
        "instead of returning a response (e.g. 'ProviderError').",
    )


class ScriptExhaustion(str, Enum):
    """Behaviour when the script runs out of steps."""

    FINAL_ANSWER = "final_answer"
    REPEAT_LAST = "repeat_last"
    PROVIDER_ERROR = "provider_error"


# ---------------------------------------------------------------------------
# Tool specifications
# ---------------------------------------------------------------------------

class ToolSpec(BaseModel):
    """Controlled tool definition shared by both adapters."""

    name: str = Field(..., min_length=1, description="Unique tool name.")
    description: str = Field(..., min_length=1, description="Tool description.")
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool arguments."
    )
    kind: str = Field(
        default="controlled",
        description="Tool family: controlled, search_knowledge, "
        "create_support_case, lookup_ticket_history, fail_tool, retry_tool.",
    )


# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------

class AgentScenarioSpec(BaseModel):
    """A single deterministic agent comparison scenario."""

    scenario_id: str = Field(..., min_length=1, description="Unique scenario id.")
    category: str = Field(
        ..., description="Behaviour category (see SCENARIO_CATEGORIES)."
    )
    user_request: str = Field(..., min_length=1, description="End-user request.")
    description: str = Field(
        default="", description="Human-readable scenario description."
    )
    script: list[ScriptedStep] = Field(
        default_factory=list,
        description="Deterministic provider responses, replayed in order.",
    )
    script_exhaustion: ScriptExhaustion = Field(
        default=ScriptExhaustion.FINAL_ANSWER,
        description="Behaviour when the script is exhausted.",
    )
    tools: list[ToolSpec] = Field(
        default_factory=list, description="Tools available to the agent."
    )
    trusted_tenant_id: str = Field(
        default="tenant-alpha",
        description="Trusted server-side tenant id bound into tool handlers.",
    )
    spoofed_tenant_id: str = Field(
        default="tenant-beta",
        description="LLM-supplied tenant id that must never override context.",
    )
    is_approved: bool = Field(
        default=False, description="Trusted approval state (application context)."
    )
    requires_approval: bool = Field(
        default=False, description="Whether the scenario exercises a gated tool."
    )
    max_iterations: int = Field(
        default=10, gt=0, description="Maximum agent loop iterations."
    )
    max_tool_calls: int = Field(
        default=20, gt=0, description="Maximum tool calls per run."
    )
    provider_retry_max_attempts: int = Field(
        default=1, ge=1, description="Custom provider retry attempts (1 = none)."
    )
    tool_retry_max_attempts: int = Field(
        default=1, ge=1, description="Custom tool retry attempts (1 = none)."
    )
    expected_tool_names: list[str] = Field(
        default_factory=list, description="Tool names expected to be invoked."
    )
    expected_outcome: str = Field(
        default="success",
        description="Expected final outcome: success, provider_failure, "
        "max_iterations_reached, tool_failure, approval_required.",
    )
    expected_tool_call_count: int | None = Field(
        default=None, description="Expected number of executed tool calls."
    )
    expected_final_answer_contains: str | None = Field(
        default=None, description="Substring expected in the final answer."
    )
    expected_tool_failures: int = Field(
        default=0, description="Expected number of failed tool calls."
    )
    cancellation_supported: bool = Field(
        default=False,
        description="Whether cancellation/interruption is exercised for this scenario.",
    )
    supported_implementations: list[str] = Field(
        default_factory=lambda: ["custom", "framework"],
        description="Implementations that can execute this scenario.",
    )
    safety_constraints: list[str] = Field(
        default_factory=list,
        description="Safety properties verified by this scenario.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Controlled tool handlers (deterministic, no side effects)
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, list[dict[str, str]]] = {
    "tenant-alpha": [
        {
            "content": "To reset your password, go to Account Settings and click "
            "'Forgot password'. A reset link is sent to your email.",
            "document": "password_reset",
        },
        {
            "content": "Refund requests are reviewed within 5 business days for "
            "eligible charges within the last 30 days.",
            "document": "billing_refunds",
        },
        {
            "content": "If you cannot log in, verify your credentials and clear "
            "your browser cache or use the account recovery flow.",
            "document": "login_troubleshooting",
        },
    ],
    "tenant-beta": [
        {
            "content": "Tenant-beta confidential: unrelated account notes.",
            "document": "beta_private",
        }
    ],
}

_retry_state: dict[str, int] = {}


def _redact(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive-looking values from tool arguments for reporting."""
    sensitive = {
        "email", "phone", "ssn", "token", "password", "secret", "api_key",
        "apikey", "credit_card", "card_number", "authorization",
    }
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if key.lower() in sensitive:
            out[key] = "[REDACTED]"
        else:
            out[key] = value
    return out


def handle_search_knowledge(
    arguments: dict[str, Any], *, tenant_id: str | None = None
) -> dict[str, Any]:
    """Deterministic knowledge search bound to the trusted tenant id."""
    query = (arguments.get("query") or "").lower()
    docs = _KNOWLEDGE_BASE.get(tenant_id or "", [])
    matches = [d for d in docs if query and query in d["content"].lower()]
    return {
        "tool": "search_knowledge",
        "results": matches,
        "count": len(matches),
        "tenant_id": tenant_id,
    }


def handle_lookup_ticket_history(
    arguments: dict[str, Any], *, tenant_id: str | None = None
) -> list[dict[str, Any]]:
    """Deterministic ticket history lookup (read-only, empty fixture)."""
    return []


def handle_create_support_case(
    arguments: dict[str, Any],
    *,
    tenant_id: str | None = None,
    is_approved: bool = False,
) -> dict[str, Any]:
    """Deterministic support-case creation, gated by trusted approval state.

    When not approved the action is NOT performed; an approval_required result
    is returned instead. No real ticket, agent assignment, or notification is
    created.
    """
    if not is_approved:
        return {
            "tool": "create_support_case",
            "success": False,
            "error": "approval_required",
            "message": "create_support_case requires approval and was not executed.",
            "results": [],
        }
    return {
        "tool": "create_support_case",
        "success": True,
        "case_id": "CASE-TEST-001",
        "status": "created",
        "tenant_id": tenant_id,
        "results": [],
    }


def handle_fail_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool that always raises a simulated execution failure."""
    raise RuntimeError("simulated tool failure")


def handle_retry_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool that fails a configured number of times then succeeds."""
    key = "retry_tool"
    _retry_state[key] = _retry_state.get(key, 0) + 1
    fail_until = int(arguments.get("fail_until", 1))
    if _retry_state[key] < fail_until:
        raise RuntimeError("transient failure")
    return {"tool": "retry_tool", "result": "ok", "attempt": _retry_state[key]}


def handle_echo_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Tool that echoes its (redacted) message argument."""
    return {"tool": "echo_tool", "echo": arguments.get("message", "")}


CONTROLLED_TOOL_HANDLERS: dict[str, Any] = {
    "search_knowledge": handle_search_knowledge,
    "lookup_ticket_history": handle_lookup_ticket_history,
    "create_support_case": handle_create_support_case,
    "fail_tool": handle_fail_tool,
    "retry_tool": handle_retry_tool,
    "echo_tool": handle_echo_tool,
}

# Schemas for the controlled tools.
TOOL_SPECS: dict[str, ToolSpec] = {
    "search_knowledge": ToolSpec(
        name="search_knowledge",
        description="Search the authenticated tenant's support documentation.",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "lookup_ticket_history": ToolSpec(
        name="lookup_ticket_history",
        description="Look up the tenant's historical support tickets.",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1}},
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "create_support_case": ToolSpec(
        name="create_support_case",
        description="Create a support case (approval required).",
        parameters_schema={
            "type": "object",
            "properties": {
                "subject": {"type": "string", "minLength": 1, "maxLength": 200},
                "description": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 2000,
                },
                "priority": {
                    "type": "string",
                    "pattern": "^(low|medium|high|urgent)$",
                },
                "category": {"type": "string", "minLength": 1, "maxLength": 100},
            },
            "required": ["subject", "description", "priority", "category"],
            "additionalProperties": False,
        },
    ),
    "fail_tool": ToolSpec(
        name="fail_tool",
        description="A tool that always fails (simulated execution failure).",
        parameters_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
    "retry_tool": ToolSpec(
        name="retry_tool",
        description="A tool that fails N times then succeeds (retry behaviour).",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "fail_until": {"type": "integer", "default": 1},
            },
            "additionalProperties": False,
        },
    ),
    "echo_tool": ToolSpec(
        name="echo_tool",
        description="Echoes its message argument.",
        parameters_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
}


# ---------------------------------------------------------------------------
# Scenario dataset
# ---------------------------------------------------------------------------

TRUSTED_TENANT = "tenant-alpha"
SPOOFED_TENANT = "tenant-beta"


def _step(
    content: str | None = None,
    tool_calls: list[ScriptedToolCall] | None = None,
    raises: str | None = None,
) -> ScriptedStep:
    return ScriptedStep(
        content=content,
        tool_calls=list(tool_calls or []),
        raises=raises,
    )


def _tc(
    call_id: str,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> ScriptedToolCall:
    return ScriptedToolCall(id=call_id, name=name, arguments=arguments or {})


def build_scenarios() -> list[AgentScenarioSpec]:
    """Build the complete, versioned agent evaluation dataset.

    Scenarios use realistic support-ticket workflows and deterministic fake
    providers/handlers so execution is reproducible without paid APIs.
    """
    scenarios: list[AgentScenarioSpec] = []

    # 1. Simple direct response without tool execution.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-simple-direct-response",
            category="simple_direct_response",
            user_request="Hello, can you help me with my account?",
            description="Model answers directly with no tool calls.",
            script=[
                _step(content="I can help with your account. What do you need?")
            ],
            expected_outcome="success",
            expected_tool_call_count=0,
            expected_final_answer_contains="help",
            safety_constraints=["no_unauthorized_tool_execution"],
        )
    )

    # 2. Successful single-tool execution.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-single-tool-success",
            category="single_tool_execution",
            user_request="How do I reset my password?",
            description="Model calls search_knowledge once, then answers.",
            script=[
                _step(
                    tool_calls=[
                        _tc("call-1", "search_knowledge", {"query": "password reset"})
                    ]
                ),
                _step(
                    content="To reset your password, go to Account Settings and click 'Forgot password'."
                ),
            ],
            tools=[TOOL_SPECS["search_knowledge"]],
            expected_tool_names=["search_knowledge"],
            expected_outcome="success",
            expected_tool_call_count=1,
            expected_final_answer_contains="Forgot password",
        )
    )

    # 3. Multiple sequential tool calls.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-multiple-sequential-tools",
            category="multiple_sequential_tool_calls",
            user_request="Find my billing info and ticket history.",
            description="Model calls search_knowledge then lookup_ticket_history.",
            script=[
                _step(
                    tool_calls=[
                        _tc("call-1", "search_knowledge", {"query": "billing refund"})
                    ]
                ),
                _step(
                    tool_calls=[
                        _tc(
                            "call-2",
                            "lookup_ticket_history",
                            {"query": "billing"},
                        )
                    ]
                ),
                _step(content="Here is your billing info and ticket history."),
            ],
            tools=[
                TOOL_SPECS["search_knowledge"],
                TOOL_SPECS["lookup_ticket_history"],
            ],
            expected_tool_names=["search_knowledge", "lookup_ticket_history"],
            expected_outcome="success",
            expected_tool_call_count=2,
        )
    )

    # 4. Conditional tool routing.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-conditional-tool-routing",
            category="conditional_tool_routing",
            user_request="I have a billing refund question.",
            description="Model routes to search_knowledge for a billing query.",
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1",
                            "search_knowledge",
                            {"query": "billing refund"},
                        )
                    ]
                ),
                _step(content="Refunds are reviewed within 5 business days."),
            ],
            tools=[
                TOOL_SPECS["search_knowledge"],
                TOOL_SPECS["create_support_case"],
            ],
            expected_tool_names=["search_knowledge"],
            expected_outcome="success",
            expected_tool_call_count=1,
        )
    )

    # 5. Tool execution failure.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-tool-execution-failure",
            category="tool_execution_failure",
            user_request="Look up a record that fails.",
            description="Model calls fail_tool, which errors, then answers.",
            script=[
                _step(tool_calls=[_tc("call-1", "fail_tool", {"query": "x"})]),
                _step(content="The lookup failed, sorry."),
            ],
            tools=[TOOL_SPECS["fail_tool"]],
            expected_tool_names=["fail_tool"],
            expected_outcome="success",
            expected_tool_call_count=1,
            expected_tool_failures=1,
            expected_final_answer_contains="failed",
        )
    )

    # 6. Provider failure or malformed response.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-provider-failure",
            category="provider_failure",
            user_request="Anything at all.",
            description="Provider raises on the first request.",
            script=[_step(raises="ProviderError")],
            provider_retry_max_attempts=1,
            expected_outcome="provider_failure",
            expected_tool_call_count=0,
            safety_constraints=["provider_failure_is_safe"],
        )
    )

    # 7. Tool retry behaviour.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-tool-retry-behaviour",
            category="tool_retry_behaviour",
            user_request="Use the flaky lookup tool.",
            description="retry_tool fails once then succeeds on retry.",
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1", "retry_tool", {"query": "x", "fail_until": 2}
                        )
                    ]
                ),
                _step(content="Done after retry."),
            ],
            tools=[TOOL_SPECS["retry_tool"]],
            tool_retry_max_attempts=2,
            expected_tool_names=["retry_tool"],
            expected_outcome="success",
            expected_tool_call_count=1,
        )
    )

    # 8. Maximum tool-call limit.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-maximum-tool-call-limit",
            category="maximum_tool_call_limit",
            user_request="Do many lookups at once.",
            description="Model requests 5 tools; only 3 may execute.",
            script=[
                _step(
                    tool_calls=[
                        _tc(f"call-{i}", "search_knowledge", {"query": f"q{i}"})
                        for i in range(5)
                    ]
                ),
                _step(content="Stopped at the tool-call limit."),
            ],
            tools=[TOOL_SPECS["search_knowledge"]],
            max_tool_calls=3,
            expected_tool_names=["search_knowledge"],
            expected_outcome="success",
            expected_tool_call_count=3,
        )
    )

    # 9. Repeated tool-call / loop prevention.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-loop-prevention",
            category="loop_prevention",
            user_request="Keep searching forever.",
            description="Model requests the same tool every iteration; loop stops.",
            script=[
                _step(
                    tool_calls=[
                        _tc("call-loop", "search_knowledge", {"query": "loop"})
                    ]
                )
            ],
            script_exhaustion=ScriptExhaustion.REPEAT_LAST,
            tools=[TOOL_SPECS["search_knowledge"]],
            max_iterations=3,
            expected_outcome="max_iterations_reached",
            expected_tool_call_count=3,
            safety_constraints=["loop_prevention_enforced"],
        )
    )

    # 10. Missing or invalid tool arguments.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-missing-invalid-arguments",
            category="missing_invalid_arguments",
            user_request="Create a case with incomplete arguments.",
            description="Model calls create_support_case missing required subject.",
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1",
                            "create_support_case",
                            {
                                "description": "d",
                                "priority": "low",
                                "category": "billing",
                            },
                        )
                    ]
                ),
                _step(content="Case creation was rejected due to invalid arguments."),
            ],
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=True,
            expected_tool_names=["create_support_case"],
            expected_outcome="success",
            expected_tool_call_count=1,
            expected_tool_failures=1,
            safety_constraints=["invalid_argument_rejection"],
        )
    )

    # 11. Tenant isolation.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-tenant-isolation",
            category="tenant_isolation",
            user_request="Read my tenant's support records.",
            description=(
                "Model supplies a spoofed tenant_id; trusted tenant context "
                "must be used instead."
            ),
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1",
                            "search_knowledge",
                            {"query": "password", "tenant_id": SPOOFED_TENANT},
                        )
                    ]
                ),
                _step(content="Found your records."),
            ],
            tools=[TOOL_SPECS["search_knowledge"]],
            trusted_tenant_id=TRUSTED_TENANT,
            spoofed_tenant_id=SPOOFED_TENANT,
            expected_tool_names=["search_knowledge"],
            expected_outcome="success",
            expected_tool_call_count=1,
            safety_constraints=[
                "cross_tenant_access_prevention",
                "tenant_id_from_trusted_context",
            ],
        )
    )

    # 12. Approval-required operation.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-approval-required",
            category="approval_required",
            user_request="Create a support case for a billing issue.",
            description=(
                "create_support_case is gated; without approval it must not execute."
            ),
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1",
                            "create_support_case",
                            {
                                "subject": "Billing dispute",
                                "description": "Wrong charge",
                                "priority": "high",
                                "category": "billing",
                            },
                        )
                    ]
                ),
                _step(
                    content="Your request requires approval before a case can be created."
                ),
            ],
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            requires_approval=True,
            expected_tool_names=["create_support_case"],
            expected_outcome="success",
            expected_tool_call_count=1,
            expected_tool_failures=1,
            safety_constraints=["approval_gated_action_enforcement"],
        )
    )

    # 13. Cancellation / execution interruption (custom supports async cancel).
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-cancellation",
            category="cancellation",
            user_request="A long-running lookup.",
            description=(
                "Cancellation propagation, where supported by the implementation."
            ),
            script=[_step(content="Answer.")],
            cancellation_supported=True,
            supported_implementations=["custom"],
            expected_outcome="success",
        )
    )

    # 14. Final answer after successful tool execution.
    scenarios.append(
        AgentScenarioSpec(
            scenario_id="agent-final-answer-after-tool",
            category="final_answer_after_tool",
            user_request="What is the refund policy?",
            description="Model calls search_knowledge, then answers from the result.",
            script=[
                _step(
                    tool_calls=[
                        _tc(
                            "call-1",
                            "search_knowledge",
                            {"query": "refund policy"},
                        )
                    ]
                ),
                _step(
                    content="Refund requests are reviewed within 5 business days for eligible charges."
                ),
            ],
            tools=[TOOL_SPECS["search_knowledge"]],
            expected_tool_names=["search_knowledge"],
            expected_outcome="success",
            expected_tool_call_count=1,
            expected_final_answer_contains="5 business days",
        )
    )

    return scenarios


SCENARIO_CATEGORIES: list[str] = sorted({s.category for s in build_scenarios()})


class AgentEvaluationDataset(BaseModel):
    """Versioned agent evaluation dataset for the F4.4 comparison."""

    dataset_id: str = Field(default="agent-eval-v1", min_length=1)
    name: str = "agent-evaluation"
    version: str = "1.0.0"
    description: str = (
        "Deterministic support-ticket agent scenarios for comparing the Custom "
        "Agent Orchestrator and the Framework Edition (LangGraph)."
    )
    scenarios: list[AgentScenarioSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_agent_dataset() -> AgentEvaluationDataset:
    """Return the canonical versioned agent evaluation dataset."""
    scenarios = build_scenarios()
    return AgentEvaluationDataset(
        scenarios=scenarios,
        metadata={
            "total_scenarios": len(scenarios),
            "categories": SCENARIO_CATEGORIES,
        },
    )
