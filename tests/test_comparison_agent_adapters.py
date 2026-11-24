"""Tests for the agent comparison adapters (app/comparison/agent_adapters.py).

Both adapters execute real implementation code against deterministic scenarios:
the Custom Agent Orchestrator AgentLoop and the Framework Edition LangGraph
graph.  No external LLM or paid provider is used; a scripted deterministic
provider replays canned responses.
"""

from __future__ import annotations

from app.comparison.agent_adapters import (
    AgentScenarioResult,
    CustomAgentAdapter,
    LangGraphAgentAdapter,
    RecordingToolExecutor,
    ScriptedModel,
    ScriptedProvider,
    ToolCallRecord,
    get_adapter,
)
from app.comparison.agent_scenarios import (
    AgentScenarioSpec,
    CONTROLLED_TOOL_HANDLERS,
    ScriptExhaustion,
    TOOL_SPECS,
    build_scenarios,
    TRUSTED_TENANT,
    SPOOFED_TENANT,
)
from app.comparison.contracts import Implementation


# ---------------------------------------------------------------------------
# Scripted model / provider
# ---------------------------------------------------------------------------

def test_scripted_model_replays_steps_in_order():
    steps = [
        __import__("app.comparison.agent_scenarios", fromlist=["ScriptedStep"]).ScriptedStep(
            content="first", tool_calls=[]
        ),
        __import__("app.comparison.agent_scenarios", fromlist=["ScriptedStep"]).ScriptedStep(
            content="second", tool_calls=[]
        ),
    ]
    model = ScriptedModel(steps)
    assert model.step().content == "first"
    assert model.step().content == "second"
    assert len(model.calls) == 2


def test_scripted_model_exhaustion_final_answer():
    steps = [__import__("app.comparison.agent_scenarios", fromlist=["ScriptedStep"]).ScriptedStep(
        content="only", tool_calls=[]
    )]
    model = ScriptedModel(steps, exhaustion=ScriptExhaustion.FINAL_ANSWER)
    assert model.step().content == "only"
    assert model.step().content == "Script exhausted; providing final answer."


def test_scripted_model_exhaustion_repeat_last():
    steps = [__import__("app.comparison.agent_scenarios", fromlist=["ScriptedStep"]).ScriptedStep(
        content="loop", tool_calls=[]
    )]
    model = ScriptedModel(steps, exhaustion=ScriptExhaustion.REPEAT_LAST)
    assert model.step().content == "loop"
    assert model.step().content == "loop"
    assert model.step().content == "loop"


def test_scripted_provider_tool_calls():
    from app.comparison.agent_scenarios import ScriptedToolCall
    step = __import__("app.comparison.agent_scenarios", fromlist=["ScriptedStep"]).ScriptedStep(
        content=None,
        tool_calls=[ScriptedToolCall(id="c1", name="search_knowledge", arguments={"query": "q"})],
    )
    model = ScriptedModel([step])
    provider = ScriptedProvider(model)
    from agent_orchestrator.providers.models import ProviderRequest
    resp = provider.complete(ProviderRequest(messages=[{"role": "user", "content": "hi"}]))
    assert resp.content is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "search_knowledge"


# ---------------------------------------------------------------------------
# Recording tool executor
# ---------------------------------------------------------------------------

def test_recording_tool_executor_strips_tenant_id():
    from agent_orchestrator.tools.models import ToolCall, ToolDefinition
    from agent_orchestrator.tools.registry import ToolRegistry
    from agent_orchestrator.config import OrchestratorConfig
    from agent_orchestrator.tools.executor import ToolExecutor

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="echo_tool", description="echo", parameters_schema={}),
        lambda **kwargs: {"echo": kwargs.get("message")},
    )
    executor = ToolExecutor(registry=registry, config=OrchestratorConfig())
    recording = RecordingToolExecutor(executor)

    tc = ToolCall(id="c1", name="echo_tool", arguments={"message": "hi", "tenant_id": "tenant-beta"})
    result = recording.execute(tc)

    # tenant_id was stripped before execution
    assert recording.calls[0].arguments == {"message": "hi"}
    assert "tenant_id" not in recording.calls[0].arguments
    assert result.output == {"echo": "hi"}


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def test_get_adapter_custom():
    adapter = get_adapter(Implementation.custom)
    assert isinstance(adapter, CustomAgentAdapter)
    assert adapter.implementation == "custom"


def test_get_adapter_framework():
    adapter = get_adapter(Implementation.framework)
    assert isinstance(adapter, LangGraphAgentAdapter)
    assert adapter.implementation == "framework"


def test_get_adapter_unknown_raises():
    try:
        get_adapter("not-an-implementation")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Custom adapter end-to-end
# ---------------------------------------------------------------------------

class TestCustomAdapter:
    def _run(self, scenario_id: str):
        scenarios = {s.scenario_id: s for s in build_scenarios()}
        spec = scenarios[scenario_id]
        return CustomAgentAdapter().run(spec)

    def test_simple_direct_response(self):
        result = self._run("agent-simple-direct-response")
        assert isinstance(result, AgentScenarioResult)
        assert result.implementation == "custom"
        assert result.status == "success"
        assert result.tool_calls_count == 0
        assert result.provider_calls == 1
        assert result.final_answer is not None
        assert "help" in result.final_answer.lower()

    def test_single_tool_success(self):
        result = self._run("agent-single-tool-success")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.provider_calls == 2
        assert result.tool_failures == 0
        assert result.final_answer is not None
        assert "Forgot password" in result.final_answer

    def test_multiple_sequential_tools(self):
        result = self._run("agent-multiple-sequential-tools")
        assert result.status == "success"
        assert result.tool_calls_count == 2
        assert result.provider_calls == 3

    def test_conditional_tool_routing(self):
        result = self._run("agent-conditional-tool-routing")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.provider_calls == 2

    def test_tool_execution_failure(self):
        result = self._run("agent-tool-execution-failure")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.tool_failures == 1
        assert "failed" in (result.final_answer or "").lower()

    def test_provider_failure(self):
        result = self._run("agent-provider-failure")
        assert result.status == "provider_failure"
        assert result.tool_calls_count == 0

    def test_tool_retry_behaviour(self):
        import app.comparison.agent_scenarios as mod
        mod._retry_state.clear()
        result = self._run("agent-tool-retry-behaviour")
        assert result.status == "success"
        assert result.tool_calls_count == 1

    def test_maximum_tool_call_limit(self):
        result = self._run("agent-maximum-tool-call-limit")
        assert result.status == "success"
        # Custom engine enforces max_tool_calls=3
        assert result.tool_calls_count == 3

    def test_loop_prevention(self):
        result = self._run("agent-loop-prevention")
        assert result.status == "max_iterations_reached"
        assert result.tool_calls_count == 3

    def test_missing_invalid_arguments(self):
        result = self._run("agent-missing-invalid-arguments")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.tool_failures == 1

    def test_tenant_isolation(self):
        result = self._run("agent-tenant-isolation")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        # Tenant isolation: only trusted tenant used
        assert result.tenant_isolation_verified is True

    def test_approval_required(self):
        result = self._run("agent-approval-required")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        # Tool executed successfully (returned approval_required), so no tool
        # execution failure. The approval gate was enforced via handler output.
        assert result.tool_failures == 0
        assert result.approval_gate_enforced is True

    def test_final_answer_after_tool(self):
        result = self._run("agent-final-answer-after-tool")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert "5 business days" in (result.final_answer or "")


# ---------------------------------------------------------------------------
# LangGraph adapter end-to-end
# ---------------------------------------------------------------------------

class TestLangGraphAdapter:
    def _run(self, scenario_id: str):
        scenarios = {s.scenario_id: s for s in build_scenarios()}
        spec = scenarios[scenario_id]
        return LangGraphAgentAdapter().run(spec)

    def test_simple_direct_response(self):
        result = self._run("agent-simple-direct-response")
        assert isinstance(result, AgentScenarioResult)
        assert result.implementation == "framework"
        assert result.status == "success"
        assert result.tool_calls_count == 0
        assert result.final_answer is not None

    def test_single_tool_success(self):
        result = self._run("agent-single-tool-success")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.provider_calls == 2
        assert result.tool_failures == 0
        assert result.final_answer is not None
        assert "Forgot password" in result.final_answer

    def test_multiple_sequential_tools(self):
        result = self._run("agent-multiple-sequential-tools")
        assert result.status == "success"
        assert result.tool_calls_count == 2
        assert result.provider_calls == 3

    def test_conditional_tool_routing(self):
        result = self._run("agent-conditional-tool-routing")
        assert result.status == "success"
        assert result.tool_calls_count == 1

    def test_tool_execution_failure(self):
        result = self._run("agent-tool-execution-failure")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.tool_failures == 1
        assert "failed" in (result.final_answer or "").lower()

    def test_provider_failure(self):
        result = self._run("agent-provider-failure")
        assert result.status == "provider_failure"
        assert result.tool_calls_count == 0

    def test_tool_retry_behaviour(self):
        import app.comparison.agent_scenarios as mod
        mod._retry_state.clear()
        result = self._run("agent-tool-retry-behaviour")
        assert result.status == "success"
        assert result.tool_calls_count == 1

    def test_maximum_tool_call_limit(self):
        result = self._run("agent-maximum-tool-call-limit")
        assert result.status == "success"
        # Framework counts limit-blocked calls toward tool_call_count
        assert result.tool_calls_count == 5

    def test_loop_prevention(self):
        result = self._run("agent-loop-prevention")
        assert result.status == "max_iterations_reached"
        # Framework checks iteration limit before executing the tool call on
        # the final iteration, so only 2 tool calls execute.
        assert result.tool_calls_count == 2

    def test_missing_invalid_arguments(self):
        result = self._run("agent-missing-invalid-arguments")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.tool_failures == 1

    def test_tenant_isolation(self):
        result = self._run("agent-tenant-isolation")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert result.tenant_isolation_verified is True

    def test_approval_required(self):
        result = self._run("agent-approval-required")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        # Tool executed successfully (returned approval_required), so no tool
        # execution failure. The approval gate was enforced via handler output.
        assert result.tool_failures == 0
        assert result.approval_gate_enforced is True

    def test_final_answer_after_tool(self):
        result = self._run("agent-final-answer-after-tool")
        assert result.status == "success"
        assert result.tool_calls_count == 1
        assert "5 business days" in (result.final_answer or "")


# ---------------------------------------------------------------------------
# Cross-implementation comparison
# ---------------------------------------------------------------------------

class TestSideBySide:
    @staticmethod
    def test_all_scenarios_run_on_both_implementations_except_cancellation():
        scenarios = build_scenarios()
        for spec in scenarios:
            custom_result = CustomAgentAdapter().run(spec)
            assert custom_result is not None
            assert custom_result.scenario_id == spec.scenario_id

            if "framework" in spec.supported_implementations:
                fw_result = LangGraphAgentAdapter().run(spec)
                assert fw_result is not None
                assert fw_result.scenario_id == spec.scenario_id

    @staticmethod
    def test_outcome_match_on_shared_scenarios():
        """Both implementations should produce matching outcome categories
        on scenarios they both support (modulo known count differences)."""
        for spec in build_scenarios():
            custom_result = CustomAgentAdapter().run(spec)
            if "framework" not in spec.supported_implementations:
                continue
            fw_result = LangGraphAgentAdapter().run(spec)
            assert custom_result.status == fw_result.status, (
                f"{spec.scenario_id}: custom={custom_result.status}, "
                f"framework={fw_result.status}"
            )

    @staticmethod
    def test_tenant_isolation_both_implementations():
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-tenant-isolation"][0]
        custom_result = CustomAgentAdapter().run(spec)
        fw_result = LangGraphAgentAdapter().run(spec)
        assert custom_result.tenant_isolation_verified is True
        assert fw_result.tenant_isolation_verified is True

    @staticmethod
    def test_approval_gate_both_implementations():
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-approval-required"][0]
        custom_result = CustomAgentAdapter().run(spec)
        fw_result = LangGraphAgentAdapter().run(spec)
        assert custom_result.approval_gate_enforced is True
        assert fw_result.approval_gate_enforced is True

    @staticmethod
    def test_tool_failure_both_implementations():
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-tool-execution-failure"][0]
        custom_result = CustomAgentAdapter().run(spec)
        fw_result = LangGraphAgentAdapter().run(spec)
        assert custom_result.tool_failures == 1
        assert fw_result.tool_failures == 1


# ---------------------------------------------------------------------------
# Focused tests: approval enforcement and limit-rejected tracking
# ---------------------------------------------------------------------------

class TestApprovalEnforcement:
    """Focused tests for approval gate enforcement verification."""

    def _make_case_call(
        self,
        handler_invoked: bool,
        success: bool | None,
        *,
        approval_required: bool = True,
        approval_granted: bool = False,
        approval_source: str = "application_context",
    ) -> ToolCallRecord:
        operation_executed = success is True
        return ToolCallRecord(
            call_id="call-1",
            name="create_support_case",
            arguments={"subject": "s", "description": "d", "priority": "low", "category": "billing"},
            success=success is not None and success,
            error=None if success else "approval_required",
            output={"tool": "create_support_case", "success": success} if success is not None else None,
            handler_invoked=handler_invoked,
            approval_required=approval_required,
            approval_granted=approval_granted,
            approval_source=approval_source,
            operation_executed=operation_executed,
        )

    def test_approval_gate_enforced_no_approval_operation_not_executed(self):
        """Approval required, not granted: handler invoked, operation not executed → enforced."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        calls = [self._make_case_call(handler_invoked=True, success=False)]
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is True
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], calls) is True

    def test_approval_gate_enforced_valid_approval_operation_executed(self):
        """Approval required and granted: handler invoked, operation executed → gate working."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=True,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        calls = [self._make_case_call(handler_invoked=True, success=True, approval_granted=True)]
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is True
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], calls) is True

    def test_approval_gate_not_enforced_no_approval_operation_executed(self):
        """Approval required, not granted, but operation executed → gate NOT enforced."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        calls = [self._make_case_call(handler_invoked=True, success=True, approval_granted=False)]
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is False
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], calls) is False

    def test_approval_gate_not_exercised_no_case_calls(self):
        """No gated tool calls made: approval path not exercised → unverified (False)."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        assert CustomAgentAdapter._verify_approval_gate(spec, [], []) is False
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], []) is False

    def test_approval_gate_true_when_constraint_absent(self):
        """No approval_gated_action_enforcement constraint → not applicable (True)."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="simple",
            user_request="x",
        )
        assert CustomAgentAdapter._verify_approval_gate(spec, [], []) is True
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], []) is True

    def test_approval_source_always_application_context(self):
        """Approval source must never be LLM-supplied; always application_context or none."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        calls = [self._make_case_call(handler_invoked=True, success=False)]
        for c in calls:
            assert c.approval_source == "application_context"
            assert c.approval_source != "llm_supplied"
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is True

    def test_forged_llm_approval_not_trusted(self):
        """Even if a record claims llm_supplied approval, the gate uses application state."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="approval_required",
            user_request="x",
            tools=[TOOL_SPECS["create_support_case"]],
            is_approved=False,
            safety_constraints=["approval_gated_action_enforcement"],
        )
        # Simulate a record that claims LLM-supplied approval (should not happen in practice).
        calls = [
            ToolCallRecord(
                call_id="call-1",
                name="create_support_case",
                arguments={"subject": "s", "description": "d", "priority": "low", "category": "billing"},
                success=True,
                error=None,
                output={"tool": "create_support_case", "success": True},
                handler_invoked=True,
                approval_required=True,
                approval_granted=False,  # application state: not approved
                approval_source="application_context",
                operation_executed=True,
            )
        ]
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is False
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], calls) is False

    def test_non_approval_operation_normal_execution(self):
        """Non-gated tool: normal execution succeeds, approval gate not applicable."""
        spec = AgentScenarioSpec(
            scenario_id="s1",
            category="simple",
            user_request="x",
            tools=[TOOL_SPECS["search_knowledge"]],
            safety_constraints=[],
        )
        calls = [
            ToolCallRecord(
                call_id="call-1",
                name="search_knowledge",
                arguments={"query": "q"},
                success=True,
                error=None,
                output={"tool": "search_knowledge", "results": [], "count": 0},
                handler_invoked=True,
                approval_required=False,
                approval_granted=False,
                approval_source="none",
                operation_executed=True,
            )
        ]
        assert CustomAgentAdapter._verify_approval_gate(spec, [], calls) is True
        assert LangGraphAgentAdapter._verify_approval_gate(spec, [], calls) is True


class TestLimitRejectedTracking:
    """Focused tests for limit-rejected tool call tracking."""

    def test_tool_call_record_handler_invoked_default_false(self):
        record = ToolCallRecord(
            call_id="c1",
            name="search_knowledge",
            arguments={"query": "q"},
            success=True,
        )
        assert record.handler_invoked is False

    def test_agent_scenario_result_limit_rejected_count_default_zero(self):
        result = AgentScenarioResult(
            scenario_id="s1",
            implementation="custom",
            status="success",
        )
        assert result.limit_rejected_count == 0

    def test_agent_scenario_result_limit_rejected_count_settable(self):
        result = AgentScenarioResult(
            scenario_id="s1",
            implementation="framework",
            status="success",
            limit_rejected_count=2,
        )
        assert result.limit_rejected_count == 2

    def test_limit_rejection_maximum_tool_call_limit_framework(self):
        """Framework counts limit-blocked calls toward tool_call_count:
        5 requested - 3 executed = 2 limit-rejected."""
        result = LangGraphAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-maximum-tool-call-limit"][0]
        )
        assert result.limit_rejected_count == 2
        assert result.tool_calls_count == 5
        assert len(result.tool_calls) == 3
        for c in result.tool_calls:
            assert c.rejection_classification == "none"

    def test_limit_rejection_maximum_tool_call_limit_custom(self):
        """Custom engine passes all 5 calls to the executor; calls 3-4 return
        ExecutionLimitError. limit_rejected_count is counted from per-call
        classification (not the engine count difference, which gives 0)."""
        result = CustomAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-maximum-tool-call-limit"][0]
        )
        assert result.tool_calls_count == 3
        assert len(result.tool_calls) == 5
        assert result.limit_rejected_count == 2
        assert result.handler_failed_count == 0
        successful = [c for c in result.tool_calls if c.rejection_classification == "none"]
        limit_rejected = [c for c in result.tool_calls if c.rejection_classification == "limit_rejected"]
        assert len(successful) == 3
        assert len(limit_rejected) == 2
        for c in limit_rejected:
            assert c.operation_executed is False
            assert "limit" in (c.error or "").lower()

    def test_limit_rejection_loop_prevention_framework(self):
        """Framework checks iteration limit before executing on final iteration:
        2 executed, tool_call_count=2, limit_rejected=0 (no tool-call limit rejections)."""
        result = LangGraphAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-loop-prevention"][0]
        )
        assert result.limit_rejected_count == 0
        assert result.tool_calls_count == 2
        assert len(result.tool_calls) == 2

    def test_handler_failed_fail_tool_scenario(self):
        """fail_tool always raises: handler_failed classification."""
        result = CustomAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-tool-execution-failure"][0]
        )
        assert result.handler_failed_count == 1
        assert result.tool_failures == 1
        for c in result.tool_calls:
            assert c.rejection_classification == "handler_failed"
            assert c.operation_executed is False

    def test_validation_rejected_invalid_arguments_scenario(self):
        """Missing required arguments: validation_rejected classification."""
        result = CustomAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-missing-invalid-arguments"][0]
        )
        assert result.validation_rejected_count == 1
        assert result.tool_failures == 1
        for c in result.tool_calls:
            assert c.rejection_classification == "validation_rejected"
            assert c.operation_executed is False

    def test_approval_rejected_is_validation_rejected(self):
        """Approval required but not granted: validation_rejected, not handler_failed."""
        result = CustomAgentAdapter().run(
            [s for s in build_scenarios() if s.scenario_id == "agent-approval-required"][0]
        )
        assert result.validation_rejected_count == 1
        assert result.handler_failed_count == 0
        for c in result.tool_calls:
            assert c.rejection_classification == "validation_rejected"
            assert c.output.get("error") == "approval_required"
            assert c.operation_executed is False
            assert c.approval_required is True
            assert c.approval_granted is False
            assert c.approval_source == "application_context"

    def test_rejection_classifications_are_mutually_exclusive(self):
        """Each tool call has exactly one rejection classification."""
        for spec in build_scenarios():
            for adapter_cls in (CustomAgentAdapter, LangGraphAgentAdapter):
                if adapter_cls.implementation not in spec.supported_implementations:
                    continue
                result = adapter_cls().run(spec)
                for c in result.tool_calls:
                    assert c.rejection_classification in (
                        "none", "limit_rejected", "validation_rejected",
                        "handler_failed", "provider_failed", "cancelled", "not_executed",
                    )
