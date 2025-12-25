"""
Tests for agent/planner.py execution planning.
"""

import pytest
from unittest.mock import MagicMock

from agent.planner import (
    AgentPlanner,
    ExecutionPlan,
    PlanStep,
    ToolType,
    Intent,
)


class TestPlanStep:
    """Test PlanStep dataclass."""
    
    def test_plan_step_defaults(self):
        """Should initialize with defaults."""
        step = PlanStep(
            step_id=1,
            tool=ToolType.CACHE_LOOKUP,
            description="Test step"
        )
        
        assert step.step_id == 1
        assert step.tool == ToolType.CACHE_LOOKUP
        assert step.description == "Test step"
        assert step.status == "pending"
        assert step.params == {}
        assert step.depends_on == []
        assert step.result is None
    
    def test_plan_step_with_all_fields(self):
        """Should store all provided fields."""
        step = PlanStep(
            step_id=2,
            tool=ToolType.AIR_QUALITY_API,
            description="Fetch air quality",
            status="running",
            params={"lat": 42.0, "lon": 23.0},
            depends_on=[1],
            result={"pm25": 10.0}
        )
        
        assert step.status == "running"
        assert step.params["lat"] == 42.0
        assert step.depends_on == [1]
        assert step.result["pm25"] == 10.0


class TestExecutionPlan:
    """Test ExecutionPlan dataclass."""
    
    def test_execution_plan_is_complete_empty(self):
        """Should be complete when no steps."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[]
        )
        
        assert plan.is_complete is True
    
    def test_execution_plan_is_complete_pending(self):
        """Should not be complete with pending steps."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test", status="pending")
            ]
        )
        
        assert plan.is_complete is False
    
    def test_execution_plan_is_complete_all_done(self):
        """Should be complete when all steps completed."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test", status="completed"),
                PlanStep(step_id=2, tool=ToolType.AIR_QUALITY_API, description="Test", status="failed"),
                PlanStep(step_id=3, tool=ToolType.WEATHER_API, description="Test", status="skipped"),
            ]
        )
        
        assert plan.is_complete is True
    
    def test_execution_plan_has_failures(self):
        """Should detect failures."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test", status="completed"),
                PlanStep(step_id=2, tool=ToolType.AIR_QUALITY_API, description="Test", status="failed"),
            ]
        )
        
        assert plan.has_failures is True
    
    def test_execution_plan_no_failures(self):
        """Should detect no failures."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test", status="completed"),
            ]
        )
        
        assert plan.has_failures is False
    
    def test_get_next_steps_all_pending(self):
        """Should return steps with no dependencies."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test"),
                PlanStep(step_id=2, tool=ToolType.AIR_QUALITY_API, description="Test", depends_on=[1]),
            ]
        )
        
        ready = plan.get_next_steps()
        
        assert len(ready) == 1
        assert ready[0].step_id == 1
    
    def test_get_next_steps_dependencies_met(self):
        """Should return steps with completed dependencies."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test", status="completed"),
                PlanStep(step_id=2, tool=ToolType.AIR_QUALITY_API, description="Test", depends_on=[1]),
                PlanStep(step_id=3, tool=ToolType.LLM_REASONING, description="Test", depends_on=[2]),
            ]
        )
        
        ready = plan.get_next_steps()
        
        assert len(ready) == 1
        assert ready[0].step_id == 2
    
    def test_to_dict(self):
        """Should convert plan to dictionary."""
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=[
                PlanStep(step_id=1, tool=ToolType.CACHE_LOOKUP, description="Test"),
            ],
            metadata={"key": "value"}
        )
        
        result = plan.to_dict()
        
        assert result["intent"] == "analyze_air_quality"
        assert len(result["steps"]) == 1
        assert result["metadata"]["key"] == "value"


class TestAgentPlanner:
    """Test AgentPlanner methods."""
    
    def test_init(self):
        """Should initialize planner."""
        planner = AgentPlanner()
        
        assert planner is not None
    
    def test_plan_air_quality_analysis(self):
        """Should create air quality analysis plan."""
        planner = AgentPlanner()
        
        plan = planner.plan_air_quality_analysis(42.69, 23.32, 6)
        
        assert plan.intent == Intent.ANALYZE_AIR_QUALITY
        assert len(plan.steps) == 5
        assert plan.steps[0].tool == ToolType.CACHE_LOOKUP
        assert plan.steps[1].tool == ToolType.AIR_QUALITY_API
        assert plan.steps[2].tool == ToolType.WEATHER_API
        assert plan.steps[3].tool == ToolType.LLM_REASONING
        assert plan.steps[4].tool == ToolType.CACHE_STORE
    
    def test_plan_apod_request(self):
        """Should create APOD request plan."""
        planner = AgentPlanner()
        
        plan = planner.plan_apod_request()
        
        assert plan.intent == Intent.GET_APOD
        assert len(plan.steps) >= 2
        # Should have cache lookup first
        assert plan.steps[0].tool == ToolType.CACHE_LOOKUP


class TestToolType:
    """Test ToolType enum."""
    
    def test_tool_type_values(self):
        """Should have expected tool types."""
        assert ToolType.CACHE_LOOKUP.value == "cache_lookup"
        assert ToolType.AIR_QUALITY_API.value == "air_quality_api"
        assert ToolType.WEATHER_API.value == "weather_api"
        assert ToolType.LLM_REASONING.value == "llm_reasoning"
        assert ToolType.CACHE_STORE.value == "cache_store"


class TestIntent:
    """Test Intent enum."""
    
    def test_intent_values(self):
        """Should have expected intents."""
        assert Intent.ANALYZE_AIR_QUALITY.value == "analyze_air_quality"
        assert Intent.GET_APOD.value == "get_apod"
