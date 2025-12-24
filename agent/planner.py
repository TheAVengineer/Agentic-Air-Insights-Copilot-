"""
Agent planner for deciding which tools to call.

The planner implements a simple but effective planning strategy:
1. Parse user intent from structured request or natural language
2. Determine required data sources
3. Generate execution plan
4. Track plan execution state

This is a lightweight custom planner (not using Semantic Kernel)
designed for the specific Air & Insights use case.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Supported user intents."""
    ANALYZE_AIR_QUALITY = "analyze_air_quality"
    GET_APOD = "get_apod"
    HELP = "help"
    UNKNOWN = "unknown"


class ToolType(str, Enum):
    """Available tools the agent can use."""
    WEATHER_API = "weather_api"
    AIR_QUALITY_API = "air_quality_api"
    NASA_APOD_API = "nasa_apod_api"
    LLM_REASONING = "llm_reasoning"
    CACHE_LOOKUP = "cache_lookup"
    CACHE_STORE = "cache_store"


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    
    step_id: int
    tool: ToolType
    description: str
    params: dict = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan for a user request."""
    
    intent: Intent
    steps: list[PlanStep] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete or failed."""
        return all(
            s.status in ("completed", "failed", "skipped")
            for s in self.steps
        )
    
    @property
    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(s.status == "failed" for s in self.steps)
    
    def get_next_steps(self) -> list[PlanStep]:
        """Get steps that are ready to execute (dependencies met)."""
        ready = []
        completed_ids = {s.step_id for s in self.steps if s.status == "completed"}
        
        for step in self.steps:
            if step.status != "pending":
                continue
            
            # Check if all dependencies are completed
            deps_met = all(d in completed_ids for d in step.depends_on)
            if deps_met:
                ready.append(step)
        
        return ready
    
    def to_dict(self) -> dict:
        """Convert plan to dictionary for logging."""
        return {
            "intent": self.intent.value,
            "steps": [
                {
                    "step_id": s.step_id,
                    "tool": s.tool.value,
                    "description": s.description,
                    "status": s.status,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }


class AgentPlanner:
    """
    Plans execution steps for user requests.
    
    The planner creates execution plans that the orchestrator follows.
    Plans include:
    - Cache lookup (always first for efficiency)
    - API calls (can run in parallel if independent)
    - LLM reasoning (after data is fetched)
    - Cache storage (if new data was fetched)
    """
    
    def __init__(self):
        """Initialize the planner."""
        logger.info("AgentPlanner initialized")
    
    def plan_air_quality_analysis(
        self,
        latitude: float,
        longitude: float,
        hours: int,
    ) -> ExecutionPlan:
        """
        Create execution plan for air quality analysis.
        
        Plan:
        1. Check cache for existing data
        2. If cache miss: fetch air quality and weather in parallel
        3. Generate LLM guidance
        4. Store result in cache
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            hours: Forecast hours
            
        Returns:
            ExecutionPlan with ordered steps
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hours": hours,
        }
        
        steps = [
            # Step 1: Check cache
            PlanStep(
                step_id=1,
                tool=ToolType.CACHE_LOOKUP,
                description="Check cache for recent data",
                params=params,
            ),
            # Step 2: Fetch air quality (if cache miss)
            PlanStep(
                step_id=2,
                tool=ToolType.AIR_QUALITY_API,
                description="Fetch air quality from Open-Meteo",
                params=params,
                depends_on=[1],
            ),
            # Step 3: Fetch weather (parallel with step 2)
            PlanStep(
                step_id=3,
                tool=ToolType.WEATHER_API,
                description="Fetch weather from Open-Meteo",
                params=params,
                depends_on=[1],
            ),
            # Step 4: LLM reasoning (after data fetched)
            PlanStep(
                step_id=4,
                tool=ToolType.LLM_REASONING,
                description="Generate exercise guidance with LLM",
                params=params,
                depends_on=[2, 3],
            ),
            # Step 5: Store in cache
            PlanStep(
                step_id=5,
                tool=ToolType.CACHE_STORE,
                description="Cache result for future requests",
                params=params,
                depends_on=[4],
            ),
        ]
        
        plan = ExecutionPlan(
            intent=Intent.ANALYZE_AIR_QUALITY,
            steps=steps,
            metadata={
                "location": f"{latitude}, {longitude}",
                "hours": hours,
            },
        )
        
        logger.info(f"Created air quality plan: {plan.to_dict()}")
        return plan
    
    def plan_apod_request(self) -> ExecutionPlan:
        """
        Create execution plan for NASA APOD request.
        
        Plan:
        1. Check cache for today's APOD
        2. If cache miss: fetch from NASA API
        3. Generate 2-line summary with LLM
        4. Store in cache
        
        Returns:
            ExecutionPlan for APOD request
        """
        steps = [
            PlanStep(
                step_id=1,
                tool=ToolType.CACHE_LOOKUP,
                description="Check cache for today's APOD",
                params={"type": "apod"},
            ),
            PlanStep(
                step_id=2,
                tool=ToolType.NASA_APOD_API,
                description="Fetch APOD from NASA",
                params={},
                depends_on=[1],
            ),
            PlanStep(
                step_id=3,
                tool=ToolType.LLM_REASONING,
                description="Generate 2-line summary",
                params={"type": "apod_summary"},
                depends_on=[2],
            ),
            PlanStep(
                step_id=4,
                tool=ToolType.CACHE_STORE,
                description="Cache APOD for today",
                params={"type": "apod"},
                depends_on=[3],
            ),
        ]
        
        plan = ExecutionPlan(
            intent=Intent.GET_APOD,
            steps=steps,
            metadata={"type": "apod"},
        )
        
        logger.info(f"Created APOD plan: {plan.to_dict()}")
        return plan
    
    def update_step_status(
        self,
        plan: ExecutionPlan,
        step_id: int,
        status: str,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Update the status of a plan step.
        
        Args:
            plan: The execution plan
            step_id: Step to update
            status: New status
            result: Optional result data
            error: Optional error message
        """
        for step in plan.steps:
            if step.step_id == step_id:
                step.status = status
                step.result = result
                step.error = error
                logger.debug(f"Step {step_id} -> {status}")
                return
        
        logger.warning(f"Step {step_id} not found in plan")
