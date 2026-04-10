from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class Observation(BaseModel):
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    customer_tier: Literal["free", "pro", "enterprise"] = Field(..., description="Customer subscription level")
    issue_type: Literal["billing", "technical", "account", "other"] = Field(..., description="Estimated category of the issue")
    sentiment: Literal["positive", "neutral", "negative"] = Field(..., description="Estimated customer sentiment")
    urgency_hint: float = Field(..., ge=0.0, le=1.0, description="Heuristic-based urgency score")
    history: List[str] = Field(default_factory=list, description="Previous interaction history")
    available_actions: List[str] = Field(
        default=["auto_reply", "escalate", "request_info"],
        description="List of valid response strategies"
    )
    description: str = Field(..., description="The actual text content of the support ticket")

class Action(BaseModel):
    classification: Optional[str] = Field(default="unknown", description="Agent's refined category for the ticket")
    priority: Optional[str] = Field(default="unknown", description="Assigned priority level")
    response_strategy: Optional[str] = Field(default="unknown", description="Action to take")

class Reward(BaseModel):
    classification_accuracy: float = Field(..., ge=0.0, le=1.0)
    priority_correctness: float = Field(..., ge=0.0, le=1.0)
    response_quality: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    total_reward: float = Field(..., ge=0.0, le=1.0)

class TaskConfig(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    seed: int
