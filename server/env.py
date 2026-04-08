import time
import random
from typing import Dict, Any, Tuple, Optional
from .models import Observation, Action, Reward
from .grader import calculate_reward
from .utils import get_ticket_data

class TicketOptimizationEnv:
    def __init__(self):
        self.current_ticket: Optional[Dict[str, Any]] = None
        self.start_time: float = 0.0
        self.done: bool = False
        self.task_id: str = "easy_1"

    def reset(self, task_id: str = "easy_1") -> Observation:
        self.task_id = task_id
        # Fetch deterministic ticket based on task_id
        self.current_ticket = get_ticket_data(task_id)
        self.start_time = time.time()
        self.done = False
        
        return Observation(
            ticket_id=self.current_ticket["id"],
            customer_tier=self.current_ticket["customer_tier"],
            issue_type=self.current_ticket["issue_type"],
            sentiment=self.current_ticket["sentiment"],
            urgency_hint=self.current_ticket["urgency_hint"],
            history=self.current_ticket["history"],
            description=self.current_ticket["description"]
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if not self.current_ticket or self.done:
            # For robustness, we reset automatically if step is called without active ticket
            self.reset(self.task_id)
            
        processing_time = time.time() - self.start_time
        reward_obj = calculate_reward(action, self.current_ticket["ground_truth"], processing_time)
        
        self.done = True
        
        # In this task-based environment, each step completes the ticket
        obs = self.state() 
        
        return obs, reward_obj.total_reward, True, {"reward_details": reward_obj.dict()}

    def state(self) -> Observation:
        if not self.current_ticket:
            # Initialize with default if needed for state() calls
            self.reset("easy_1")
        return Observation(
            ticket_id=self.current_ticket["id"],
            customer_tier=self.current_ticket["customer_tier"],
            issue_type=self.current_ticket["issue_type"],
            sentiment=self.current_ticket["sentiment"],
            urgency_hint=self.current_ticket["urgency_hint"],
            history=self.current_ticket["history"],
            description=self.current_ticket["description"]
        )

# Global singleton
env = TicketOptimizationEnv()
