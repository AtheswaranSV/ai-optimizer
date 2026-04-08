from typing import Dict, Any

TASKS = {
    "easy_1": {
        "id": "T-1001",
        "customer_tier": "free",
        "issue_type": "billing",
        "sentiment": "neutral",
        "urgency_hint": 0.2,
        "history": ["User joined 2 days ago"],
        "description": "I was charged twice for the basic plan. Can I get a refund for the second charge?",
        "ground_truth": {
            "classification": "billing",
            "priority": "low",
            "response_strategy": "auto_reply",
            "customer_tier": "free",
            "sentiment": "neutral"
        }
    },
    "medium_1": {
        "id": "T-2001",
        "customer_tier": "pro",
        "issue_type": "technical",
        "sentiment": "negative",
        "urgency_hint": 0.7,
        "history": ["Ticket #998: Resolved billing issue", "Ticket #1005: Pending technical bug"],
        "description": "The API is returning 500 errors for all my production requests! This is urgent as it's blocking our deployment.",
        "ground_truth": {
            "classification": "technical",
            "priority": "high",
            "response_strategy": "escalate",
            "customer_tier": "pro",
            "sentiment": "negative"
        }
    },
    "hard_1": {
        "id": "T-3001",
        "customer_tier": "enterprise",
        "issue_type": "account",
        "sentiment": "negative",
        "urgency_hint": 0.9,
        "history": ["Key Account Manager contacted yesterday", "Last login: 5 minutes ago"],
        "description": "My account is locked and I cannot access the dashboard. We have a board meeting in 1 hour and I need the reports. I've tried resetting the password but the email never arrives. FIX THIS NOW.",
        "ground_truth": {
            "classification": "technical",  # Deep reasoning: it's a login issue (account) but caused by email delivery failure (technical)
            "priority": "high",
            "response_strategy": "escalate",
            "customer_tier": "enterprise",
            "sentiment": "negative"
        }
    }
}

def get_ticket_data(task_id: str) -> Dict[str, Any]:
    return TASKS.get(task_id, TASKS["easy_1"])
