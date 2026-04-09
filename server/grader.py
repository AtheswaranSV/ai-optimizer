from typing import Any, Dict
from .models import Action, Reward

def calculate_reward(action: Action, ground_truth: Dict[str, Any], processing_time: float) -> Reward:
    # 1. Classification Accuracy (0.35)
    class_acc = 1.0 if action.classification.lower() == ground_truth["classification"].lower() else 0.0
    
    # 2. Priority Correctness (0.25)
    priority_map = {"low": 1, "medium": 2, "high": 3}
    act_p = priority_map.get(action.priority, 1)
    gt_p = priority_map.get(ground_truth["priority"], 1)
    # Distance penalty: 1.0 - (dist / max_dist)
    prio_corr = 1.0 - (abs(act_p - gt_p) / 2.0)
    
    # 3. Response Quality (0.25)
    resp_qual = 1.0 if action.response_strategy == ground_truth["response_strategy"] else 0.0
    
    # Check for critical failures (e.g., auto-replying to an angry enterprise customer)
    if action.response_strategy == "auto_reply" and ground_truth["customer_tier"] == "enterprise" and ground_truth["sentiment"] == "negative":
        resp_qual *= 0.5  # Penalize risky automation
        
    # 4. Efficiency Score (0.15)
    # Decisions under 5 seconds get 1.0, then decay
    eff_score = 1.0 if processing_time < 5.0 else max(0.0, 1.0 - (processing_time - 5.0) / 10.0)

    total = (0.35 * class_acc) + (0.25 * prio_corr) + (0.25 * resp_qual) + (0.15 * eff_score)
    
    # Clamp to strictly (0, 1) — OpenEnv requires scores exclusively between 0 and 1
    total = round(max(0.001, min(0.999, total)), 4)
    
    return Reward(
        classification_accuracy=round(max(0.001, min(0.999, class_acc)), 4),
        priority_correctness=round(max(0.001, min(0.999, prio_corr)), 4),
        response_quality=round(max(0.001, min(0.999, resp_qual)), 4),
        efficiency_score=round(max(0.001, min(0.999, eff_score)), 4),
        total_reward=total
    )
