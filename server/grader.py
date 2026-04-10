from typing import Any, Dict
from .models import Action, Reward

def calculate_reward(action: Action, ground_truth: Dict[str, Any], processing_time: float) -> Reward:
    # Safely extract action fields whether action is a dict or a Pydantic model
    act_cls = action.get("classification") if isinstance(action, dict) else getattr(action, "classification", None)
    act_prio = action.get("priority") if isinstance(action, dict) else getattr(action, "priority", None)
    act_strat = action.get("response_strategy") if isinstance(action, dict) else getattr(action, "response_strategy", None)

    # 1. Classification Accuracy (0.35)
    class_str = str(act_cls).lower() if act_cls else "unknown"
    gt_class_str = str(ground_truth["classification"]).lower()
    class_acc = 1.0 if class_str == gt_class_str else 0.0
    
    # 2. Priority Correctness (0.25)
    priority_map = {"low": 1, "medium": 2, "high": 3}
    act_prio_str = str(act_prio).lower() if act_prio else "unknown"
    act_p = priority_map.get(act_prio_str, 1)
    gt_p = priority_map.get(ground_truth["priority"], 1)
    # Distance penalty: 1.0 - (dist / max_dist)
    prio_corr = 1.0 - (abs(act_p - gt_p) / 2.0)
    
    # 3. Response Quality (0.25)
    resp_strat_str = str(act_strat).lower() if act_strat else "unknown"
    gt_resp_strat_str = str(ground_truth["response_strategy"]).lower()
    resp_qual = 1.0 if resp_strat_str == gt_resp_strat_str else 0.0
    
    # Check for critical failures (e.g., auto-replying to an angry enterprise customer)
    if resp_strat_str == "auto_reply" and ground_truth["customer_tier"] == "enterprise" and ground_truth["sentiment"] == "negative":
        resp_qual *= 0.5  # Penalize risky automation
        
    # 4. Efficiency Score (0.15)
    # Decisions under 5 seconds get 1.0, then decay
    eff_score = 1.0 if processing_time < 5.0 else max(0.0, 1.0 - (processing_time - 5.0) / 10.0)

    # Clamp to strictly (0, 1) using deep margins (0.05 to 0.95) to survive platform rounding
    clamped_class = max(0.05, min(0.95, class_acc))
    clamped_prio = max(0.05, min(0.95, prio_corr))
    clamped_resp = max(0.05, min(0.95, resp_qual))
    clamped_eff = max(0.05, min(0.95, eff_score))
    total = (0.35 * clamped_class) + (0.25 * clamped_prio) + (0.25 * clamped_resp) + (0.15 * clamped_eff)
    
    return Reward(
        classification_accuracy=round(clamped_class, 4),
        priority_correctness=round(clamped_prio, 4),
        response_quality=round(clamped_resp, 4),
        efficiency_score=round(clamped_eff, 4),
        total_reward=round(total, 4)
    )
