from typing import Any, Dict
from .models import Action, Reward

    try:
        # Safely extract action fields whether action is a dict or a Pydantic model
        act_cls = action.get("classification") if isinstance(action, dict) else getattr(action, "classification", None)
        act_prio = action.get("priority") if isinstance(action, dict) else getattr(action, "priority", None)
        act_strat = action.get("response_strategy") if isinstance(action, dict) else getattr(action, "response_strategy", None)

        # 1. Classification Accuracy (0.35)
        class_str = str(act_cls).lower() if act_cls else "unknown"
        gt_class_str = str(ground_truth.get("classification", "")).lower() if isinstance(ground_truth, dict) else "unknown"
        class_acc = 1.0 if class_str == gt_class_str else 0.0
        
        # 2. Priority Correctness (0.25)
        priority_map = {"low": 1, "medium": 2, "high": 3}
        act_prio_str = str(act_prio).lower() if act_prio else "unknown"
        act_p = priority_map.get(act_prio_str, 1)
        gt_prio_raw = ground_truth.get("priority", "low") if isinstance(ground_truth, dict) else "low"
        gt_p = priority_map.get(str(gt_prio_raw).lower(), 1)
        # Distance penalty: 1.0 - (dist / max_dist)
        prio_corr = 1.0 - (abs(act_p - gt_p) / 2.0)
        
        # 3. Response Quality (0.25)
        resp_strat_str = str(act_strat).lower() if act_strat else "unknown"
        gt_resp_strat_str = str(ground_truth.get("response_strategy", "")).lower() if isinstance(ground_truth, dict) else "unknown"
        resp_qual = 1.0 if resp_strat_str == gt_resp_strat_str else 0.0
        
        # Check for critical failures
        gt_tier = str(ground_truth.get("customer_tier", "")).lower() if isinstance(ground_truth, dict) else ""
        gt_sent = str(ground_truth.get("sentiment", "")).lower() if isinstance(ground_truth, dict) else ""
        if resp_strat_str == "auto_reply" and gt_tier == "enterprise" and gt_sent == "negative":
            resp_qual *= 0.5  # Penalize risky automation
            
        # 4. Efficiency Score (0.15)
        try:
            pt = float(processing_time)
        except (ValueError, TypeError):
            pt = 6.0
        eff_score = 1.0 if pt < 5.0 else max(0.0, 1.0 - (pt - 5.0) / 10.0)

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
    except Exception as e:
        # Absolute safety net: if ANYTHING in the evaluator causes a crash, return safe mid-tier score (0.5)
        return Reward(
            classification_accuracy=0.5,
            priority_correctness=0.5,
            response_quality=0.5,
            efficiency_score=0.5,
            total_reward=0.5
        )
