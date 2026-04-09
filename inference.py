import os
import json
import re
import time
import requests

# ── Evaluator-injected credentials ───────────────────────────────────────────
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://athes7755-ai-workflow-optimizer-env.hf.space")
API_BASE_URL  = os.getenv("API_BASE_URL")   # Injected by hackathon evaluator - REQUIRED
API_KEY       = os.getenv("API_KEY")        # Injected by hackathon evaluator - REQUIRED
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = """You are an expert enterprise support triage AI. Analyse support tickets and respond with a JSON action.

CLASSIFICATION (use ROOT CAUSE, not surface label):
- "billing"   → charges, refunds, pricing, payment
- "technical" → API errors, bugs, login failures, email delivery issues, system outages, failed resets
- "account"   → access permissions, team management (only when no underlying technical failure)
- "other"     → anything else

PRIORITY:
- "high"   → urgency_hint > 0.6, OR production outage, OR enterprise + negative sentiment
- "medium" → urgency_hint 0.3-0.6, OR pro customer issue
- "low"    → urgency_hint < 0.3, free-tier, routine issue

RESPONSE STRATEGY:
- "escalate"     → high priority OR pro/enterprise + negative sentiment OR production impact
- "auto_reply"   → low priority, free-tier, neutral sentiment, routine
- "request_info" → unclear issue needing more details

CRITICAL: Never auto_reply to enterprise customers with negative sentiment.
CRITICAL: An "account" ticket whose root cause is a technical failure (email, login system) → classify as "technical".

Output ONLY valid JSON, no markdown, no explanation:
{"classification": "...", "priority": "...", "response_strategy": "..."}"""


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handles markdown code blocks."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find first JSON object
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def build_user_prompt(obs: dict) -> str:
    return f"""Analyse this support ticket:

Ticket ID    : {obs['ticket_id']}
Customer Tier: {obs['customer_tier']}
Issue Label  : {obs['issue_type']}  (surface label — classify by ROOT CAUSE)
Sentiment    : {obs['sentiment']}
Urgency Hint : {obs['urgency_hint']} (0=low, 1=critical)
History      : {obs.get('history', [])}
Description  : {obs['description']}

Respond with ONLY this JSON:
{{"classification": "<billing|technical|account|other>", "priority": "<low|medium|high>", "response_strategy": "<auto_reply|escalate|request_info>"}}"""


def call_llm(obs: dict) -> dict:
    """Call the evaluator's LLM proxy. No response_format / temperature for max LiteLLM compatibility."""
    from openai import OpenAI

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs)}
        ],
        max_tokens=200
        # NOTE: No response_format — LiteLLM proxies often don't support it
        # NOTE: No temperature  — LiteLLM proxies often don't support it
    )

    content = completion.choices[0].message.content
    return extract_json(content)


def validate_action(action: dict, obs: dict) -> dict:
    """Clamp values to valid enums and apply safety rules."""
    valid_priorities = {"low", "medium", "high"}
    valid_strategies = {"auto_reply", "escalate", "request_info"}

    classification = str(action.get("classification", obs.get("issue_type", "other"))).lower()
    priority       = action.get("priority", "medium")
    strategy       = action.get("response_strategy", "escalate")

    if priority not in valid_priorities:
        priority = "high" if float(obs.get("urgency_hint", 0)) > 0.6 else "low"
    if strategy not in valid_strategies:
        strategy = "escalate"

    # Safety: never auto_reply to enterprise + negative
    if strategy == "auto_reply" and obs.get("customer_tier") == "enterprise" and obs.get("sentiment") == "negative":
        strategy = "escalate"

    return {"classification": classification, "priority": priority, "response_strategy": strategy}


def rule_based_fallback(obs: dict) -> dict:
    """Used ONLY when API_BASE_URL is not injected (local/offline testing)."""
    urgency    = float(obs.get("urgency_hint", 0.5))
    tier       = obs.get("customer_tier", "free")
    sentiment  = obs.get("sentiment", "neutral")
    issue_type = obs.get("issue_type", "other")
    desc       = obs.get("description", "").lower()

    # Root-cause: detect technical failure inside account tickets
    if issue_type == "account" and any(kw in desc for kw in ["password", "email", "login", "locked", "reset"]):
        classification = "technical"
    else:
        classification = issue_type

    if urgency > 0.6 or (tier == "enterprise" and sentiment == "negative") or "production" in desc or "outage" in desc:
        priority = "high"
    elif urgency > 0.3 or tier == "pro":
        priority = "medium"
    else:
        priority = "low"

    if priority == "high" or (tier in ("pro", "enterprise") and sentiment == "negative"):
        strategy = "escalate"
    elif priority == "low" and sentiment != "negative" and tier == "free":
        strategy = "auto_reply"
    else:
        strategy = "request_info"

    if strategy == "auto_reply" and tier == "enterprise" and sentiment == "negative":
        strategy = "escalate"

    return {"classification": classification, "priority": priority, "response_strategy": strategy}


def run_evaluation(task_id: str = "easy_1"):
    print("[START]")

    # 1. Reset environment
    try:
        resp = requests.post(f"{HF_SPACE_URL}/reset?task_id={task_id}", timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}")
        print("[END]")
        return

    print(f"[STEP] state={json.dumps(obs)}")

    # 2. Get action
    # When API_BASE_URL is injected by evaluator → MUST call LLM through their proxy
    # When not set (local/offline) → use rule-based fallback
    if API_BASE_URL and API_KEY:
        try:
            raw_action  = call_llm(obs)
            action_data = validate_action(raw_action, obs)
        except Exception as e:
            print(f"[WARN] LLM call failed ({e}), using rule-based fallback")
            action_data = rule_based_fallback(obs)
    else:
        print("[INFO] API_BASE_URL not set, using rule-based fallback")
        action_data = rule_based_fallback(obs)

    # 3. Submit action
    try:
        step_resp = requests.post(f"{HF_SPACE_URL}/step", json=action_data, timeout=30)
        step_resp.raise_for_status()
        result = step_resp.json()
        reward = result.get("reward", 0)
    except Exception as e:
        print(f"[ERROR] step failed: {e}")
        reward = 0

    print(f"[STEP] action={json.dumps(action_data)} reward={reward}")
    print("[END]")


if __name__ == "__main__":
    for task in ["easy_1", "medium_1", "hard_1"]:
        print(f"\n--- EVALUATING TASK: {task} ---")
        run_evaluation(task)
