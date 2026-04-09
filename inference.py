import os
import json
import re
import requests

# ── Evaluator-injected credentials (read exactly as instructed) ───────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY      = os.environ.get("API_KEY", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://athes7755-ai-workflow-optimizer-env.hf.space")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an expert enterprise support triage AI.

CLASSIFICATION (use ROOT CAUSE, not the surface label):
- "billing"   → charges, refunds, pricing, payment
- "technical" → API errors, bugs, login failures, email delivery issues, outages, failed resets
- "account"   → access permissions, team management (only when no underlying technical failure)
- "other"     → anything else

PRIORITY:
- "high"   → urgency_hint > 0.6  OR  production outage  OR  enterprise + negative sentiment
- "medium" → urgency_hint 0.3-0.6  OR  pro customer with bug
- "low"    → urgency_hint < 0.3, free-tier, routine issue

RESPONSE STRATEGY:
- "escalate"     → high priority  OR  pro/enterprise + negative sentiment  OR  production impact
- "auto_reply"   → low priority, free-tier, neutral sentiment, routine issue
- "request_info" → ambiguous, needs more information before acting

RULES:
- NEVER use auto_reply for enterprise + negative sentiment.
- If an "account" ticket has a technical root cause (failed email, login system) → classify as "technical".

Respond ONLY with valid JSON, no markdown, no explanation:
{"classification": "...", "priority": "...", "response_strategy": "..."}"""


def build_prompt(obs: dict) -> str:
    return f"""Analyse this support ticket:

Ticket ID    : {obs['ticket_id']}
Customer Tier: {obs['customer_tier']}
Issue Label  : {obs['issue_type']}  (surface label — always classify by ROOT CAUSE)
Sentiment    : {obs['sentiment']}
Urgency Hint : {obs['urgency_hint']} (0=none, 1=critical)
History      : {obs.get('history', [])}
Description  : {obs['description']}

Return ONLY this JSON:
{{"classification": "<billing|technical|account|other>", "priority": "<low|medium|high>", "response_strategy": "<auto_reply|escalate|request_info>"}}"""


def parse_json(text: str) -> dict:
    """Robustly extract JSON object from LLM response text."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def call_llm(obs: dict) -> dict:
    """
    Call the LiteLLM proxy using direct HTTP requests.
    This avoids any openai library version compatibility issues.
    Uses the evaluator-injected API_BASE_URL and API_KEY.
    """
    base = API_BASE_URL.rstrip("/")
    endpoint = f"{base}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(obs)}
        ],
        "max_tokens": 200
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return parse_json(content)


def validate_action(action: dict, obs: dict) -> dict:
    """Clamp all values to valid enums and enforce safety rules."""
    valid_p = {"low", "medium", "high"}
    valid_s = {"auto_reply", "escalate", "request_info"}

    classification = str(action.get("classification", obs.get("issue_type", "other"))).lower()
    priority       = action.get("priority", "medium")
    strategy       = action.get("response_strategy", "escalate")

    if priority not in valid_p:
        priority = "high" if float(obs.get("urgency_hint", 0)) > 0.6 else "low"
    if strategy not in valid_s:
        strategy = "escalate"

    # Safety: never auto_reply for enterprise + negative sentiment
    if strategy == "auto_reply" \
            and obs.get("customer_tier") == "enterprise" \
            and obs.get("sentiment") == "negative":
        strategy = "escalate"

    return {
        "classification":    classification,
        "priority":          priority,
        "response_strategy": strategy
    }


def run_evaluation(task_id: str = "easy_1"):
    print("[START]")

    # 1. Reset the environment
    resp = requests.post(
        f"{HF_SPACE_URL}/reset?task_id={task_id}",
        timeout=30
    )
    resp.raise_for_status()
    obs = resp.json()
    print(f"[STEP] state={json.dumps(obs)}")

    # 2. Call LLM through the evaluator's proxy (mandatory)
    raw_action  = call_llm(obs)
    action_data = validate_action(raw_action, obs)

    # 3. Submit action to environment
    step_resp = requests.post(
        f"{HF_SPACE_URL}/step",
        json=action_data,
        timeout=30
    )
    step_resp.raise_for_status()
    result = step_resp.json()
    reward = result.get("reward", 0)

    print(f"[STEP] action={json.dumps(action_data)} reward={reward}")
    print("[END]")


if __name__ == "__main__":
    for task in ["easy_1", "medium_1", "hard_1"]:
        print(f"\n--- EVALUATING TASK: {task} ---")
        run_evaluation(task)
