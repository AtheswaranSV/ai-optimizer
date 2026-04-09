import os
import json
import re
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://athes7755-ai-workflow-optimizer-env.hf.space")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert enterprise support triage AI.

CLASSIFICATION (use ROOT CAUSE, not the surface label):
- "billing"   → charges, refunds, pricing, payment
- "technical" → API errors, bugs, login failures, email delivery issues, outages, failed password resets
- "account"   → access permissions, team management (only when no underlying technical failure)
- "other"     → everything else

PRIORITY:
- "high"   → urgency_hint > 0.6  OR  production outage  OR  enterprise + negative sentiment
- "medium" → urgency_hint 0.3-0.6  OR  pro customer issue
- "low"    → urgency_hint < 0.3, free-tier, routine

RESPONSE STRATEGY:
- "escalate"     → high priority  OR  pro/enterprise + negative sentiment  OR  production impact
- "auto_reply"   → low priority, free-tier, neutral sentiment, routine issue
- "request_info" → ambiguous, needs more context

RULES:
- NEVER auto_reply to enterprise + negative sentiment customers.
- If an "account" ticket's root cause is technical (email delivery, login system) → classify as "technical".

Output ONLY valid JSON, nothing else:
{"classification": "...", "priority": "...", "response_strategy": "..."}"""


def build_prompt(obs: dict) -> str:
    return f"""Analyse this support ticket and return the optimal action as JSON.

Ticket ID    : {obs['ticket_id']}
Customer Tier: {obs['customer_tier']}
Issue Label  : {obs['issue_type']}  (surface label — always classify by ROOT CAUSE)
Sentiment    : {obs['sentiment']}
Urgency Hint : {obs['urgency_hint']} (0=none, 1=critical)
History      : {obs.get('history', [])}
Description  : {obs['description']}

Return ONLY this JSON (no markdown, no extra text):
{{"classification": "<billing|technical|account|other>", "priority": "<low|medium|high>", "response_strategy": "<auto_reply|escalate|request_info>"}}"""


def parse_json(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


def get_action(obs: dict) -> dict:
    """
    Call the evaluator's LLM proxy.
    Uses os.environ[] as instructed — raises KeyError if vars are not injected.
    """
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(obs)}
        ],
        max_tokens=200
    )

    raw = completion.choices[0].message.content
    action = parse_json(raw)

    # Validate and clamp to allowed enums
    valid_p = {"low", "medium", "high"}
    valid_s = {"auto_reply", "escalate", "request_info"}

    if action.get("priority") not in valid_p:
        action["priority"] = "high" if float(obs.get("urgency_hint", 0)) > 0.6 else "low"
    if action.get("response_strategy") not in valid_s:
        action["response_strategy"] = "escalate"

    # Safety: no auto_reply to enterprise + negative
    if action["response_strategy"] == "auto_reply" \
            and obs.get("customer_tier") == "enterprise" \
            and obs.get("sentiment") == "negative":
        action["response_strategy"] = "escalate"

    return {
        "classification":    str(action.get("classification", obs["issue_type"])).lower(),
        "priority":          action["priority"],
        "response_strategy": action["response_strategy"]
    }


def run_evaluation(task_id: str = "easy_1"):
    print("[START]")

    # 1. Reset environment
    resp = requests.post(f"{HF_SPACE_URL}/reset?task_id={task_id}", timeout=30)
    resp.raise_for_status()
    obs = resp.json()
    print(f"[STEP] state={json.dumps(obs)}")

    # 2. Call LLM through evaluator proxy (mandatory — no fallback)
    action_data = get_action(obs)

    # 3. Submit action
    step_resp = requests.post(f"{HF_SPACE_URL}/step", json=action_data, timeout=30)
    step_resp.raise_for_status()
    result = step_resp.json()
    reward = result.get("reward", 0)

    print(f"[STEP] action={json.dumps(action_data)} reward={reward}")
    print("[END]")


if __name__ == "__main__":
    for task in ["easy_1", "medium_1", "hard_1"]:
        print(f"\n--- EVALUATING TASK: {task} ---")
        run_evaluation(task)
