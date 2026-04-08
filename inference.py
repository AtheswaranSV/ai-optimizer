import os
import json
import requests

# Configuration from Environment Variables
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://athes7755-ai-workflow-optimizer-env.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL")  # Injected by the hackathon evaluator
API_KEY = os.getenv("API_KEY")             # Injected by the hackathon evaluator
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


def get_llm_action(obs, task_id):
    """Call the LLM and return an action dict. Falls back gracefully if LLM is unavailable."""
    # Lazy import so the module loads even without a valid token
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        system_prompt = "You are a professional support specialist bot. You always return valid JSON."
        user_prompt = f"""
Analyze the following support ticket and decide the best action.

Ticket ID: {obs['ticket_id']}
Customer Tier: {obs['customer_tier']}
Issue Category: {obs['issue_type']}
Sentiment: {obs['sentiment']}
Description: {obs['description']}

Respond in JSON format only:
{{
    "classification": "<issue_type>",
    "priority": "low" | "medium" | "high",
    "response_strategy": "auto_reply" | "escalate" | "request_info"
}}
"""

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=200
        )

        action_data = json.loads(completion.choices[0].message.content)

        # Validate required fields
        assert "classification" in action_data
        assert action_data.get("priority") in ["low", "medium", "high"]
        assert action_data.get("response_strategy") in ["auto_reply", "escalate", "request_info"]
        return action_data

    except Exception:
        # Deterministic fallback based on task signals
        priority = "high" if obs.get("urgency_hint", 0) > 0.6 else "medium" if obs.get("urgency_hint", 0) > 0.3 else "low"
        strategy = "escalate" if obs.get("customer_tier") == "enterprise" or obs.get("sentiment") == "negative" else "auto_reply"
        return {
            "classification": obs.get("issue_type", "other"),
            "priority": priority,
            "response_strategy": strategy
        }


def run_evaluation(task_id="easy_1"):
    print("[START]")

    # 1. Reset Environment
    try:
        resp = requests.post(f"{HF_SPACE_URL}/reset?task_id={task_id}", timeout=30)
        resp.raise_for_status()
        obs = resp.json()
    except Exception as e:
        print(f"[ERROR] reset failed: {e}")
        print("[END]")
        return

    print(f"[STEP] state={json.dumps(obs)}")

    # 2. Get Action from LLM (with fallback)
    action_data = get_llm_action(obs, task_id)

    # 3. Submit Action to Environment
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
    # Run all mandatory tasks
    for task in ["easy_1", "medium_1", "hard_1"]:
        print(f"\n--- EVALUATING TASK: {task} ---")
        run_evaluation(task)
