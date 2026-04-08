import os
import json
import requests
from openai import OpenAI

# Configuration from Environment Variables
# Updated configs in inference.py
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://athes7755-ai-workflow-optimizer-env.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1") # Some regions still use this
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")


# Initialize OpenAI client (API Compatible)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_evaluation(task_id="easy_1"):
    print("[START]")
    
    # 1. Start / Reset Environment
    try:
        resp = requests.post(f"{HF_SPACE_URL}/reset?task_id={task_id}")
        resp.raise_for_status()
        obs = resp.json()
        print(f"[STEP] state={json.dumps(obs)}")
    except Exception as e:
        print(f"Error during reset: {e}")
        return

    # 2. LLM Reasoning & Action Selection
    system_prompt = "You are a professional support specialist bot. You always return JSON."
    user_prompt = f"""
    Analyze the following support ticket and decide the best action.
    
    Ticket ID: {obs['ticket_id']}
    Customer Tier: {obs['customer_tier']}
    Issue Category: {obs['issue_type']}
    Sentiment: {obs['sentiment']}
    Description: {obs['description']}
    
    Respond in JSON format:
    {{
        "classification": "string",
        "priority": "low" | "medium" | "high",
        "response_strategy": "auto_reply" | "escalate" | "request_info"
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Extraction & Fallback Strategy
        try:
            action_data = json.loads(completion.choices[0].message.content)
            # Basic validation check
            if "priority" not in action_data:
                raise ValueError("Incomplete model response")
        except:
            # Fallback action
            action_data = {
                "classification": obs['issue_type'],
                "priority": "medium",
                "response_strategy": "request_info"
            }
            
        # 3. Environment Step
        step_resp = requests.post(f"{HF_SPACE_URL}/step", json=action_data)
        step_resp.raise_for_status()
        result = step_resp.json()
        
        print(f"[STEP] action={json.dumps(action_data)} reward={result['reward']}")
        print("[END]")
        
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    # Test all mandatory tasks
    for task in ["easy_1", "medium_1", "hard_1"]:
        print(f"\n--- EVALUATING TASK: {task} ---")
        run_evaluation(task)
