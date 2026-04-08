from fastapi import FastAPI, HTTPException, Body, Query
from typing import Optional, Dict, Any
from .models import Observation, Action, Reward
from .env import env

app = FastAPI(
    title="AI Workflow Optimizer Environment",
    description="A production-grade environment for evaluating AI Agent decision making in support workflows."
)

@app.api_route("/reset", methods=["GET", "POST"])
async def reset(
    task_id: Optional[str] = Query(None), 
    body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Resets the environment to a specific task state.
    Supports both GET and POST, and task_id from query param or JSON body.
    """
    try:
        # Determine task_id: query param > body > default
        selected_task_id = task_id
        if not selected_task_id and body and "task_id" in body:
            selected_task_id = body["task_id"]
        
        if not selected_task_id:
            selected_task_id = "easy_1"
            
        obs = env.reset(selected_task_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(action: Action):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {
        "status": "online", 
        "environment": "ai-workflow-optimizer-env",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
