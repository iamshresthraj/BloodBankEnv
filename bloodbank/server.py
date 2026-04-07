from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .environment import BloodBankEnv
from .models import Action

app = FastAPI(title="BloodBank OpenEnv")
envs = {}

class ResetRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    episode_id: str
    action: Action

@app.post("/reset")
def reset(req: ResetRequest):
    env = BloodBankEnv(task_id=req.task_id)
    obs = env.reset()
    envs[env.episode_id] = env
    state = env.state()
    return {"observation": obs.dict(), "state": state.dict()}

@app.post("/step")
def step(req: StepRequest):
    env = envs.get(req.episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
        
    obs, reward, done, info = env.step(req.action)
    state = env.state()
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "state": state.dict()
    }

@app.get("/state/{episode_id}")
def get_state(episode_id: str):
    env = envs.get(episode_id)
    if not env:
        raise HTTPException(status_code=404, detail="Episode not found")
    return env.state().dict()
