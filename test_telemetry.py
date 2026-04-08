import asyncio
import os
import json
from bloodbank.client import BloodBankEnvClient
from bloodbank.models import Action

# Mocked telemetry from inference.py
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action_obj, obs_dict, reward, total_reward, done, error) -> None:
    done_str = str(done).lower()
    error_str = error if error else "null"
    action_json = json.dumps(action_obj.dict()) if action_obj else "{}"
    print(f"[STEP] step={step} action={action_json} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success, steps, score, rewards) -> None:
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_csv}", flush=True)

async def test_telemetry():
    env_url = "http://localhost:8000"
    client = await BloodBankEnvClient.from_docker_image()
    
    log_start("task_1_easy_basic_fulfillment", "BloodBankEnv", "DummyTestAgent")
    
    res = await client.reset()
    rewards = []
    
    for step in range(1, 4): # just 3 steps
        action = Action(allocations=[]) # Idle action
        res = await client.step(action)
        reward = res.reward
        rewards.append(reward)
        log_step(step, action, res.observation.dict(), reward, sum(rewards), res.done, None)
        if res.done: break
        
    log_end(True, len(rewards), 0.8, rewards)

if __name__ == "__main__":
    asyncio.run(test_telemetry())
