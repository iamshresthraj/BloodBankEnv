import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from bloodbank.client import BloodBankEnvClient
from bloodbank.models import Action, Allocation, BloodType, Request

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# API_BASE_URL points to the LLM router, not the environment!
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "task_3_hard_adaptive_management")
BENCHMARK = os.getenv("BENCHMARK", "BloodBankEnv")

MAX_STEPS = 30
TEMPERATURE = 0.7
MAX_TOKENS = 500
SUCCESS_SCORE_THRESHOLD = 0.6  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI managing an emergency hospital blood bank.
    Each turn you receive observation of inventory (units by blood type), pending emergency requests, and a list of new donations.
    Valid Blood Types: O-, O+, A-, A+, B-, B+, AB-, AB+.
    Your goal is to save lives by matching requests (prioritizing 'emergency' and 'urgent') with compatible blood types.
    Keep waste down by prioritizing older units if appropriate.
    
    You must output ONLY valid JSON format representing the 'allocations'. No markdown, no explanations. 
    Format example:
    {
       "allocations": [
          {"request_id": "REQ_1_abcd", "allocated_units": 2, "prioritize_near_expiry": true}
       ]
    }
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action_obj: Action, obs_dict: dict, reward: float, total_reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "None"
    
    req_map = {}
    if "pending_requests" in obs_dict:
        for req in obs_dict["pending_requests"]:
            if isinstance(req, dict):
                req_map[req.get("request_id")] = req.get("blood_type", "Unknown")
            else:
                req_map[req.request_id] = req.blood_type
            
    allocs = []
    if action_obj and hasattr(action_obj, "allocations"):
        for a in action_obj.allocations:
            btype = req_map.get(a.request_id, "Unknown")
            allocs.append(f"{a.allocated_units} units of {btype} to {a.request_id}")
    
    action_str = " | ".join(allocs) if allocs else "No Allocations"
    
    print(
        f"[STEP {step}] Total Reward: {total_reward:.2f} (Step: {reward:.2f}) | Done: {done} | Action: [{action_str}] | Error: {error_val}",
        flush=True, # Added spaces and improved formatting for better readability
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    total_reward = sum(rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} total_reward={total_reward:.2f}", flush=True)

def build_user_prompt(step: int, obs: dict, last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Observation (Inventory & Requests):
        {json.dumps(obs, indent=2)}
        
        Send your strict JSON Action allocations.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: dict, last_reward: float) -> tuple[Action, str]:
    import time
    user_prompt = build_user_prompt(step, obs, last_reward)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                response_format={"type": "json_object"}
            )
            text = (completion.choices[0].message.content or "").strip()
            parsed_json = json.loads(text)
            action_obj = Action(**parsed_json)
            return action_obj, json.dumps(parsed_json)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "quota" in exc_str.lower() or "exhausted" in exc_str.lower():
                print(f"[DEBUG] Rate limit hit (attempt {attempt+1}/{max_retries}). Sleeping for 60 seconds before retrying...", flush=True)
                time.sleep(60)
                continue
            print(f"[DEBUG] Model generation failed, falling back to dummy: {exc}", flush=True)
            break

    # Fallback to dummy action to prevent crash
    dummy = Action(allocations=[])
    return dummy, json.dumps(dummy.dict())

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await BloodBankEnvClient.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_obs = result.observation.dict()
        last_reward = 0.0
        score = result.score

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_obj, action_str = get_model_action(client, step, last_obs, last_reward)
            error_msg = None

            try:
                result = await env.step(action_obj)
                obs_dict = result.observation.dict()
                reward = result.reward or 0.0
                done = result.done
                score = result.score
            except Exception as e:
                obs_dict = last_obs
                reward = 0.0
                done = True
                error_msg = str(e)
                score = 0.0

            rewards.append(reward)
            steps_taken = step
            last_obs = obs_dict
            last_reward = reward
            total_current_reward = sum(rewards)

            log_step(step=step, action_obj=action_obj, obs_dict=last_obs, reward=reward, total_reward=total_current_reward, done=done, error=error_msg)

            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
