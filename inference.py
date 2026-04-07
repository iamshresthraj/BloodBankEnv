import asyncio
import os
import textwrap
import json
import re
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

MAX_STEPS = 33
MAX_TOTAL_REWARD = 100.0
MAX_STEP_REWARD = MAX_TOTAL_REWARD / MAX_STEPS  # ~3.03
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
            allocs.append(f"[Req ID: {a.request_id}, Blood Type: {btype}, Units: {a.allocated_units}, Prioritize near expiry: {a.prioritize_near_expiry}]")
    
    action_str = " | ".join(allocs) if allocs else "No Allocations"
    
    print(
        f"[STEP {step}] Step Reward: {reward:.2f} / {MAX_STEP_REWARD:.2f} | Cumulative: {total_reward:.2f} / {MAX_TOTAL_REWARD:.0f} | Done: {done} | Allocations: {action_str} | Error: {error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    total_reward = sum(rewards)
    print("\n" + "="*70, flush=True)
    print("  BLOODBANKENV - FINAL EVALUATION REPORT", flush=True)
    print("="*70, flush=True)
    print(f"  {'Step':<8} {'Reward':>10} {'Max':>10} {'% Earned':>10}", flush=True)
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    for i, r in enumerate(rewards, 1):
        pct = (r / MAX_STEP_REWARD) * 100 if MAX_STEP_REWARD > 0 else 0
        print(f"  Step {i:<3} {r:>10.2f} {MAX_STEP_REWARD:>10.2f} {pct:>9.1f}%", flush=True)
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    overall_pct = (total_reward / MAX_TOTAL_REWARD) * 100
    print(f"  {'TOTAL':<8} {total_reward:>10.2f} {MAX_TOTAL_REWARD:>10.0f} {overall_pct:>9.1f}%", flush=True)
    print("="*70, flush=True)
    print(f"  Grader Score : {score:.3f} / 1.000", flush=True)
    print(f"  Total Reward : {total_reward:.2f} / {MAX_TOTAL_REWARD:.0f}", flush=True)
    print(f"  Steps Played : {steps} / {MAX_STEPS}", flush=True)
    print(f"  Result       : {'PASS ✅' if success else 'FAIL ❌'}", flush=True)
    print("="*70 + "\n", flush=True)
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
    import re
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
                stream=False
                # Removed response_format={"type": "json_object"} for broader model compatibility
            )
            text = (completion.choices[0].message.content or "").strip()
            
            # Robust JSON extraction: look for the first '{' and last '}'
            json_match = re.search(r"(\{.*\})", text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            
            parsed_json = json.loads(text)
            action_obj = Action(**parsed_json)
            return action_obj, json.dumps(parsed_json)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "quota" in exc_str.lower() or "exhausted" in exc_str.lower():
                print(f"[DEBUG] Rate limit hit (attempt {attempt+1}/{max_retries}). Sleeping for 60 seconds before retrying...", flush=True)
                time.sleep(60)
                continue
            print(f"[DEBUG] Model generation failed (Attempt {attempt+1}): {exc}", flush=True)
            if attempt == max_retries - 1:
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
            prev_obs = last_obs
            last_obs = obs_dict
            last_reward = reward
            total_current_reward = sum(rewards)

            log_step(step=step, action_obj=action_obj, obs_dict=prev_obs, reward=reward, total_reward=total_current_reward, done=done, error=error_msg)

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
