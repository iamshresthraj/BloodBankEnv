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
BENCHMARK = os.getenv("BENCHMARK", "BloodBankEnv")

# All 3 tasks that the validator expects to see graded
ALL_TASKS = [
    "task_1_easy_basic_fulfillment",
    "task_2_medium_expiry_rotation",
    "task_3_hard_adaptive_management",
]

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
          {
             "request_id": "REQ_1_abcd", 
             "allocated_units": 2, 
             "prioritize_near_expiry": true,
             "allocated_blood_type": "O-" 
          }
       ]
    }
    Note: 'allocated_blood_type' is optional, but you can set it if you want to explicitly dispense a compatible universal donor type instead of an exact match.
    """
).strip()


def clamp_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive). Never 0.0, never 1.0."""
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 4)


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
            req_btype = req_map.get(a.request_id, "Unknown")
            alloc_btype = getattr(a, "allocated_blood_type", None) or req_btype
            allocs.append(f"[Req: {a.request_id}, Needs: {req_btype}, Got: {alloc_btype}, Units: {a.allocated_units}]")
    
    action_str = " | ".join(allocs) if allocs else "No Allocations"
    
    print(f"\n========== STEP {step} FULL DATA ==========", flush=True)
    print(f"Observation: {json.dumps(obs_dict, indent=2)}", flush=True)
    if action_obj and hasattr(action_obj, "allocations"):
        print("Action allocations: ", end="", flush=True)
        try:
            print(json.dumps([a.model_dump() if hasattr(a, "model_dump") else a.dict() for a in action_obj.allocations], indent=2), flush=True)
        except Exception:
            print(str(action_obj.allocations), flush=True)
    print("===========================================", flush=True)
    
    # Verbose log (extra context for debugging — validators ignore this)
    print(
        f"  -> Step Reward: {reward:.2f} / {MAX_STEP_REWARD:.2f} | Cumulative: {total_reward:.2f} / {MAX_TOTAL_REWARD:.0f} | Allocations: {action_str}",
        flush=True,
    )

    # === REQUIRED OpenEnv stdout format ===
    done_str = str(done).lower()
    error_str = error_val if error_val != "None" else "null"
    action_json = "{}"
    if action_obj and hasattr(action_obj, "allocations"):
        try:
            action_json = json.dumps(action_obj.model_dump() if hasattr(action_obj, "model_dump") else action_obj.dict())
        except Exception:
            action_json = json.dumps({"allocations": []})
    print(
        f"[STEP] step={step} action={action_json} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )

def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    total_reward = sum(rewards)
    print("\n" + "="*70, flush=True)
    print(f"  BLOODBANKENV - FINAL EVALUATION REPORT ({task})", flush=True)
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
    print(f"  Result       : {'PASS' if success else 'FAIL'}", flush=True)
    print("="*70 + "\n", flush=True)
    rewards_csv = ",".join(f"{r:.2f}" for r in rewards)
    # CRITICAL: Include task= and score= in [END] line so the validator can parse them
    print(f"[END] success={str(success).lower()} task={task} steps={steps} score={score:.4f} rewards={rewards_csv}", flush=True)

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


async def run_single_task(task_name: str, client: OpenAI) -> float:
    """Run a single task episode. Returns the clamped grader score."""
    env = await BloodBankEnvClient.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.01  # default: strictly > 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_name)
        obs_obj = result.observation
        last_obs = obs_obj.dict()
        last_reward = 0.0
        score = clamp_score(result.score)

        if obs_obj.is_live_data:
            print(f"[ENV] Live Data Source: {obs_obj.data_source}", flush=True)
        else:
            print(f"[ENV] Data Source: Synthetic Fallback", flush=True)

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
                score = clamp_score(result.score)
            except Exception as e:
                obs_dict = last_obs
                reward = 0.0
                done = True
                error_msg = str(e)
                score = clamp_score(0.0)

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
        # Ensure score is strictly within (0, 1) one final time before logging
        score = clamp_score(score)
        log_end(task=task_name, success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Determine which tasks to run
    # If TASK_NAME env var is set to a specific single task, still run all 3 for validator compliance
    task_override = os.getenv("TASK_NAME", "")

    if task_override and task_override in ALL_TASKS:
        # Run the specified task first (primary), then the others with fewer steps
        tasks_to_run = [task_override] + [t for t in ALL_TASKS if t != task_override]
    else:
        tasks_to_run = ALL_TASKS

    print(f"\n{'='*70}", flush=True)
    print(f"  BLOODBANKENV - RUNNING {len(tasks_to_run)} TASKS", flush=True)
    print(f"  Tasks: {', '.join(tasks_to_run)}", flush=True)
    print(f"{'='*70}\n", flush=True)

    all_scores = {}
    for task_name in tasks_to_run:
        print(f"\n{'#'*70}", flush=True)
        print(f"  STARTING TASK: {task_name}", flush=True)
        print(f"{'#'*70}\n", flush=True)

        task_score = await run_single_task(task_name, client)
        all_scores[task_name] = task_score

    # Final summary
    print(f"\n{'='*70}", flush=True)
    print(f"  BLOODBANKENV - ALL TASKS COMPLETE", flush=True)
    print(f"{'='*70}", flush=True)
    for t, s in all_scores.items():
        print(f"  {t}: score={s:.4f}", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
