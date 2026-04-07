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

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

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
    user_prompt = build_user_prompt(step, obs, last_reward)
    
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
        print(f"[DEBUG] Model generation failed, falling back to dummy: {exc}", flush=True)
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

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

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
