"""
Heat Treatment Scheduler Inference Script for OpenEnv Hackathon.

This script demonstrates how to run an LLM-based agent on the Heat Treatment Scheduler
environment and report results in the standardized OpenEnv format.

The agent uses an LLM to decide which temperature control action (0-5) and duration to 
execute at each step, attempting to grow nanoprecipitates to a target radius while 
managing continuous thermodynamics and kinetics. Success requires reaching the dynamic 
target radius (loaded from materials.json) without melting the material (T >= T_melt) 
or over-coarsening (r > r_target_max).

Environment Requirements:
    API_BASE_URL         - OpenAI-compatible API endpoint (default: https://api.openai.com/v1)
    HF_TOKEN             - Hugging Face Access Token / API key.
    MODEL_NAME           - LLM identifier (e.g., 'gpt-4o', 'gpt-4', custom LLM)

Submission Requirements:
    - This script must be named `inference.py` and placed in the project root directory
    - Must use OpenAI Client with credentials from environment variables above
    - Must emit exactly three line types to stdout in the specified format (see STDOUT FORMAT below)
    - All LLM API calls must use the environment-configured endpoint and model

Output Tracking:
    The script tracks every step and final outcome through standardized STDOUT logging,
    enabling consistent measurement and comparison across different agent implementations.

STDOUT FORMAT
The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=heat_treatment_scheduler model=<model_name>
    [STEP]  step=<n> action=<action_num> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode completion, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is an error message from the observation, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1]

Example:
    [START] task=medium-bake env=heat_treatment_scheduler model=gpt-4o
    [STEP] step=1 action=3 reward=-1.23 done=false error=null
    [STEP] step=2 action=4 reward=-0.85 done=false error=null
    [STEP] step=3 action=2 reward=198.50 done=true error=null
    [END] success=true steps=3 rewards=-1.23,-0.85,198.50
"""

import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from server.heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment, AgentGrade
from models import HeatTreatmentSchedulerAction

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if API_KEY is None:
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("BENCHMARK", "heat_treatment_scheduler")
MAX_STEPS = 50
TEMPERATURE = 0.0
MAX_TOKENS = 15
SUCCESS_SCORE_THRESHOLD = 0.8 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Metallurgical Process Engineer controlling a precipitation hardening furnace.

    Your objective: Grow nanoprecipitates to the exact target radius without melting the material or triggering Ostwald ripening. 
    You must account for the specific thermal mass of the hardware (which creates lag in heating/cooling) and the oxidation buildup on the alloy surface (which acts as an insulator, reducing heat transfer efficiency over time).

    You must output exactly TWO numbers separated by a comma: [Action_Num, Duration_Minutes]
    
    ACTION_NUM (0-5):
        0: Aggressive cooling (-50°C)
        1: Gentle cooling (-10°C)
        2: Hold furnace temperature (0°C change)
        3: Gentle heating (+10°C)
        4: Aggressive heating (+50°C)
        5: TERMINATE EPISODE (Use only when Current Radius is within the Target Radius bounds)

    DURATION_MINUTES:
        The number of minutes to hold this furnace state (between 1.0 and 600.0).
        Example 1: "4, 120" means aggressive heat for 2 hours.
        Example 2: "2, 15" means maintain current furnace temp for 15 minutes.

    Respond with ONLY the two numbers separated by a comma. No text, no markdown.
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


def build_user_prompt(step: int, obs, last_reward: float, history: List[str], env: HeatTreatmentSchedulerEnvironment) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    # Un-normalize using the dynamically loaded alloy properties
    current_time = obs.time * 180000 
    current_temp = obs.temperature * env.alloy.temp_max 
    current_r = obs.radius * env.alloy.r_max_clip 
    target_r = obs.target_radius * env.alloy.r_max_clip 

    return textwrap.dedent(
        f"""
        Step: {step}/{MAX_STEPS}
        Furnace State:
        - Elapsed Time: {current_time:.0f}s
        - Temperature: {current_temp:.1f}°C
        - Current Radius: {current_r:.3f} nm
        - Target Radius: {target_r:.3f} nm
        
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Choose your next action (0-5) and duration in minutes.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str], env: HeatTreatmentSchedulerEnvironment) -> tuple[int, float]:
    user_prompt = build_user_prompt(step, obs, last_reward, history, env)
    user_prompt = f"Context: Processing {env.alloy.name} in {env.hardware.name}.\n" + user_prompt
    
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
        )
        text = (completion.choices[0].message.content or "").strip()

        # Regex to match "Digit, Number" (e.g., "3, 120" or "4, 45.5")
        match = re.search(r"([0-5])\s*,\s*(\d+(?:\.\d+)?)", text)
        if match:
            action_num = int(match.group(1))
            duration = float(match.group(2))
            return action_num, min(max(duration, 1.0), 600.0) # Clamp duration safely
            
        return 2, 60.0 # Default fallback
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 2, 60.0


def run_single_task(task_name: str, client: OpenAI) -> None:
    # We now map tasks to entirely different physical scenarios!
    TASK_CONFIG = {
        "easy-bake": {"difficulty": AgentGrade.EASY, "alloy": "Al_96_Cu_4", "hardware": "lab_scale"},
        "medium-bake": {"difficulty": AgentGrade.MEDIUM, "alloy": "Fe_99_C_1", "hardware": "industrial_standard"},
        "hard-bake": {"difficulty": AgentGrade.HARD, "alloy": "Ti_6Al_4V", "hardware": "massive_casting"},
    }
    config = TASK_CONFIG.get(task_name, TASK_CONFIG["medium-bake"])

    env = HeatTreatmentSchedulerEnvironment(
        t=0.0, 
        T=20.0, 
        r=0.0, 
        difficulty=config["difficulty"],
        alloy_key=config["alloy"],
        hardware_key=config["hardware"]
    )

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset() 
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(obs, 'done', False):
                break

            action_num, duration = get_model_message(client, step, obs, last_reward, history, env)

            # Pass BOTH action_num and duration_minutes
            action = HeatTreatmentSchedulerAction(
                action_num=action_num,
                duration_minutes=duration
            )
            obs = env.step(action)

            reward = float(getattr(obs, 'reward', 0.0))
            done = bool(getattr(obs, 'done', False))
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Format the action for the logger as "action_num,duration"
            log_step(step=step, action=f"{action_num},{duration:.0f}", reward=reward, done=done, error=error)

            history.append(f"Step {step}: Action {action_num} for {duration:.0f}m -> reward {reward:+.2f}")

            if done:
                break

        final_temp = obs.temperature * env.alloy.temp_max  
        final_r = obs.radius * env.alloy.r_max_clip  
        
        # Check against dynamic alloy limits
        if final_temp < env.alloy.temp_melt and final_r <= env.alloy.r_target_max:
            proximity_error = abs(env.r_target - final_r)
            # Normalize score calculation against the success window size
            window_size = env.alloy.r_target_max - env.alloy.r_target_min
            raw_score = 1.0 - (proximity_error / (window_size * 2))
        else:
            raw_score = 0.0  
            
        score = max(0.01, min(raw_score, 0.99))  
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    tasks = ["easy-bake", "medium-bake", "hard-bake"]
    
    for task in tasks:
        run_single_task(task, client)


if __name__ == "__main__":
    main()