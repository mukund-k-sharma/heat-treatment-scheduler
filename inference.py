"""
Heat Treatment Scheduler Inference Script for OpenEnv Hackathon.

This script demonstrates how to run an LLM-based agent on the Heat Treatment Scheduler
environment and report results in the standardized OpenEnv format.

The agent uses an LLM to decide which temperature control action (0-5) to execute at each
step, attempting to grow nanoprecipitates to a target radius while managing thermal process
constraints. Success requires reaching target radius (10-15 nm) without melting (>1100°C)
or over-coarsening (>15 nm).

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
from models import HeatTreatmentSchedulerAction, R_MIN, R_MAX, TEMP_MAX

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", "medium-bake")
BENCHMARK = os.getenv("BENCHMARK", "heat_treatment_scheduler")
MAX_STEPS = 50
TEMPERATURE = 0.0
MAX_TOKENS = 5
SUCCESS_SCORE_THRESHOLD = 0.8  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Metallurgical Process Engineer controlling a precipitation hardening furnace.

    Your objective: Grow nanoprecipitates to the exact target radius through precise temperature control, in the shortest time possible.

    Critical Safety & Quality Constraints:
    - MELTING TRAGEDY: Temperatures >1100°C will melt the alloy, resulting in catastrophic material destruction.
    - OVER-COARSENING: Allowing the radius to exceed 15.0 nm triggers Ostwald ripening, permanently ruining the material's structural integrity.
    - ENERGY EFFICIENCY: Prolonged furnace operation wastes massive amounts of energy. You must hit the target radius and terminate as quickly as safely possible.

    Thermodynamic Operating Regimes:
    - Frozen Zone (<400°C): Zero atomic diffusion. No precipitate growth occurs.
    - Safe Growth Zone (400°C - 750°C): Diffusion-controlled growth. Growth rate scales exponentially with temperature, but naturally saturates and slows down as the radius approaches the target size. 
    - Danger Zone (>750°C): Triggers dangerous Ostwald ripening where large particles aggressively cannibalize small ones, rapidly accelerating past the 15nm failure threshold.

    You MUST respond with ONLY a SINGLE DIGIT (0-5) representing the furnace control action:
        0: Aggressive cooling (-50°C)
        1: Gentle cooling (-10°C)
        2: Hold temperature (0°C - no change)
        3: Gentle heating (+10°C)
        4: Aggressive heating (+50°C)
        5: TERMINATE EPISODE (Press only when Current Radius exactly matches Target Radius)

    Respond with ONLY the digit. Do not include any text, reasoning, or markdown.
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


def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    
    # Denormalize observations to physical units
    current_time = obs.time * 180000 
    current_temp = obs.temperature * TEMP_MAX 
    current_r = obs.radius * R_MAX 
    target_r = obs.target_radius * R_MAX 

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
        Choose your next action (0-5).
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
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
        # Parse for digit 0-5
        match = re.search(r"[0-5]", text)
        return match.group(0) if match else "2"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "2"


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Map task names to environment configurations
    TASK_CONFIG = {
        "easy-bake": {"difficulty": AgentGrade.EASY, "target": 14.0},
        "medium-bake": {"difficulty": AgentGrade.MEDIUM, "target": 12.0},
        "hard-bake": {"difficulty": AgentGrade.HARD, "target": 14.5},
    }
    config = TASK_CONFIG.get(TASK_NAME, TASK_CONFIG["medium-bake"])

    # Instantiate local environment matching the logic of previous working deployments
    env = HeatTreatmentSchedulerEnvironment(
        t=0.0, T=20.0, r=0.0, 
        r_target=config["target"], 
        difficulty=config["difficulty"]
    )

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset() # OpenENV.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(obs, 'done', False):
                break

            message = get_model_message(client, step, obs, last_reward, history)
            action_num = int(message)

            action = HeatTreatmentSchedulerAction(action_num=action_num)
            obs = env.step(action)

            reward = float(getattr(obs, 'reward', 0.0))
            done = bool(getattr(obs, 'done', False))
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate score [0, 1] based on target proximity
        final_temp = obs.temperature * TEMP_MAX  
        final_r = obs.radius * R_MAX  
        
        if final_temp < 1100 and final_r <= 15.0:
            proximity_error = abs(config["target"] - final_r)
            score = max(0.0, 1.0 - (proximity_error / 5.0))
        else:
            score = 0.0  # Failed constraints
            
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()