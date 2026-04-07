"""
Heat Treatment Scheduler Inference Script for OpenEnv Hackathon.

This script demonstrates how to run an LLM-based agent on the Heat Treatment Scheduler
environment and report results in the standardized OpenEnv format.

The agent uses an LLM to decide which temperature control action (0-5) to execute at each
step, attempting to grow nanoprecipitates to a target radius while managing thermal process
constraints. Success requires reaching target radius (10-15 nm) without melting (>1100°C)
or over-coarsening (>15 nm).

Environment Requirements:
    LLM_API_BASE_URL     - OpenAI-compatible API endpoint (default: https://api.openai.com/v1)
    API_KEY or HF_TOKEN  - API key for authentication
    OPENAI_API_KEY       - OpenAI API key (alternative to API_KEY)
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
    [STEP]  step=<n> action=<action_idx> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode completion, always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is an error message from the observation, or null if none.
    - All fields on a single line with no newlines within a line.

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
from typing import List
from openai import OpenAI

from server.heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment, AgentGrade
from models import HeatTreatmentSchedulerAction, R_MIN, R_MAX, TEMP_MAX
from logging_config import get_logger

# Module logger
logger = get_logger(__name__)

# Mandatory Configuration from environment variables
LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# Task Configuration
TASK_NAME = os.getenv("TASK_NAME", "medium-bake")
ENV_NAME = "heat_treatment_scheduler"
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0")) 

# Map task names to environment configurations
TASK_CONFIG = {
    "easy-bake": {"difficulty": AgentGrade.EASY, "target": 14.0},
    "medium-bake": {"difficulty": AgentGrade.MEDIUM, "target": 12.0},
    "hard-bake": {"difficulty": AgentGrade.HARD, "target": 14.5},
}

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert Metallurgical Process Engineer controlling a precipitation hardening furnace.
    
    Your objective: Grow nanoprecipitates to the exact target radius through precise temperature control.
    
    Critical constraints to avoid:
    - MELTING: Temperature >1100°C causes catastrophic failure (-200 reward penalty)
    - OVER-COARSENING: Radius >15 nm (ripening failure) causes material degradation (-100 penalty)
    - TIME EFFICIENCY: Each hour of operation incurs ~1.0 reward penalty
    
    Optimal strategy (400-750°C): Controlled growth phase - growth rate increases with temperature
                                   but slows as radius approaches target (saturation effect)
    
    Safe zone: 400-750°C provides diffusion-controlled growth with natural saturation
    Danger zone: >750°C triggers Ostwald ripening (large particles grow at expense of small ones)
    Frozen zone: <400°C has zero precipitate growth (no progress toward goal)
    
    You MUST respond with ONLY a SINGLE DIGIT (0-5) representing the furnace control action:
        0: Aggressive cooling (-50°C)
        1: Gentle cooling (-10°C)
        2: Hold temperature (0°C - no change)
        3: Gentle heating (+10°C)
        4: Aggressive heating (+50°C)
        5: TERMINATE EPISODE (only when current radius matches target radius)
    
    Respond with ONLY the digit, no explanation.
    """
).strip()

def build_user_prompt(step: int, obs) -> str:
    """
    Build LLM prompt with denormalized furnace state.
    
    Converts normalized observations (0-1 range) to physical units so the LLM understands
    actual temperature, time, and precipitate size. This makes the decision-making more
    interpretable and aligned with physical metallurgy concepts.
    """
    # Denormalize observations to physical units for LLM comprehension
    # Observations come normalized: value / maximum
    current_time = obs.time * 180000  # Elapsed time in seconds (normalize factor: 180,000s = 50 hours)
    current_temp = obs.temperature * TEMP_MAX  # Temperature in °C (normalize factor: 1200°C)
    current_r = obs.radius * R_MAX  # Current radius in nm (normalize factor: 15 nm)
    target_r = obs.target_radius * R_MAX  # Target radius in nm
    
    return f"""Step {step}/{MAX_STEPS}

Furnace State (Physical Units):
- Elapsed Time: {current_time:.0f} seconds ({current_time/3600:.1f} hours)
- Oven Temperature: {current_temp:.1f}°C
- Current Precipitate Radius: {current_r:.3f} nm
- Target Radius: {target_r:.3f} nm

Thermal Regime Context:
- <400°C (Frozen): No precipitate growth
- 400-750°C (Growth Sweet Spot): Controlled diffusion-limited growth, natural saturation
- >750°C (Ripening Danger): Grain coarsening, material over-coarsens, brittle
- >1100°C (Melting Catastrophe): Material breakdown

Choose your next action (0-5):"""

def parse_model_action(response_text: str) -> int:
    """
    Extract temperature action index (0-5) from LLM response.
    
    Maps LLM response to discrete action:
        0: Aggressive cooling (-50°C)
        1: Gentle cooling (-10°C)
        2: Hold temperature (default fallback)
        3: Gentle heating (+10°C)
        4: Aggressive heating (+50°C)
        5: Terminate episode
    
    Falls back to action 2 (hold) if parsing fails.
    """
    if not response_text:
        return 2  # Default to Hold if no response
    match = re.search(r"[0-5]", response_text)
    if match:
        return int(match.group(0))
    return 2  # Default to Hold on parse failure

def main() -> None:
    """
    Run heat treatment scheduler episode with LLM-based agent control.
    
    Process:
    1. Initialize OpenAI client with configured endpoint and credentials
    2. Load task configuration (difficulty, target radius)
    3. Create environment with task parameters
    4. Run episode loop: observe -> get LLM action -> execute -> record reward
    5. Emit standardized [START], [STEP], [END] logging for evaluation
    
    Success criteria: Reach target radius (10-15 nm) without exceeding 1100°C or 15 nm radius
    """
    logger.info(f"Starting inference with model: {MODEL_NAME}, task: {TASK_NAME}")
    
    client = OpenAI(base_url=LLM_API_BASE_URL, api_key=API_KEY)
    logger.debug(f"OpenAI client initialized with base_url: {LLM_API_BASE_URL}")
    
    # Load task configuration (maps task names to difficulty and target radius)
    config = TASK_CONFIG.get(TASK_NAME, TASK_CONFIG["medium-bake"])
    logger.info(f"Task config loaded: difficulty={config['difficulty'].name}, target={config['target']} nm")
    
    # Initialize Heat Treatment Scheduler environment
    # Parameters: initial_time, initial_temp, initial_radius, target_radius, difficulty
    env = HeatTreatmentSchedulerEnvironment(
        t=0.0,                      # Start at 0 seconds elapsed
        T=20.0,                     # Start at cold (20°C) - frozen phase
        r=0.0,                      # Start with zero precipitate size
        r_target=config["target"],  # Agent goal: grow to this radius
        difficulty=config["difficulty"]  # Noise level: EASY/MEDIUM/HARD
    )
    logger.info(f"Environment initialized with target radius: {config['target']} nm")
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    # Log episode start with task, environment, and model information
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)
    logger.debug(f"Episode start: task={TASK_NAME}, env={ENV_NAME}, model={MODEL_NAME}")
    
    try:
        obs = env.reset()  # Get initial observation (all zeros/low values)
        logger.debug("Environment reset complete")
        
        # Main episode loop: run for up to MAX_STEPS or until done
        for step in range(1, MAX_STEPS + 1):
            if getattr(obs, 'done', False):
                break
            
            # Build observation context prompt for LLM
            # Denormalizes observations to physical units (°C, nm, seconds)
            user_prompt = build_user_prompt(step, obs)
            
            # Query LLM for temperature control decision
            # LLM takes system prompt (expert guidance), user prompt (current state),
            # and returns a digit 0-5 representing the action
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,  # 0.0 for deterministic, >0 for sampling
                    max_tokens=5,  # Single digit response limited to 5 tokens
                )
                response_text = completion.choices[0].message.content or ""
                logger.debug(f"Step {step}: LLM response received: {response_text}")
            except Exception as exc:
                logger.error(f"Step {step}: LLM API error: {str(exc)}")
                response_text = "2"  # Default to hold temperature on API failure
            
            # Parse LLM response to action index (0-5)
            action_idx = parse_model_action(response_text)
            logger.debug(f"Step {step}: Parsed action index: {action_idx}")
            
            # Execute action in environment
            # Environment simulates physics: temperature change -> precipitate growth
            action = HeatTreatmentSchedulerAction(action_num=action_idx)
            obs = env.step(action)
            
            # Extract step results
            reward = float(getattr(obs, 'reward', 0.0))  # Dense reward from environment
            done = bool(getattr(obs, 'done', False))  # Episode termination flag
            error = "null"  # No error string for valid actions
            
            # Track cumulative rewards and step count
            rewards.append(reward)
            steps_taken = step
            
            logger.debug(f"Step {step}: action={action_idx}, reward={reward:.2f}, done={done}")
            
            # Log step in standardized format: [STEP] step=n action=a reward=r done=d error=e
            print(f"[STEP] step={step} action={action_idx} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)
            
            if done:
                # Determine success: temperature safe, radius in target range
                # Success: T < 1100°C (no melting) AND R_MIN ≤ r ≤ R_MAX (reached target)
                final_temp = obs.temperature * TEMP_MAX  # Denormalize temperature
                final_r = obs.radius * R_MAX  # Denormalize radius
                if final_temp < 1100 and R_MIN <= final_r <= R_MAX:
                    success = True
                logger.info(f"Episode completed at step {step}, final_temp={final_temp:.1f}°C, final_r={final_r:.3f} nm, success={success}")
                break
                
    except Exception as exc:
        # On any exception, mark as failure but still log results
        logger.exception(f"Exception occurred during episode execution: {str(exc)}")
        success = False
    finally:
        # Emit final result log in standardized format
        # [END] success=<bool> steps=<count> rewards=<comma-separated list with 2 decimals>
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} rewards={rewards_str}", flush=True)
        logger.info(f"Episode finished: success={success}, steps={steps_taken}, total_reward={sum(rewards):.2f}")

if __name__ == "__main__":
    main()