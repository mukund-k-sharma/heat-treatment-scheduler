"""
Trajectory Generator for Direct Preference Optimization (DPO).
Reuses the environment and prompts from inference.py to generate chosen/rejected pairs.
"""

import os
import sys
import json
import re
import textwrap
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# Add the root directory to the path so we can import from the server module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment, AgentGrade
from models import HeatTreatmentSchedulerAction

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# We use a HIGH temperature for generation to force the LLM to explore.
# This ensures we get both spectacular successes (chosen) and terrible failures (rejected).
TEMPERATURE = 0.8  
MAX_STEPS = 20
ROLLOUTS_PER_TASK = 10 # TODO: increase it to 50+ during hackathon


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

    Respond with ONLY the two numbers separated by a comma. No text, no markdown.
    """
).strip()

def build_user_prompt(step: int, obs, last_reward: float, env: HeatTreatmentSchedulerEnvironment) -> str:
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
        
        Last step reward: {last_reward:.2f}
        Choose your next action (0-5) and duration in minutes.
        """
    ).strip()

def run_rollout(client: OpenAI, config: dict, rollout_id: int) -> Tuple[float, List[Dict]]:
    """Runs a single episode and returns the final score and the raw conversation history."""
    env = HeatTreatmentSchedulerEnvironment(
        t=0.0, 
        T=20.0, 
        r=0.0, 
        difficulty=config["difficulty"],
        alloy_key=config["alloy"],
        hardware_key=config["hardware"]
    )
    
    obs = env.reset()
    last_reward = 0.0
    
    # 1. Initialize the conversation array for DPO
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Inject context only on the first step
    context_setup = f"Context: Processing {env.alloy.name} in {env.hardware.name}.\n"
    
    for step in range(1, MAX_STEPS + 1):
        if getattr(obs, 'done', False):
            break
            
        user_msg = build_user_prompt(step, obs, last_reward, env)
        if step == 1:
            user_msg = context_setup + user_msg
            
        # 2. Append User Message
        conversation.append({"role": "user", "content": user_msg})
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=TEMPERATURE,
                max_tokens=15,
            )
            # 3. Capture EXACT raw text from LLM (crucial for training)
            raw_llm_text = (completion.choices[0].message.content or "").strip()
            conversation.append({"role": "assistant", "content": raw_llm_text})
            
            # 4. Parse action for the environment
            match = re.search(r"([0-5])\s*,\s*(\d+(?:\.\d+)?)", raw_llm_text)
            if match:
                action_num = int(match.group(1))
                duration = float(match.group(2))
            else:
                action_num, duration = 2, 60.0 # Fallback
                
            duration = min(max(duration, 1.0), 600.0)
            action = HeatTreatmentSchedulerAction(action_num=action_num, duration_minutes=duration)
            obs = env.step(action)
            last_reward = float(getattr(obs, 'reward', 0.0))
            
        except Exception as e:
            print(f"Rollout API error: {e}")
            break

    # Calculate final proximity score [0 to 1]
    final_temp = obs.temperature * env.alloy.temp_max  
    final_r = obs.radius * env.alloy.r_max_clip  
    
    if final_temp < env.alloy.temp_melt and final_r <= env.alloy.r_target_max:
        proximity_error = abs(env.r_target - final_r)
        window_size = env.alloy.r_target_max - env.alloy.r_target_min
        raw_score = 1.0 - (proximity_error / (window_size * 2))
    else:
        raw_score = 0.0  # Failed (melted or over-coarsened)
        
    score = max(0.01, min(raw_score, 0.99))
    print(f"  -> Rollout {rollout_id} complete. Score: {score:.3f}")
    return score, conversation

def generate_dpo_dataset():
    if not API_KEY:
        raise ValueError("API_KEY or HF_TOKEN environment variable is required")
        
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # We test on the HARD setting to force the LLM to deal with massive thermal lag
    TASK_CONFIG = {"difficulty": AgentGrade.HARD, "alloy": "Ti_6Al_4V", "hardware": "massive_casting"}
    
    print(f"Starting generation: {ROLLOUTS_PER_TASK} rollouts for DPO pairing...")
    trajectories = []
    
    for i in range(ROLLOUTS_PER_TASK):
        score, conv = run_rollout(client, TASK_CONFIG, i + 1)
        trajectories.append({"score": score, "conversation": conv})

    # Sort trajectories by score (Highest score first)
    trajectories.sort(key=lambda x: x["score"], reverse=True)
    
    # Pair the top 30% with the bottom 30%
    num_pairs = max(1, int(ROLLOUTS_PER_TASK * 0.3))
    chosen_group = trajectories[:num_pairs]
    rejected_group = trajectories[-num_pairs:]
    
    dataset = []
    for chosen, rejected in zip(chosen_group, rejected_group):
        # Hugging Face TRL Conversational format:
        # We define the "prompt" as the System + Step 1 User message.
        # "chosen" and "rejected" are the remaining back-and-forth messages.
        
        prompt_msgs = chosen["conversation"][:2] 
        chosen_msgs = chosen["conversation"][2:]
        rejected_msgs = rejected["conversation"][2:]
        
        dataset.append({
            "prompt": prompt_msgs,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"]
        })

    # Ensure datasets directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "datasets"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "datasets", "dpo_dataset.jsonl")
    
    with open(out_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSuccessfully saved {len(dataset)} DPO pairs to {out_path}")

if __name__ == "__main__":
    generate_dpo_dataset()