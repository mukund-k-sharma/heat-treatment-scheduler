"""
Synthetic Trajectory Generator for DPO.
Bypasses the LLM API to generate mathematically accurate 'Chosen' and 'Rejected' runs 
using hardcoded physical recipes, allowing us to build the PyTorch pipeline locally.
"""

import os
import json
import sys
import textwrap
from typing import List, Dict

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment, AgentGrade
from models import HeatTreatmentSchedulerAction

SYSTEM_PROMPT = "You are an expert Metallurgical Process Engineer controlling a precipitation hardening furnace..."

def build_user_prompt(step: int, obs, last_reward: float, env: HeatTreatmentSchedulerEnvironment) -> str:
    current_time = obs.time * 180000 
    current_temp = obs.temperature * env.alloy.temp_max 
    current_r = obs.radius * env.alloy.r_max_clip 
    target_r = obs.target_radius * env.alloy.r_max_clip 
    return f"Step: {step}/20\n- Time: {current_time:.0f}s\n- Temp: {current_temp:.1f}°C\n- Radius: {current_r:.3f} nm\n- Target: {target_r:.3f} nm\nLast reward: {last_reward:.2f}\nAction?"

def generate_synthetic_run(action_sequence: List[tuple], config: dict) -> dict:
    """Runs a hardcoded sequence of actions through the environment to generate a conversation history."""
    env = HeatTreatmentSchedulerEnvironment(
        t=0.0, T=20.0, r=0.0, 
        difficulty=config["difficulty"],
        alloy_key=config["alloy"],
        hardware_key=config["hardware"]
    )
    
    obs = env.reset()
    last_reward = 0.0
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_setup = f"Context: Processing {env.alloy.name} in {env.hardware.name}.\n"
    
    for step, (action_num, duration) in enumerate(action_sequence, 1):
        if getattr(obs, 'done', False):
            break
            
        user_msg = build_user_prompt(step, obs, last_reward, env)
        if step == 1: user_msg = context_setup + user_msg
            
        conversation.append({"role": "user", "content": user_msg})
        
        # Simulate LLM response
        llm_text = f"{action_num}, {duration}"
        conversation.append({"role": "assistant", "content": llm_text})
        
        # Step environment
        action = HeatTreatmentSchedulerAction(action_num=action_num, duration_minutes=duration)
        obs = env.step(action)
        last_reward = float(getattr(obs, 'reward', 0.0))

    # Calculate final score
    final_temp, final_r = obs.temperature * env.alloy.temp_max, obs.radius * env.alloy.r_max_clip  
    if final_temp < env.alloy.temp_melt and final_r <= env.alloy.r_target_max:
        raw_score = 1.0 - (abs(env.r_target - final_r) / (env.alloy.r_target_max - env.alloy.r_target_min * 2))
    else:
        raw_score = 0.0  
        
    return {"score": max(0.01, min(raw_score, 0.99)), "conversation": conversation}

def create_mock_dataset():
    config = {"difficulty": AgentGrade.HARD, "alloy": "Ti_6Al_4V", "hardware": "massive_casting"}
    
    # CHOSEN: Smart predictive braking (Heat fast, then immediately aggressively cool before it overshoots)
    chosen_actions = [(4, 60.0), (4, 60.0), (0, 60.0), (2, 60.0), (5, 1.0)]
    
    # REJECTED: Ignored thermal lag (Heated way too long, melted or triggered Ostwald ripening)
    rejected_actions = [(4, 120.0), (4, 120.0), (4, 120.0), (5, 1.0)]
    
    dataset = []
    # Generate 10 identical pairs just so our PyTorch dataloader has enough rows to test
    for _ in range(10):
        chosen = generate_synthetic_run(chosen_actions, config)
        rejected = generate_synthetic_run(rejected_actions, config)
        
        dataset.append({
            "prompt": chosen["conversation"][:2],
            "chosen": chosen["conversation"][2:],
            "rejected": rejected["conversation"][2:],
            "chosen_score": chosen["score"],
            "rejected_score": rejected["score"]
        })

    os.makedirs(os.path.join(os.path.dirname(__file__), "datasets"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "datasets", "dpo_dataset.jsonl")
    with open(out_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"✅ Generated 10 synthetic DPO pairs to {out_path}")

if __name__ == "__main__":
    create_mock_dataset()