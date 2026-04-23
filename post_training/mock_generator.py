"""
Synthetic Trajectory Generator for DPO.
Bypasses the LLM API to generate physically realistic 'Chosen' and 'Rejected' trajectories,
allowing us to build the DPO training pipeline locally without API calls.

Generates two scenarios using Fe_99_C_1 (Steel 1095) — the most responsive alloy
for meaningful precipitate growth within the 50-hour time budget:

  A) lab_scale (EASY): Fast thermal response (~10 min time constant).
     Challenge: Precise temperature targeting to stay in Growth zone (490-952°C).
     CHOSEN ramps to 920°C (Growth), REJECTED overshoots to 970°C (Ripening).

  B) massive_casting (HARD): Extreme thermal lag (~2.8 hr time constant).
     Challenge: Predictive braking — cutting furnace heat hours before material
     reaches target, letting residual thermal momentum do the work.
     CHOSEN brakes early, REJECTED keeps heating and overshoots into Ripening.
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

# Full system prompt — must match generate_trajectories.py for consistent DPO training
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

MAX_STEPS = 20


def build_user_prompt(step: int, obs, last_reward: float, env: HeatTreatmentSchedulerEnvironment) -> str:
    current_time = obs.time * 180000 
    current_temp = obs.temperature * env.alloy.temp_max 
    current_r = obs.radius * env.alloy.r_max_clip 
    target_r = obs.target_radius * env.alloy.r_max_clip 
    return f"Step: {step}/{MAX_STEPS}\n- Time: {current_time:.0f}s\n- Temp: {current_temp:.1f}°C\n- Radius: {current_r:.3f} nm\n- Target: {target_r:.3f} nm\nLast reward: {last_reward:.2f}\nAction?"


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

    # Calculate final score (same formula as inference.py and generate_trajectories.py)
    final_temp, final_r = obs.temperature * env.alloy.temp_max, obs.radius * env.alloy.r_max_clip  
    if final_temp < env.alloy.temp_melt and final_r <= env.alloy.r_target_max:
        raw_score = 1.0 - (abs(env.r_target - final_r) / ((env.alloy.r_target_max - env.alloy.r_target_min) * 2))
    else:
        raw_score = 0.0  
        
    return {"score": max(0.01, min(raw_score, 0.99)), "conversation": conversation}


# ======================== SCENARIO DEFINITIONS ========================
#
# Fe_99_C_1 (Steel 1095) physics:
#   T_melt = 1400°C
#   Growth zone:   490°C - 952°C  (0.35 - 0.68 × T_melt)
#   Ripening zone: 952°C - 1400°C (0.68 - 1.0 × T_melt)
#   r_target = 6.5nm (midpoint of 5-8nm window)
#   A = 5000, E = 150000 → meaningful growth rate near 900-950°C
#
# At T=920°C: k ≈ 1.0e-3 nm/s → ~3.6 nm/hr in Growth zone
# At T=970°C: material in Ripening → positive feedback, over-coarsening

SCENARIOS = {
    # Scenario A: Fast thermal response — precise temperature targeting
    # Lab scale time constant ≈ 10 min → material tracks furnace almost instantly
    "lab_fast": {
        "config": {"difficulty": AgentGrade.EASY, "alloy": "Fe_99_C_1", "hardware": "lab_scale"},
        # CHOSEN: Ramp to 920°C (Growth zone, 32°C below Ripening), soak for growth
        "chosen_actions": (
            [(4, 10)] * 18 +    # Ramp furnace: 20 + 18×50 = 920°C (quick 10-min holds)
            [(2, 600),           # Soak at 920°C for 10 hours — precipitates grow in sweet spot
             (5, 1)]             # Terminate in target zone
        ),
        # REJECTED: One extra +50°C pushes into Ripening zone → over-coarsening
        "rejected_actions": (
            [(4, 10)] * 19 +    # Ramp furnace: 20 + 19×50 = 970°C (past 952°C Ripening boundary!)
            [(5, 1)]             # Terminate — material over-coarsened from Ostwald ripening
        ),
    },

    # Scenario B: Extreme thermal lag — predictive braking required
    # Massive casting time constant ≈ 2.8 hours → material lags furnace by hours
    "massive_lag": {
        "config": {"difficulty": AgentGrade.HARD, "alloy": "Fe_99_C_1", "hardware": "massive_casting"},
        # CHOSEN: Ramp hot, then aggressively cool before material overshoots into Ripening
        "chosen_actions": (
            [(4, 60)] * 15 +    # Ramp furnace to 770°C over 15 hours (material lags behind)
            [(4, 600),           # Furnace 820°C, hold 10 hrs — material soaks into Growth zone
             (0, 300),           # PREDICTIVE BRAKING: aggressively cool furnace
             (0, 300),           # More cooling — material still rising from residual heat
             (2, 600),           # Hold — let material stabilize in Growth zone
             (5, 1)]             # Terminate
        ),
        # REJECTED: Keeps heating without braking — residual heat pushes past Ripening
        "rejected_actions": (
            [(4, 60)] * 18 +    # Ramp furnace to 920°C over 18 hours (no braking!)
            [(2, 600),           # Hold — material continues climbing past Ripening boundary
             (5, 1)]             # Terminate — over-coarsened from Ostwald ripening
        ),
    },
}


def create_mock_dataset():
    dataset = []
    
    for scenario_name, scenario in SCENARIOS.items():
        config = scenario["config"]
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name}")
        print(f"  Alloy: {config['alloy']}, Hardware: {config['hardware']}")
        print(f"  Difficulty: {config['difficulty'].name}")
        
        # Generate 5 pairs per scenario (σ_T noise creates variation between runs)
        for i in range(5):
            chosen = generate_synthetic_run(scenario["chosen_actions"], config)
            rejected = generate_synthetic_run(scenario["rejected_actions"], config)
            
            print(f"  Pair {i+1}: chosen_score={chosen['score']:.3f}, rejected_score={rejected['score']:.3f}")
            
            dataset.append({
                "prompt": chosen["conversation"][:2],
                "chosen": chosen["conversation"][2:],
                "rejected": rejected["conversation"][2:],
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
            })

    # Save dataset
    os.makedirs(os.path.join(os.path.dirname(__file__), "datasets"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "datasets", "dpo_dataset.jsonl")
    with open(out_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nGenerated {len(dataset)} DPO pairs to {out_path}")


if __name__ == "__main__":
    create_mock_dataset()