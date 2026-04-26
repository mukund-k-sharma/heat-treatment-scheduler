---
title: Heat Treatment Scheduler - Continuous Digital Twin (V2)
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
short_description: A physics-informed continuous reinforcement learning environment for metallurgical process control.
tags:
  - openenv
  - reinforcement-learning
  - metallurgy
  - precipitation-hardening
  - semi-markov decision process (SMDP)
---

# 🏭 Continuous Heat Treatment Digital Twin: Predictive Braking with GRPO

**Hugging Face Space:** [heat-treatment-scheduler](https://huggingface.co/spaces/mukundnjoy/heat-treatment-scheduler)  
**Training Notebook (Colab):** [TRL.ipynb](https://colab.research.google.com/drive/1mdsMleIwfpBrLe2Csb2GTmKqZXQbGjC3?usp=sharing)  
**Presentation Deck:** [Heat Treatment Scheduler - Digital Twin (V2)](https://docs.google.com/presentation/d/1ZHcN1Glm7zaK1rs2FiZDN-AXNNFXhR61vBk14T-eZh8/edit?usp=sharing)

## Overview

Built for the **Meta PyTorch OpenEnv Hackathon Grand Finale** (Theme: *Long-Horizon Planning & Instruction Following*).

Precipitation hardening of aerospace alloys (like Ti-6Al-4V) requires extreme thermal precision to hit a target nanoprecipitate radius (e.g., 22.5 nm). Under-aging leaves the material weak; over-aging or melting destroys the casting.

Standard RL struggles with this because the physical environment is continuous and highly non-linear due to thermal mass lag and surface oxidation insulation. We built a cloud-distributed digital twin that uses **Meta's OpenEnv** to simulate continuous thermodynamics, and trained an LLM via **GRPO** to execute "predictive braking" to hit the exact target radius.

### The Core Challenge

The furnace air temperature changes instantly, but the material's core temperature follows Newton's Law of Cooling — creating massive thermal inertia. A 50 cm × 200 cm titanium casting can take hours to equilibrate. The agent must learn to cut the furnace heat **long before** the material reaches the target temperature, to prevent residual heat from triggering catastrophic Ostwald Ripening (grain coarsening).

### Example Task

> *"Execute a T6 treatment on Titanium Ti-6Al-4V in a massive industrial casting. First, solutionize by holding above 1000°C for 2 hours. Next, rapidly quench below 200°C. Finally, execute an artificial aging phase to grow nanoprecipitates to exactly 22.5 nm."*

The agent must interpret this multi-stage recipe, make micro-adjustments (1 minute) during critical phase transitions, and macro-holds (hours) during steady-state baking — all while predictively managing thermal lag.

### Key Innovations

- **Continuous SMDP**: Unlike standard discrete RL, actions are `[temperature_delta, duration_minutes]` pairs solved via ODE integration — the agent chooses *how long* to hold each furnace state (1 min to 10 hours).
- **Configuration-Driven Physics**: 7 alloys × 3 hardware geometries, dynamically loaded from JSON — zero-code evaluation of new materials.
- **Three Coupled ODEs**: Heat transfer (Newton's Law), oxidation insulation (Arrhenius), and precipitate growth (Arrhenius + phase thresholds) create a realistic feedback loop.
- **Dense Reward Shaping**: Proximity bonuses, energy/time penalties, catastrophic failure penalties (melting: −200, over-coarsening: −100) guide learning without sparse-reward traps.

## Deep Dives

- 🏗️ **[System Architecture](docs/architecture.md):** How we split the OpenEnv Physics Server (Hugging Face) from the Unsloth ML Optimizer (Google Colab). Covers server API, observation/action spaces, configuration system, project structure, data flow, and deployment.
- ⚛️ **[Physics Engine](docs/physics.md):** The continuous ODEs, Arrhenius kinetics, Newton's Law of Cooling, oxidation dynamics, and phase-dependent precipitate growth powering the digital twin. Includes the full reward model and solver integration details.

## Quick Start

### Prerequisites

- Python 3.10+
- OpenEnv framework (`pip install openenv-core`)

### Server + Client

```bash
# Using uv (Recommended)
uv sync
uv run --project . server --port 8000

# Streamlit dashboard
uv run streamlit run ui.py

# Using Docker
docker build -t heat_treatment_scheduler_env:latest .
docker run -p 8000:8000 heat_treatment_scheduler_env:latest
```

### Run the LLM Agent Baseline

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export API_KEY="your_hf_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

This evaluates three tasks — `easy-bake` (Al-2024, lab scale), `medium-bake` (Steel 1095, industrial billet), and `hard-bake` (Ti-6Al-4V, massive casting) — with increasing difficulty.

## Training Results

By step ~150, GRPO optimized the multi-step trajectories, allowing the agent to successfully brake the furnace temperature and park the precipitate radius inside the strict 20.0 nm - 25.0 nm target window.

<!-- ![Reward Convergence]([INSERT_WANDB_REWARD_IMAGE_URL])
*Figure 1: GRPO Reward gradient showing the agent learning to avoid melting penalties.*

![Radius Trajectory]([INSERT_WANDB_RADIUS_IMAGE_URL])
*Figure 2: The physical precipitate radius converging on the 22.5 nm target.*

![Core Temperature]([INSERT_WANDB_TEMP_IMAGE_URL])
*Figure 3: The agent learning to manage thermal lag and oxidation insulation without breaching the 1600°C failure threshold.* -->

> **Training charts** (reward convergence, radius trajectories, and temperature management) are available in the [Presentation Deck](https://docs.google.com/presentation/d/1ZHcN1Glm7zaK1rs2FiZDN-AXNNFXhR61vBk14T-eZh8/edit?usp=sharing).

Key results:

- **Reward Convergence**: The GRPO reward gradient shows the agent learning to avoid melting penalties and converging on positive terminal rewards.
- **Radius Trajectory**: The physical precipitate radius converges on the 22.5 nm target within the 20.0–25.0 nm success window.
- **Temperature Management**: The agent learns to manage thermal lag and oxidation insulation without breaching the 1600°C failure threshold.

## Post-Training & Self-Improvement

The architecture natively supports generating high-quality offline datasets for preference optimization:

1. **Trajectory Generation**: Run frontier LLMs against the physics engine to generate thousands of thermal trajectories.
2. **Preference Pairing**: Using the dense reward model, trajectories are ranked. Failed trajectories (overshooting due to thermal lag) vs. successful trajectories (proper predictive braking) are paired → `post_training/datasets/dpo_dataset.jsonl`.
3. **Self-Improvement**: Apply Direct Preference Optimization (DPO) on `unsloth/Llama-3.2-1B-Instruct` using these paired trajectories → `post_training/train_dpo.ipynb`. Goal: prove a small, self-improved model can internalize continuous differential equations and outperform a zero-shot frontier model.

## License

© Meta Platforms, Inc. and affiliates.  
Licensed under BSD-style license.
