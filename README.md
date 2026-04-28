---
title: Heat Treatment Scheduler - Continuous Digital Twin (V2)
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
short_description: Physics-informed RL environment for heat treatment.
tags:
  - openenv
  - reinforcement-learning
  - metallurgy
  - precipitation-hardening
  - semi-markov decision process (SMDP)
---

# 🏭 Continuous Heat Treatment Digital Twin: Predictive Braking with GRPO

**Hugging Face Space:** [heat-treatment-scheduler](https://huggingface.co/spaces/mukundnjoy/heat-treatment-scheduler)  
**Training Notebook (HF Space):** [TRL-heat-treatment-scheduler](https://huggingface.co/spaces/mukundnjoy/TRL-heat-treatment-scheduler)  
**Presentation Deck:** [Heat Treatment Scheduler - Digital Twin (V2)](https://docs.google.com/presentation/d/1ZHcN1Glm7zaK1rs2FiZDN-AXNNFXhR61vBk14T-eZh8/edit?usp=sharing)  
**WandB Training Dashboard:** [heat-treatment-grpo](https://wandb.ai/mukundnjoy-paypal/heat-treatment-grpo?nw=nwusermukundnjoy)  
**Technical Blog:** [BLOG.md](BLOG.md)

## Overview

Built for the **Meta PyTorch OpenEnv Hackathon Grand Finale** (Theme: *Long-Horizon Planning & Instruction Following*).

Precipitation hardening of aerospace alloys (like Ti-6Al-4V) requires extreme thermal precision to hit a target nanoprecipitate radius (e.g., 22.5 nm). Under-aging leaves the material weak; over-aging or melting destroys the casting.

Standard RL struggles with this because the physical environment is continuous and highly non-linear due to thermal mass lag and surface oxidation insulation. I built a cloud-distributed digital twin that uses **Meta's OpenEnv** to simulate continuous thermodynamics, and trained an LLM via **GRPO** to execute "predictive braking" to hit the exact target radius.

### The Core Challenge

The furnace air temperature changes instantly, but the material's core temperature follows Newton's Law of Cooling — creating massive thermal inertia. A 50 cm × 200 cm titanium casting can take hours to equilibrate. The agent must learn to cut the furnace heat **long before** the material reaches the target temperature, to prevent residual heat from triggering catastrophic Ostwald Ripening (grain coarsening).

### Key Innovations

- **Continuous SMDP**: Unlike standard discrete RL, actions are `[temperature_delta, duration_minutes]` pairs solved via ODE integration — the agent chooses *how long* to hold each furnace state (1 min to 10 hours).
- **WebSocket Stateful Sessions**: Uses OpenEnv's WebSocket protocol for persistent, stateful episode sessions (HTTP endpoints are stateless by design).
- **Configuration-Driven Physics**: 7 alloys × 3 hardware geometries, dynamically loaded from JSON — zero-code evaluation of new materials.
- **Three Coupled ODEs**: Heat transfer (Newton's Law), oxidation insulation (Arrhenius), and precipitate growth (Arrhenius + phase thresholds) create a realistic feedback loop.
- **Dense Reward Shaping**: Proximity bonuses, energy/time penalties, catastrophic failure penalties (melting: −200, over-coarsening: −100), all clamped to ±500 for gradient stability.

## Architecture

```text
┌──────────────────────────────────────────┐                        ┌─────────────────────────────────────────┐
│        ML Policy Optimizer (Client)      │  ── WSS /ws ─────────► │       Physics Engine (Server)           │
│        HF Space (GPU Notebook)           │                        │       Hugging Face Space (Docker)       │
│                                          │  ◄── JSON Obs + R ──── │                                        │
│  • Llama-3.2-1B (4-bit, Unsloth)         │                        │  • FastAPI + Meta OpenEnv              │
│  • GRPO via TRL                          │                        │  • SciPy ODE solver (solve_ivp)        │
│  • Client-side reward shaping            │                        │  • 7 alloys × 3 hardware configs       │
│  • WebSocket reward function (V5)        │                        │  • Task routing (easy/medium/hard)     │
└──────────────────────────────────────────┘                        └─────────────────────────────────────────┘
```

## Deep Dives

- 🏗️ **[System Architecture](docs/architecture.md):** OpenEnv Physics Server (Hugging Face) + Unsloth ML Optimizer (HF Space GPU). Covers server API, observation/action spaces, configuration system, project structure, and deployment.
- ⚛️ **[Physics Engine](docs/physics.md):** The continuous ODEs, Arrhenius kinetics, Newton's Law of Cooling, oxidation dynamics, and phase-dependent precipitate growth. Includes the reward model and solver integration.
- 📝 **[Blog / Technical Writeup](BLOG.md):** Full narrative of the project, bug discovery journey, reward hacking analysis, and lessons learned.

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

![GRPO Training Dashboard showing reward, radius, temperature, and growth phase metrics across 7200 training steps](docs/plots/training_dashboard.png)
*GRPO Training Dashboard (7200 steps on `easy-bake`, Al-2024). Top-left: reward climbs from −50 to +400. Top-right: radius transitions from bimodal (0 or 30 nm) to concentrated 20–28 nm range. Bottom-left: peak temperature stabilizes in the growth zone (200–400°C). Bottom-right: growth phase entry rate rises from 0% to ~90%.*

Training on `easy-bake` (Al-2024, lab scale) with GRPO showed progressive learning after resolving critical physics and architectural bugs:

| Training Phase | Steps | Observation |
|----------------|-------|-------------|
| **V1-V4 (HTTP, stateless)** | 0-200+ | Zero learning. Stateless HTTP meant no state persistence — radius flatlined at 0.0 nm. |
| **V5 (WebSocket, stateful)** | 0-10 | First successful growth phase entry. Temperature reached 277°C, radius grew to 2.82 nm. |
| **Phase mastery** | 10-50 | `entered_growth` → 1.0 consistently. Model learned to heat past 176°C threshold. |
| **Exploration** | 50-330+ | Varied recipes (3-50 steps). Radius oscillating 0-30 nm. Active temperature exploration (100-500°C). |

### Reward Curve

![Reward climbing from −50 to +400 over 7200 training steps, with individual per-step rewards shown as scatter points](docs/plots/reward_curve.png)
*GRPO reward over 7200 steps. The smoothed curve (EMA-30) shows clear upward trend from negative rewards to +400, indicating the agent progressively learns to control the furnace.*

### Growth Phase Discovery

![Growth phase entry rate rising from 0% to 90%+ over training, with baseline rate of 25% marked](docs/plots/growth_phase_entry.png)
*Growth phase entry rate (rolling-50 average). The agent learns to heat past the 176°C growth threshold, rising from 0% at initialization to consistently >85%. The baseline few-shot model achieves only ~25% (red dotted line).*

Key findings at 7200 steps:

- **Growth phase mastery**: The model consistently heats past 176°C (Al-2024 growth threshold), entry rate rising from 0% to ~90%
- **Reward convergence**: Mean reward climbs from −50 to +400 over 7200 steps
- **Predictive Braking**: Still in exploration — the model hasn't yet learned to park the radius at exactly 12.5 nm (overshoots to 20–28 nm range)
- **Reward stability**: All rewards bounded ±500 after ODE blowup fix

See [BLOG.md](BLOG.md) for the full bug discovery journey (6 bugs found and fixed) and reward hacking analysis.

## Future Work

- **Predictive Braking convergence**: Continue GRPO training to solve the radius precision problem (parking at 12.5 nm instead of overshooting to 30 nm). May require reducing the Arrhenius growth rate `A` to widen the control window.
- **Curriculum scaling**: Progress from `easy-bake` → `medium-bake` → `hard-bake` once Phase 1 converges.
- **Offline preference data**: Use the physics engine to generate trajectory pairs (successful vs failed) for preference optimization (DPO/KTO) as a complementary training signal.

## License

© Meta Platforms, Inc. and affiliates.  
Licensed under BSD-style license.
