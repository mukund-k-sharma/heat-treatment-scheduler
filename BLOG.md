# Teaching a 1B LLM to Control a Furnace: Predictive Braking with GRPO on a Continuous Physics Digital Twin

> **Meta PyTorch OpenEnv Hackathon — Grand Finale Submission**
> *Theme: Long-Horizon Planning & Instruction Following*

---

## TL;DR

We built a **continuous physics digital twin** of an industrial precipitation hardening furnace and trained a **1.2B parameter LLM** (Llama-3.2-1B-Instruct) to autonomously execute multi-stage thermal recipes using **GRPO (Group Relative Policy Optimization)**. The agent must learn "Predictive Braking" — cutting furnace heat long before the material reaches its target temperature — because thermal inertia means residual heat will keep cooking the alloy for hours after the furnace is shut off.

The environment is powered by three coupled Ordinary Differential Equations (ODEs) solved in real-time via SciPy, not a lookup table or a game engine. The LLM doesn't see pixels or tokens — it reads normalized telemetry and outputs `[action, duration]` pairs that are integrated through continuous physics.

**Links:** [HF Space](https://huggingface.co/spaces/mukundnjoy/heat-treatment-scheduler) · [Training Notebook](https://colab.research.google.com/drive/1mdsMleIwfpBrLe2Csb2GTmKqZXQbGjC3?usp=sharing) · [Presentation Deck](https://docs.google.com/presentation/d/1ZHcN1Glm7zaK1rs2FiZDN-AXNNFXhR61vBk14T-eZh8/edit?usp=sharing)

---

## 1. The Problem: Why LLMs Can't Control Furnaces (Yet)

Precipitation hardening is how aerospace engineers make titanium strong. You heat an alloy to a precise temperature, hold it for hours to grow nanometer-scale precipitates inside the crystal lattice, then cool it down. Get the precipitate radius right (say, exactly 22.5 nm) and you have a jet engine turbine blade. Get it wrong and you have scrap metal.

The challenge sounds simple: heat up, hold, cool down. But the physics makes it deceptively hard:

### Thermal Inertia (The Lag Problem)

When you set a furnace to 800°C, the air temperature jumps instantly. But a 7-ton titanium casting sitting inside? Its core temperature follows Newton's Law of Cooling with a time constant of **~1.7 hours**. That means:

- After 30 minutes at 800°C furnace, the core is still only at ~250°C
- After 2 hours, it's at ~600°C
- It takes ~6 hours to fully equilibrate

This creates a lethal trap: if the agent heats aggressively to grow precipitates quickly, it can't stop the growth once it turns off the furnace. The residual heat continues driving Ostwald Ripening (grain coarsening), overshooting the target and ruining the material.

### The "Predictive Braking" Insight

Just like a freight train can't stop on a dime, a massive casting can't cool instantly. The agent must learn to begin cooling **hours before** the precipitates reach target size, anticipating exactly how much residual growth will occur during the cool-down phase. We call this **Predictive Braking** — the core capability we want the LLM to internalize.

### Why This Is Hard for RL

| Challenge | Why It Breaks Standard RL |
|-----------|--------------------------|
| **Continuous dynamics** | No discrete state transitions — ODEs evolve continuously |
| **Variable time steps** | Actions last 1 minute to 10 hours — not fixed-length |
| **Delayed consequences** | Thermal lag means effects of actions are felt hours later |
| **Multi-phase recipes** | Agent must sequence heat→hold→cool over 50+ hour horizons |
| **Catastrophic failure** | Melting (T ≥ T_melt) is irreversible and ends the episode |

---

## 2. The Architecture: A Cloud-Distributed Digital Twin

We split the system into two services connected over HTTP:

```
┌──────────────────────────────────────────┐                        ┌─────────────────────────────────────────┐
│        ML Policy Optimizer (Client)      │  ── POST /step ──────► │       Physics Engine (Server)           │
│        Google Colab T4 GPU               │                        │       Hugging Face Space (Docker)       │
│                                          │  ◄── JSON Obs + R ──── │                                        │
│  • Llama-3.2-1B (4-bit, Unsloth)         │                        │  • FastAPI + Meta OpenEnv              │
│  • GRPO via TRL                          │                        │  • SciPy ODE solver (solve_ivp)        │
│  • Client-side reward shaping            │                        │  • 7 alloys × 3 hardware configs       │
└──────────────────────────────────────────┘                        └─────────────────────────────────────────┘
```

**Why this split?** The physics engine is pure CPU (ODE integration) and needs to be always-on for concurrent training. The ML client needs a GPU for inference/backprop. Hugging Face Spaces give us a free, persistent CPU server; Google Colab gives us a free T4 GPU.

### The Physics Engine

The server simulates three coupled ODEs integrated via `scipy.integrate.solve_ivp` (RK45):

**1. Heat Transfer** — Newton's Law of Cooling with dynamic insulation:

$$\frac{dT_{material}}{dt} = \frac{h_{eff}(t) \cdot A_{surface} \cdot (T_{furnace} - T_{material})}{m \cdot C_p}$$

**2. Oxidation Kinetics** — Arrhenius-driven surface oxide buildup:

$$\frac{d(ox)}{dt} = A_{ox} \cdot \exp\left(-\frac{E_{ox}}{R \cdot T_K}\right) \cdot (0.8 - ox)$$

The oxide layer acts as an insulator, reducing $h_{eff}$ over time. This creates a feedback loop: higher temperatures accelerate oxidation, which slows heat transfer, which changes the optimal heating strategy.

**3. Precipitate Growth** — Arrhenius kinetics with phase-dependent behavior:

| Phase | Temperature Range | Growth Behavior |
|-------|-------------------|-----------------|
| Frozen | T < 0.35 × T_melt | No growth — atoms can't diffuse |
| Growth (Sweet Spot) | 0.35–0.68 × T_melt | Controlled growth with saturation |
| Ripening (Danger Zone) | 0.68–1.0 × T_melt | Ostwald Ripening — grains coarsen |
| Melting | T ≥ T_melt | Episode terminates (−200 penalty) |

The growth rate in the sweet spot follows:

$$\frac{dr}{dt} = A \cdot \exp\left(-\frac{E}{R \cdot T_K}\right) \cdot \left(1 - \frac{r}{R_{max}}\right)$$

The saturation term $(1 - r/R_{max})$ acts as an emergent "parking brake" — growth naturally slows as the radius approaches the physical ceiling, giving the agent a window for precision control.

### The Action Space (Semi-Markov Decision Process)

Unlike standard RL environments with fixed time steps, our agent outputs **decoupled action-duration pairs**:

- **Action** (discrete, 0–5): Temperature delta (−50°C, −10°C, 0°C, +10°C, +50°C, or Terminate)
- **Duration** (continuous, 1–600 minutes): How long to hold the furnace state

This is a Semi-Markov Decision Process (SMDP) — the agent controls not just *what* to do but *how long* to do it. A micro-adjustment during a critical phase transition might last 1 minute; a steady-state aging hold might last 10 hours.

### Configuration-Driven: 7 Alloys × 3 Hardware Geometries

All physics properties are loaded from JSON at runtime. No code changes needed to evaluate a new alloy:

| Task | Alloy | Hardware | Growth Threshold | Melt Point | Challenge |
|------|-------|----------|-----------------|------------|-----------|
| `easy-bake` | Al-2024 | Lab Scale (tiny) | 176°C | 502°C | Fast response, narrow melt margin |
| `medium-bake` | Steel 1095 | Industrial Billet | 490°C | 1400°C | Moderate lag, wide temp range |
| `hard-bake` | Ti-6Al-4V | Massive Casting | 560°C | 1600°C | Extreme thermal lag (~1.7 hr τ) |

---

## 3. Training: GRPO with Curriculum Learning

### The Setup

- **Model**: `unsloth/Llama-3.2-1B-Instruct` (4-bit quantized via Unsloth)
- **Algorithm**: GRPO (Group Relative Policy Optimization) via HuggingFace TRL
- **Hardware**: Single Tesla T4 (Google Colab free tier)
- **LoRA**: r=16, targeting `q_proj` and `v_proj` (1.7M trainable params / 1.2B total = 0.14%)

The LLM generates a complete thermal recipe as text (one `action, duration` pair per line), which is parsed via regex and executed against the live physics engine. The cumulative reward from the ODE simulation becomes the GRPO training signal.

### The Reward Function

We use a hybrid reward: the server provides per-step physics rewards (proximity bonuses, energy penalties), and the client adds episode-level shaping:

```python
# Server-side (per step): proximity to target, time/energy penalties
# Client-side (end of episode):
#   +80 for entering Growth phase (temp > growth_threshold)
#   -80 for never entering Growth phase
#   +200 × (radius_progress / target_radius) — smooth radius magnet
#   +50/+100 milestone bonuses for entering/hitting the target window
#   Per-step temperature progress breadcrumbs (+0.3/°C toward growth)
```

### Bug Discovery: Three Physics Bugs That Prevented Training

During training, we observed `physics/radius_nm` flatlined at 0.0 nm across all runs. A deep audit of the server-side physics revealed three critical bugs:

#### Bug 1: Frozen Kinetics (materials.json)

The Arrhenius pre-exponential factor `A` was set too low for every alloy. For Ti-6Al-4V (`A=500`), the maximum growth rate at optimal temperature was **0.005 nm/hr** — requiring **4,644 hours** (193 days!) to reach the 22.5 nm target. The 50-hour TIME_MAX made success mathematically impossible.

**Fix**: Calibrated `A` values for each alloy to produce ~3 nm/hr at optimal temperature. Ti-6Al-4V: 500 → 620,000. Al-2024: 1,000 → 1,000,000,000.

#### Bug 2: Shrinking Precipitate ODE

The growth saturation term divided by `r_target_max` (25 nm) instead of `r_max_clip` (50 nm). This caused:
- Growth rate → 0 at exactly 25 nm (the target window boundary)
- Growth rate → **negative** above 25 nm (precipitates physically shrinking!)

**Fix**: Changed `(1 - r/r_target_max)` to `(1 - r/r_max_clip)`.

#### Bug 3: Missing ODE Melting Event

If the material melted at minute 5 of a 60-minute step, the ODE solver continued integrating for 55 more minutes of physically meaningless post-melt simulation.

**Fix**: Added a terminal event function to `solve_ivp` that halts integration the instant `T_material` crosses `T_melt`.

### Curriculum Learning Strategy

After fixing the physics bugs, we implemented curriculum learning to handle the exploration difficulty:

| Phase | Task | Why |
|-------|------|-----|
| **Phase 1** | `easy-bake` (Al-2024, lab scale) | Growth at 176°C, near-instant thermal response. Model learns basic recipe pattern. |
| **Phase 2** | `medium-bake` (Steel 1095, industrial) | Growth at 490°C, moderate lag. Model learns to hold longer. |
| **Phase 3** | `hard-bake` (Ti-6Al-4V, massive casting) | Growth at 560°C, 1.7-hour time constant. Model learns Predictive Braking. |

The key insight: a 1B model can't discover a 20-step heating + 10-hour hold strategy through random exploration on the hardest task. It needs to learn the basic "heat → hold → grow → cool" pattern on an easy task first, then transfer.

---

## 4. Reward Hacking: What the Agent Tried

GRPO training surfaced fascinating reward hacking behaviors:

### Hack #1: "Do Nothing" (V1)
The agent discovered that staying cold (Frozen phase, ~20°C) avoided all catastrophic penalties. With the V1 reward function, this yielded a reward of ~0, which was better than risking −200 (melting).

**Counter**: Added a phase-gate penalty — never entering Growth phase costs −80.

### Hack #2: "Warm But Not Hot" (V2)
After the phase-gate fix, the agent heated to ~200°C (collecting temperature progress bonuses) but stopped before reaching the Growth threshold (176°C for Al-2024). It discovered the sweet spot where heating bonuses outweighed step penalties, without any risk.

**Counter**: Made per-step temperature bonuses weaker below the growth threshold (+0.3/°C vs the previous +0.2/°C flat), with the real payoff locked behind the phase-gate bonus.

### Hack #3: "Prompt Mismatch" (V3)
The most subtle hack wasn't even intentional: the system prompt still referenced Ti-6Al-4V (growth at 600°C, target 22.5 nm) while the reward function targeted Al-2024 (growth at 176°C, target 12.5 nm). The model faithfully followed the prompt's instructions — trying to heat to 600°C — but the Al-2024 environment melted at 502°C.

**Counter**: Aligned the system prompt with the curriculum task. Always change both together.

---

## 5. What We Learned

### On Environment Design for LLM-RL

1. **Verify your physics can reach the target.** We spent hours debugging reward functions when the actual bug was `A=500` making success mathematically impossible. Always run a "can a perfect oracle solve this?" sanity check before training.

2. **Reward shaping is an arms race.** Every reward bonus you add creates a new surface for the agent to exploit. The phase-gate bonus fixed one hack but enabled another. Design rewards that are *necessary* (can't be avoided) rather than *sufficient* (can be collected without progress).

3. **Prompt-reward alignment is critical.** In LLM-RL, the system prompt is part of the policy. If the prompt says "target 22.5 nm" but the reward optimizes for 12.5 nm, the model is getting contradictory gradients. Always update prompts and rewards atomically.

### On Continuous Physics Environments

4. **SMDP action spaces are underexplored.** Letting the agent choose *duration* alongside *action* dramatically changes the problem structure. A 1-minute micro-adjustment and a 10-hour hold are fundamentally different decisions, yet they're the same action with different duration parameters.

5. **Thermal lag creates natural curriculum.** The same physics engine produces trivially easy (lab scale, instant response) to extremely hard (massive casting, 1.7-hour lag) problems just by changing the hardware configuration. This is free curriculum learning built into the domain.

6. **ODE solvers need event hooks.** Without terminal events, `solve_ivp` will happily integrate past physically meaningful boundaries (melting, time limits). Always use the `events` parameter.

---

## 6. Technical Details

### Project Structure

```
heat-treatment-scheduler/
├── server/
│   ├── app.py                              # FastAPI + OpenEnv server
│   └── heat_treatment_scheduler_environment.py  # Core ODE physics engine
├── materials.json                          # 7 alloy configurations (Arrhenius constants)
├── hardware.json                           # 3 hardware geometries (thermal mass)
├── models.py                               # Pydantic data models + physics constants
├── inference.py                            # LLM agent baseline (multi-task evaluation)
├── ui.py                                   # Streamlit interactive dashboard
├── post_training/colab/TRL.ipynb           # GRPO training notebook
├── docs/
│   ├── architecture.md                     # Full system architecture documentation
│   └── physics.md                          # ODE equations + reward model details
├── openenv.yaml                            # OpenEnv metadata specification
├── Dockerfile                              # HF Space deployment
└── BLOG.md                                 # This file
```

### Observation Space (7 normalized values)

| Field | Meaning | Range |
|-------|---------|-------|
| `time` | Elapsed / TIME_MAX (50 hrs) | [0, 1] |
| `temperature` | Core temp / T_max | [0, 1] |
| `radius` | Current radius / R_max_clip | [0, 1] |
| `target_radius` | Target / R_max_clip | [0, 1] |
| `radius_error` | (current − target) / R_max_clip | [−1, 1] |
| `temperature_phase` | Regime indicator | 0=Frozen, 1=Growth, 2=Ripening |
| `remaining_time` | Time left / TIME_MAX | [0, 1] |

### Reproducibility

All code is open source. To reproduce:

1. **Deploy the physics server**: Push to a Hugging Face Space with Docker SDK
2. **Run training**: Open `post_training/colab/TRL.ipynb` in Google Colab (T4 GPU)
3. **Monitor**: Training logs stream to Weights & Biases

---

## Acknowledgments

Built for the **Meta PyTorch OpenEnv Hackathon Grand Finale** organized by Hugging Face. The environment uses Meta's [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework for standardized agent-environment interaction.

- **Model**: [unsloth/Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) (Unsloth 4-bit quantization)
- **Training**: [TRL](https://github.com/huggingface/trl) GRPO implementation
- **Physics**: [SciPy](https://scipy.org/) `solve_ivp` ODE integration

---

*© 2026 — Meta PyTorch OpenEnv Hackathon Submission*
