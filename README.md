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

# Heat Treatment Scheduler - Continuous Digital Twin (V2)

A physics-informed reinforcement learning environment designed for the **Meta PyTorch OpenEnv Hackathon Grand Finale**.

This environment evaluates an AI agent's ability to execute **Long-Horizon Planning & Instruction Following**. Operating as a continuous **Semi-Markov Decision Process (SMDP)**, the agent acts as *an industrial Metallurgical Process Controller*, executing complex thermal recipes while predictively managing non-linear thermodynamic constraints like thermal inertia and oxidation insulation.

## The Challenge

Agents must execute multi-stage thermal recipes (e.g., T6 tempering) to grow nanoprecipitates to an exact target radius without:

- **Melting** the material (T ≥ T_melt) → Catastrophic -200 reward
- **Over-coarsening** the material (r > r_target_max) → -100 reward  
- **Wasting time/energy** (high temperatures and long durations) → Continuous penalty.

**The Core Difficulty:** Because of massive thermal inertia (lag), the agent must learn **"Predictive Braking"** — cutting the furnace heat long before the material reaches the target temperature to prevent residual heat from causing catastrophic `Ostwald Ripening`.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- OpenEnv framework (`pip install openenv-core`)

### 2. Server + Client Example

**Using uv (Recommended):**

```bash
cd heat_treatment_scheduler
uv sync
uv run --project . server --port 8000
```

**Running Streamlit dashboard:**

```bash
uv run streamlit run ui.py
```

**Using Docker:**

```bash
docker build -t heat_treatment_scheduler_env:latest .
docker run -p 8000:8000 heat_treatment_scheduler_env:latest
```

### 3. Run the LLM Agent Baseline

To evaluate the LLM agent against the three distinct material/hardware scenarios, configure your proxy and run the inference script:

```bash
# Set the required environment variables (prioritizes API_KEY for proxy routing)
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export API_KEY="your_injected_proxy_key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run the agent evaluation
python inference.py
```

## The Architecture & Physics Engine

Unlike standard discrete reinforcement learning environments, this Digital Twin is powered by a continuous ODE solver (`scipy.integrate.solve_ivp`).

### Server Components

```text
server/
├── app.py                                # FastAPI application
├── heat_treatment_scheduler_environment.py  # Core physics simulation
└── __init__.py                          # Exports
```

**Key classes:**

- `HeatTreatmentSchedulerEnvironment`: Physics engine + step/reset logic
- `AgentGrade`: Enum for difficulty (EASY, MEDIUM, HARD)
- OpenEnv integration: Automatic REST + WebSocket endpoints

### Project Structure

```text
heat_treatment_scheduler/
├── __init__.py
├── client.py              # Client (EnvClient subclass)
├── Dockerfile             # Docker file
├── hardware.json          #
├── materials.json         #
├── models.py              # Pydantic types + constants
├── inference.py           # Example LLM inference agent
├── logging_config.py      # Logging setup
├── openenv.yaml           # OpenEnv metadata
├── pyproject.toml         # Dependencies & build config
├── README.md              # This file
├── LICENSE
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI/OpenEnv server
    ├── heat_treatment_scheduler_environment.py
    ├── Dockerfile
    └── requirements.txt
```

### Difficulty Levels

Control environment stochasticity:

```python
from heat_treatment_scheduler.server import AgentGrade

AgentGrade.EASY      # σ_T=2°C, σ_r=1%, σ_t=2%     (Clean baseline)
AgentGrade.MEDIUM    # σ_T=4°C, σ_r=3%, σ_t=5%     (Realistic)
AgentGrade.HARD      # σ_T=7°C, σ_r=5%, σ_t=8%     (Challenging)
```

### Physics Engine

#### 1. The action Space (SMDP)

The action space is a decoupled **[Action, Duration]** pair, perfectly mirroring how human engineers program industrial machines via Thermal Recipes.

1. `action_num` **(Discrete, 0-5)**: Temperature control (-50°C, -10°C, 0°C, +10°C, +50°C, Terminate).

2. `duration_minutes` **(Continuous, 1.0 to 600.0)**: How long to hold the furnace state.

#### 2. Heat Transfer (Thermal Mass & Lag)

The furnace air temperature (`T_furnace`) changes instantly, but the material's core temperature (`T_material`) follows **Newton's Law of Cooling**:

$$ \frac{dT_{material}}{dt} = \frac{h(t) \cdot A_{surface} \cdot (T_{furnace} - T_{material})}{m \cdot C_p} $$

Where:

- `m` (mass) is calculated dynamically from the alloy's density.
- `C_p` is the specific heat capacity loaded from `materials.json`.
- `h(t)` is the effective Heat Transfer Coefficient. This decays over time as surface oxidation builds up at high temperatures, acting as an insulator.

#### 3. Dynamic Oxidation Kinetics (Arrhenius Insulation)

The effective heat transfer coefficient $h(t)$ decays as surface oxidation builds up. Oxidation is calculated continuously via Arrhenius kinetics:

$$ \frac{d(ox)}{dt} = A_{ox} \cdot \exp\left(-\frac{E_{ox}}{R(T_{material} + 273.15)}\right) \cdot (0.8 - ox) $$

Where:

- `A_ox` and `E_ox` are the pre-exponential factor and activation energy for oxidation, loaded from `materials.json`.
- The term `(0.8 - ox)` acts as a *saturation term*, capping the insulation effect at **80%**. As the oxide layer thickens, its growth slows down.

#### 4. Precipitate Growth (Arrhenius + Phase Thresholds)

The base reaction rate `k(T)` for precipitate growth is driven by the Arrhenius equation:

$$ k(T) = A \cdot \exp\left(-\frac{E}{R(T_{material} + 273.15)}\right) $$

Where:

- `A` is the pre-exponential factor and `E` is the activation energy for precipitate growth (from `materials.json`).
- `R` is the universal gas constant (8.314 J/(mol·K)).

The actual growth rate `dr/dt` is determined by the current thermal regime relative to the alloy's melting temperature (`T_melt`):

| Regime | Temperature Range | Growth Rate (`dr/dt`) | Physics |
| -------- | ------------------- | ----------------------- | --------- |
| **Frozen** | T < 0.35 * T_melt | 0 | Atomic diffusion is negligible. Material remains in initial microstructure. |
| **Growth** | 0.35-0.68 * T_melt | $k(T) \cdot (1 - \frac{r}{R_{MAX}})$ | Diffusion-controlled. Rate-limiting factor: atomic diffusion in the solid. The **SWEET SPOT**. |
| **Ripening** | 0.68-1.0 * T_melt | $k(T) \cdot (\frac{r}{R_{MAX}})$ | Grain coarsening (Ostwald ripening). Material becomes brittle and loses mechanical properties. |
| **Melting** | T ≥ T_melt | 0 | Material breaks down. Crystalline structure dissolves. Episode terminates. |

Where **R_MAX** is the maximum target radius of the material.

**Note**: The saturation factor $(1 - r/R_{MAX})$ acts as emergent **"Parking Brake"**, naturally slowing growth in the Controlled Growth phase as the radius approaches the target, allowing precise holds.

## Configuration-Driven Physics

The environment dynamically loads physics properties, allowing **zero-code** evaluation of entirely new alloys and geometries.

1. `materials.json` **(Intrinsic Properties)** - Defines physical constants ($A$, $E$, Melting Points, Oxidation Rates, Specific Heat).
    - `Al_96_Cu_4` (Aluminum 2024 - Default)
    - `Al_98_Cu_2` (Aluminum Al-2wt%Cu)
    - `Ti_6Al_4V` (Titanium Grade 5)
    - `Mg_AZ31B` (Magnesium AZ31B)
    - `Fe_99_C_1` (High-Carbon Steel 1095)
    - `inconel_718` (Inconel 718)
    - `cantor_equiatomic` (Cantor Alloy)

2. `hardware.json` **(Extrinsic Properties)** : Defines geometry affecting thermal mass and lag.
    - `industrial_standard` (Standard medium-weight billet - Default)
    - `lab_scale` (Small sample, low thermal mass)
    - `massive_casting` (Huge thermal mass, extremely sluggish response)


### Difficulty Levels (Curriculum Learning)

Control environment stochasticity:

```python
from heat_treatment_scheduler.server import AgentGrade

AgentGrade.EASY      # σ_T=2°C, σ_r=1%, σ_t=2%     (Clean baseline)
AgentGrade.MEDIUM    # σ_T=4°C, σ_r=3%, σ_t=5%     (Realistic)
AgentGrade.HARD      # σ_T=7°C, σ_r=5%, σ_t=8%     (Challenging)
```

**Recommended training path**: EASY → MEDIUM → HARD

## Server API & Observations

### Observation Space (7 Values, All Normalized [0,1])

| Field | Meaning | Range |
| ------- | --------- | ------- |
| `time` | Elapsed time / TIME_MAX (180,000s) | [0, 1] |
| `temperature` | Current material core temp / alloy.temp_max | [0, 1] |
| `radius` | Current radius / alloy.r_max_clip | [0, 1] |
| `target_radius` | Target radius / alloy.r_max_clip | [0, 1] |
| `radius_error` | (current radius - target radius) / alloy.r_max_clip | [-1, 1] |
| `temperature_phase` | Regime indicator | [0, 2], 0=Frozen, 1=Growth, 2=Ripening |
| `remaining_time` | Time left / TIME_MAX | [0, 1] |

### HTTP Endpoints

**POST /reset** → Reset to initial state

```json
{
  "observation": {
    "time": 0.0, 
    "temperature": 0.29, 
    "radius": 0.033,
    "target_radius": 0.83, 
    "radius_error": -0.8,
    "temperature_phase": 0.0, 
    "remaining_time": 1.0,
    "done": false, 
    "reward": 0.0
  }
}
```

**POST /step** → Execute action

```json
{
  "request": {
    "action_num": 3,
    "duration_minutes": 120.0
  },
  "response": {
    "observation": {...},
    "reward": 15.42,
    "done": false
  }
}
```

**GET /state** → Get raw environment state

```json
{
  "episode_id": "uuid",
  "step_count": 5,
  "time": 7200.0,
  "temperature": 685.4,
  "radius": 12.1,
  "target_radius": 14.0
}
```

## Evaluation & Post-Training

The dense reward model allows for high-quality trajectory generation. Successful trajectories (demonstrating proper predictive braking) and failed trajectories (overshooting due to thermal lag) can be paired.

This creates the ideal dataset for `Direct Preference Optimization (DPO)` or `Supervised Fine-Tuning (SFT)`, proving that an open-source model can self-improve to internalize continuous physical dynamics.

## References & Further Reading

For detailed physics explanations and code comments, see:

- [server/heat_treatment_scheduler_environment.py](server/heat_treatment_scheduler_environment.py) — Complete physics model with inline comments
- [models.py](models.py) — Type definitions and constant documentation
- [client.py](client.py) — Protocol and communication details
- [inference.py](inference.py) — Example LLM agent integration

## License

© Meta Platforms, Inc. and affiliates.  
Licensed under BSD-style license. See [LICENSE](LICENSE) file.

---

<!-- **For questions or issues, refer to inline code comments. All formulas and physics are extensively documented in `server/heat_treatment_scheduler_environment.py`.** -->
