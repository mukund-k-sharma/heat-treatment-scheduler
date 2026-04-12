---
title: Heat Treatment Scheduler Environment Server
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
short_description: A physics-informed reinforcement learning environment
tags:
  - openenv
  - reinforcement-learning
  - metallurgy
  - precipitation-hardening
---

# Heat Treatment Scheduler Environment

A physics-informed reinforcement learning environment for optimizing the precipitation hardening process in metal alloys. This OpenEnv environment simulates the controlled growth of nanoprecipitates through intelligent temperature control.

## What is This?

You control an industrial furnace to grow nanoprecipitates to a target size in a metal alloy. The challenge is multi-dimensional. You must reach the target radius without:

- **Melting** the material (T ≥ T_melt) → Catastrophic -200 reward
- **Over-coarsening** the material (r > r_target_max) → -100 reward  
- **Wasting time/energy** (high temperatures and long durations) → Continuous penalty.

Furthermore, you must manage **thermal mass** (sluggish heating/cooling based on the hardware geometry) and **oxidation build-up** (which acts as an insulator, reducing the effective heat transfer over time).

Success (r_target_min ≤ r ≤ r_target_max) → +100 to +200 reward (scaled by proximity)

## Quick Start

### Prerequisites

- Python 3.10+
- OpenEnv framework (`pip install openenv-core`)

### Server + Client Example

```python
from heat_treatment_scheduler import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerEnv

# Connect to running server (must start server first)
env = HeatTreatmentSchedulerEnv(base_url="http://localhost:8000")

result = env.reset()
obs = result.observation

# Run 100-step episode
for step in range(100):
    action = HeatTreatmentSchedulerAction(action_num=3, duration_minutes=60.0)  # Heat +10°C for 1 hour
    result = env.step(action)
    
    print(f"Step {step}: T_norm={obs.temperature:.2f}, r_norm={obs.radius:.2f}, reward={result.reward:.2f}")
    if result.done:
        break

env.close()
```

### Start the Server

**With uv:**

```bash
cd heat_treatment_scheduler
uv sync
uv run --project . server --port 8000
```

**With Docker:**

```bash
docker build -t heat_treatment_scheduler_env:latest .
docker run -p 8000:8000 heat_treatment_scheduler_env:latest
```

### Running the Hackathon Agent

To evaluate the LLM agent against the environment, run the inference script. The script requires access to an OpenAI-compatible endpoint and evaluates all three task difficulties (easy, medium, hard).

```bash
# Set the required environment variables (prioritizes API_KEY for proxy routing)
export API_BASE_URL="[https://router.huggingface.co/v1](https://router.huggingface.co/v1)"
export API_KEY="your_injected_proxy_key"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Run the agent evaluation
python inference.py
```

## How It Works

### The Physics

The environment simulates continuous thermodynamics and kinetics of nanoprecipitate growth in metal alloys. The physical model is broken down into three main components: heat transfer, oxidation kinetics, and precipitate growth.

#### 1. Heat Transfer (Newton's Law of Cooling)

The furnace temperature (`T_furnace`) changes instantaneously when the agent takes an action, but the material's core temperature (`T_material`) follows a continuous differential equation based on Newton's Law of Cooling:

$$ \frac{dT_{material}}{dt} = \frac{h(t) \cdot A_{surface} \cdot (T_{furnace} - T_{material})}{m \cdot C_p} $$

Where:

- `m` (mass) is calculated dynamically from the alloy's density.
- `C_p` is the specific heat capacity loaded from `materials.json`.
- `h(t)` is the effective Heat Transfer Coefficient. This decays over time as surface oxidation builds up at high temperatures, acting as an insulator.

#### 2. Oxidation Kinetics (Arrhenius)

The oxidation factor (`ox`), representing the insulation layer, grows based on the material's core temperature using Arrhenius kinetics:

$$ \frac{d(ox)}{dt} = A_{ox} \cdot \exp\left(-\frac{E_{ox}}{R(T_{material} + 273.15)}\right) \cdot (0.8 - ox) $$

Where:

- `A_ox` and `E_ox` are the pre-exponential factor and activation energy for oxidation, loaded from `materials.json`.
- The term `(0.8 - ox)` acts as a saturation term, capping the insulation effect at 80%. As the oxide layer thickens, its growth slows down.

#### 3. Precipitate Growth (Arrhenius + Phase Thresholds)

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
| **Ripening**| 0.68-1.0 * T_melt | $k(T) \cdot (\frac{r}{R_{MAX}})$ | Grain coarsening (Ostwald ripening). Material becomes brittle and loses mechanical properties. |
| **Melting**| T ≥ T_melt | 0 | Material breaks down. Crystalline structure dissolves. Episode terminates. |

Where **R_MAX** is the maximum target radius of the material. The saturation factor $(1 - r/R_{MAX})$ naturally slows growth in the Controlled Growth phase as the radius approaches the target.

In this environment implementation, the physics constants are loaded dynamically based on the chosen alloy in `materials.json` and hardware setup in `hardware.json`.

### Action Space

The action space consists of two components:

1. **`action_num`** (Discrete, 0-5): The temperature control strategy.

   ```text
   0: Aggressive cooling   (-50°C)
   1: Gentle cooling       (-10°C)
   2: Hold steady          (  0°C)
   3: Gentle heating       (+10°C)
   4: Aggressive heating   (+50°C)
   5: Terminate episode    (End early)
   ```

2. **`duration_minutes`** (Continuous, 1.0 to 600.0): How long to hold this new furnace state. This directly influences the underlying continuous differential equations.

### Observation Space (7 Values, All Normalized [0,1])

| Field | Meaning | Range |
| ------- | --------- | ------- |
| `time` | Elapsed time / TIME_MAX (180,000s) | [0, 1] |
| `temperature` | Current material core temp / alloy.temp_max | [0, 1] |
| `radius` | Current radius / alloy.r_max_clip | [0, 1] |
| `target_radius` | Target radius / alloy.r_max_clip | [0, 1] |
| `radius_error` | (current radius - target radius) / alloy.r_max_clip | [-1, 1] |
| `temperature_phase` | Regime indicator | 0=Frozen, 1=Growth, 2=Ripening |
| `remaining_time` | Time left / TIME_MAX | [0, 1] |

### Reward Function

The environment provides **dense, multi-component rewards** to guide learning:

```python
step_reward = -0.1 × error - 0.01 × error²
temp_penalty = 0.001 × T [°C]
time_penalty = 0.00028 × dt_effective [seconds]
```

| Condition | Reward |
| ----------- | -------- |
| Perfect success (r = target) | +200 -> [terminal success reward = 100 (base) + Gaussian proximity bonus + last-step reward (inclues penalties)] |
| Close to target | +100 to +199 (Gaussian scaled) |
| Over-coarsened (r > 15nm) | -100 |
| Melting (T ≥ 1100°C) | -200 |
| High temperature (T > 1000°C) | extra -0.05 per °C per step |
| Time cost | ~1.0 penalty per 1-hour step |

**Examples:**

- Optimal step: -0.05 (error) - 0.5 (temp) - 1.0 (time) = -1.55
- Perfect ending: +200 (success) - 0.5 - 1.0 = +198.5
- Failed ending: -100 (overcoarse) - 0.8 - 1.0 = -101.8

## Configuration

### Difficulty Levels (Curriculum Learning)

Control environment stochasticity:

```python
from heat_treatment_scheduler.server import AgentGrade

AgentGrade.EASY      # σ_T=2°C, σ_r=1%, σ_t=2%     (Clean baseline)
AgentGrade.MEDIUM    # σ_T=4°C, σ_r=3%, σ_t=5%     (Realistic)
AgentGrade.HARD      # σ_T=7°C, σ_r=5%, σ_t=8%     (Challenging)
```

**Recommended training path**: EASY → MEDIUM → HARD

### Material Configuration (`materials.json`)

The environment supports different metal alloys, each with their own unique physical constants (e.g., melting point, density, specific heat capacity, Arrhenius and oxidation kinetic constants). These are defined in `materials.json`.

Available alloy keys:

- `Al_96_Cu_4` (Aluminum 2024 - Default)
- `Al_98_Cu_2` (Aluminum Al-2wt%Cu)
- `Ti_6Al_4V` (Titanium Grade 5)
- `Mg_AZ31B` (Magnesium AZ31B)
- `Fe_99_C_1` (High-Carbon Steel 1095)
- `inconel_718` (Inconel 718)
- `cantor_equiatomic` (Cantor Alloy)

### Hardware Configuration (`hardware.json`)

The environment simulates different furnace and sample geometries, which affect the thermal mass and heat transfer rates (Newton's Law of Cooling). These setups are defined in `hardware.json`.

Available hardware keys:

- `industrial_standard` (Standard medium-weight billet - Default)
- `lab_scale` (Small sample, low thermal mass)
- `massive_casting` (Huge thermal mass, extremely sluggish response)

### Create an Environment

```python
from heat_treatment_scheduler.server import HeatTreatmentSchedulerEnvironment, AgentGrade

env = HeatTreatmentSchedulerEnvironment(
    t=0.0,             # Initial time (seconds)
    T=350.0,           # Initial temperature (°C)
    r=0.5,             # Initial radius (nm)
    difficulty=AgentGrade.MEDIUM,
    alloy_key="Al_96_Cu_4",            # Loads from materials.json
    hardware_key="industrial_standard" # Loads from hardware.json
)
obs = env.reset()
action_result = env.step(HeatTreatmentSchedulerAction(action_num=3))
```

## Server API

### HTTP Endpoints

**POST /reset** → Reset to initial state

```json
{
  "observation": {
    "time": 0.0, "temperature": 0.29, "radius": 0.033,
    "target_radius": 0.83, "radius_error": -0.8,
    "temperature_phase": 0.0, "remaining_time": 1.0,
    "done": false, "reward": 0.0
  }
}
```

**POST /step** → Execute action

```json
{
  "request": {"action_num": 3},
  "response": {
    "observation": {...},
    "reward": -1.23,
    "done": false
  }
}
```

**GET /state** → Get raw environment state

```json
{
  "episode_id": "uuid",
  "step_count": 5,
  "time": 18000.0,
  "temperature": 500.0,
  "radius": 5.2,
  "target_radius": 12.5
}
```

**WS /ws** → WebSocket for persistent sessions (recommended for many steps)

## Key Physics Insights

### The "Parking Brake" Effect

A neat emergent behavior: In the growth phase, as radius approaches R_max:

$$\frac{dr}{dt} = k(T) \cdot \left(1 - \frac{r}{R_{\max}}\right) \to 0$$

Since we clamp `dr = max(dr, 0)`, the agent can hold the precipitate at exactly the target size (15 nm) with zero growth—effectively "parking" without triggering ripening. This mirrors real-world saturation perfectly.

### Saturation Dynamics

The growth formula naturally slows down as you approach the target, providing implicit guidance to the agent without explicit penalty near the goal.

### Energy Cost

Every degree Celsius (relative to action) adds cost. This encourages using the optimal **400-750°C** window efficiently rather than overshooting to 900°C.

## Constraints and Limits

| Parameter | Min | Max | Unit |
| ----------- | ----- | ----- | ------ |
| Time | 0 | 180,000 | seconds (50 hours max) |
| Temperature | 0 | alloy.temp_max | °C (clipped) |
| Radius (clipped) | 0 | alloy.r_max_clip | nm |
| Target Radius | alloy.r_target_min | alloy.r_target_max | nm |

## Expected Behavior

Trained agents typically learn to manage the continuous physics engine by:

1. **Quick heat-up**: Provide aggressive heating actions with appropriate durations to overcome thermal mass and reach the growth phase.
2. **Growth management**: Maintain material temperature in the sweet spot (0.35 - 0.68 * T_melt).
3. **Oxidation mitigation**: Limit unnecessary time at extremely high temperatures to prevent runaway insulation buildup.
4. **Precision control**: Hold temperature to let saturation naturally slow growth.
5. **Timely exit**: Call the termination action when the radius is within the target bounds.

## Deployment

### Docker Build & Run

```bash
docker build -t heat_treatment_scheduler_env:latest .
docker run -p 8000:8000 heat_treatment_scheduler_env:latest
```

### Push to Hugging Face Spaces

```bash
huggingface-cli login
openenv push  # or: openenv push --private
```

## Architecture

### Client Library

```python
from heat_treatment_scheduler import (
    HeatTreatmentSchedulerEnv,
    HeatTreatmentSchedulerAction,
    HeatTreatmentSchedulerObservation
)
```

- **HeatTreatmentSchedulerEnv**: Client wrapper (EnvClient subclass)
- **Action**: Discrete int 0-5 selecting temperature control
- **Observation**: 7-element tuple normalized to [0, 1]

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

## Implementation Notes

### Physics Simulation

- Arrhenius equation with noise for realism
- Four distinct thermal regimes with different growth laws
- Time steps jittered (±50%) to simulate furnace variability
- Growth rate, temperature, and timing all stochastic

### Normalization

All observations normalized to [0, 1] for stable neural network training.

### Reward Shaping

- Dense step rewards guide each action
- Terminal bonuses/penalties give episode outcome feedback
- Multi-component penalties prevent local optima (ignoring time, energy)

## Troubleshooting

| Issue | Cause | Fix |
| ------- | ------- | ----- |
| Agent stuck below 0.35 * T_melt | Frozen phase feels "safe" | Increase initial temperature or growth reward |
| Quick death (melting) | Overshoot in early training | Anticipate thermal mass lag, don't heat aggressively too long |
| Agent cannot heat up late game | Severe oxidation buildup | Lower the time spent at high temperature |
| No growth | Incorrect phase logic | Check temperature_phase computation |
| Slow convergence | Large state space | Use curriculum learning (EASY → MEDIUM → HARD) |

## Common Questions

**Q: What's the optimal strategy?** \
A: Heat aggressively at first to overcome thermal mass, back off before overshooting T_melt, maintain steady temp in the growth phase, let saturation slow growth naturally, and exit when r = target.

**Q: Can I change the physics?** \
A: Yes! Modify physical constants (A, E, melting points, etc.) in `materials.json` and `hardware.json`. To change the underlying differential equations or thermal regimes, modify `server/heat_treatment_scheduler_environment.py`.

**Q: How do I use this with PyTorch/TensorFlow?** \
A: The client returns `HeatTreatmentSchedulerObservation` (7-float tuple). Convert with:

```python
obs_array = np.array([obs.time, obs.temperature, obs.radius, ...])
```

**Q: What about multi-agent training?** \
A: `SUPPORTS_CONCURRENT_SESSIONS = True` in `HeatTreatmentSchedulerEnvironment` allows multiple simultaneous WebSocket clients.

## References & Further Reading

For detailed physics explanations and code comments, see:

- [server/heat_treatment_scheduler_environment.py](server/heat_treatment_scheduler_environment.py) — Complete physics model with inline comments
- [models.py](models.py) — Type definitions and constant documentation
- [client.py](client.py) — Protocol and communication details
- [inference.py](inference.py) — Example LLM agent integration

## Contributing

When modifying the environment:

1. **Preserve physics**: Keep Arrhenius equation and phase transitions intact
2. **Test boundaries**: Verify behavior at critical temps (400°C, 750°C, 1100°C)
3. **Document changes**: Update comments in affected code sections
4. **Validate reward**: Ensure terminal conditions match outcome descriptions
5. **Check API**: Verify client-side parsing of all observation fields

## License

© Meta Platforms, Inc. and affiliates.  
Licensed under BSD-style license. See [LICENSE](LICENSE) file.

---

**For questions or issues, refer to inline code comments. All formulas and physics are extensively documented in `server/heat_treatment_scheduler_environment.py`.**
