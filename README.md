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

You control an industrial furnace to grow nanoprecipitates to a target size in a metal alloy. The challenge: reach the target radius without:

- **Melting** the material (T ≥ 1100°C) → -200 reward
- **Over-coarsening** the material (r > 15 nm) → -100 reward  
- **Wasting time/energy** (high temperature) → continuous penalty

Success (10 nm ≤ r ≤ 15 nm) → +100 to +200 reward

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
    action = HeatTreatmentSchedulerAction(action_num=3)  # Heat +10°C
    result = env.step(action)
    
    print(f"Step {step}: T={obs.temperature*1200:.0f}°C, r={obs.radius*15:.2f}nm, reward={result.reward:.2f}")
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

The environment simulates four thermal regimes controlling precipitate growth:

| Regime | Temperature | Growth Rate | Physics | Strategy |
|--------|-------------|-------------|---------|----------|
| **Frozen** | T < 400°C | dr/dt = 0 | No diffusion | Waste of energy, skip it |
| **Growth** | 400-750°C | dr/dt = k(T)·(1-r/R_MAX) | Diffusion-controlled, saturation | **SWEET SPOT** - where you win |
| **Ripening** | 750-1100°C | dr/dt = k(T)·(r/R_MAX) | Grain coarsening, positive feedback | Failure zone - avoid! |
| **Melting** | >1100°C | — | Material destruction | Game over |

Where **R_MAX = 15 nm** (maximum radius). The saturation factor (1 - r/R_MAX) naturally slows growth in the Growth phase as the radius approaches the target.

**Arrhenius Equation** (how growth rate changes with temperature):
$$k(T) = A \times \exp\left(-\frac{E}{R(T+273.15)}\right)$$

Where:

- `k(T)` is the temperature-dependent growth-rate constant
- `A` is the pre-exponential factor
- `E` is the activation energy
- `R` is the universal gas constant
- `T` is the furnace temperature in `°C`
- `273.15` converts Celsius to Kelvin

In this environment implementation, the code uses:

- `R = 8.314 J/(mol·K)`
- `A = 1000 s^-1`
- `E = 120 kJ/mol`
- `T` in `°C`

These are model constants for the benchmark, **not universal material constants**.

For real systems:

- **R** is fixed at **8.314 J/(mol·K)**
- **A** is material- and mechanism-dependent, and often falls in the range **10^7 to 10^13 s^-1**
- **E** is material- and process-dependent, and often falls in the range **50 to 300 kJ/mol**

### Action Space (6 Discrete Actions)

```text
0: Aggressive cooling   (-50°C)
1: Gentle cooling       (-10°C)
2: Hold steady          ( 0°C)
3: Gentle heating       (+10°C)
4: Aggressive heating   (+50°C)
5: Terminate episode
```

**Note**: Action 5 is a terminal action and does not map to temperature change.

### Observation Space (7 Values, All Normalized [0,1])

| Field | Meaning | Range |
| ------- | --------- | ------- |
| `time` | Elapsed time / (TIME_MAX=180,000s) | [0, 1] |
| `temperature` | Current Oven temp / (TEMP_MAX=1200°C) | [0, 1] |
| `radius` | Current radius/ (RADIUS_MAX=15 nm) | [0, 1] |
| `target_radius` | Target radius / (RADIUS_MAX = 15 nm) | [0, 1] |
| `radius_error` | (current radius - target radius) / (RADIUS_MAX=15 nm) | [-1, 1] |
| `temperature_phase` | Regime indicator | 0=Frozen, 1=Growth, 2=Ripening |
| `remaining_time` | Time left / (TIME_MAX=180,000s) | [0, 1] |

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

### Create an Environment

```python
from heat_treatment_scheduler.server import HeatTreatmentSchedulerEnvironment, AgentGrade

env = HeatTreatmentSchedulerEnvironment(
    t=0.0,             # Initial time (seconds)
    T=350.0,           # Initial temperature (°C)
    r=0.5,             # Initial radius (nm)
    r_target=12.5,     # Target radius (nm). Must be 10-15.
    difficulty=AgentGrade.MEDIUM
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
| Temperature | 0 | 1,200 | °C (clipped) |
| Radius (clipped) | 0 | 30 | nm |
| Target Radius (must set) | 10 | 15 | nm |

## Expected Behavior

Trained agents typically learn:

1. **Quick heat-up** (steps 1-3): Move from 350°C → growth phase
2. **Growth management** (steps 4-50): Maintain T in sweet spot (500-700°C)
3. **Precision control**: Hold temperature to let saturation naturally slow growth
4. **Timely exit**: Terminate when r reaches target (within 10-15 nm range)

Total optimal episode: ~30-50 steps at MEDIUM difficulty, earning ~150-200 reward.

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

```
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
| Agent stuck below 400°C | Frozen phase feels "safe" | Increase initial temperature or growth reward |
| Quick death (melting) | Overshoot in early training | Lower aggressive heating costs or add gradient penalty |
| No growth | Incorrect phase logic | Check temperature_phase computation in `_get_obs()` |
| Slow convergence | Large state space | Use curriculum learning (EASY → MEDIUM → HARD) |
| Reward scale issues | Mismatched time constants | Verify time_penalty coefficient (0.00028) |

## Common Questions

**Q: What's the optimal strategy?** \
A: Heat quickly to 500-600°C, maintain steady temp, let saturation slow growth naturally, exit when r = target.

**Q: Can I change the physics?** \
A: Yes! Modify constants (A, E, thresholds) in `HeatTreatmentSchedulerEnvironment` and `models.py`.

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
