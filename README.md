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

## Heat Treatment Scheduler Environment

A physics-informed reinforcement learning environment for optimizing the precipitation hardening process in metal alloys. This OpenEnv environment simulates the controlled growth of nanoprecipitates through intelligent temperature control.

## Overview

The heat treatment scheduler challenges RL agents to learn optimal temperature control strategies to grow nanoprecipitates to a target size within a material while managing time, energy, and process constraints.

### Key Features

- **Physics-Based Simulation**: Implements Arrhenius equation, diffusion-controlled growth, and Ostwald ripening
- **Four Thermal Regimes**: Distinct growth phases (Frozen, Growth, Ripening, Melting)
- **Dense Reward Shaping**: Multi-component reward guiding toward optimal behavior
- **Stochastic Environment**: Configurable difficulty levels for curriculum learning
- **WebSocket API**: Efficient persistent client connections for multi-step interactions
- **Fully Observable**: Complete state information normalized for neural network training

## Quick Start

The simplest way to use the environment is with the `HeatTreatmentSchedulerEnv` client class:

```python
from heat_treatment_scheduler import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerEnv

try:
    # Connect to running server
    env = HeatTreatmentSchedulerEnv(base_url="http://localhost:8000")
    
    # Reset to initial state
    result = env.reset()
    obs = result.observation
    
    print(f"Target radius: {obs.target_radius:.2f} (normalized)")
    print(f"Current type: {obs.temperature_phase:.0f} (0=frozen, 1=growth, 2=ripening)")
    
    # Run episode
    total_reward = 0
    for step in range(100):
        # Action: gentle heating (+10°C)
        action = HeatTreatmentSchedulerAction(action_num=3)
        result = env.step(action)
        
        obs = result.observation
        total_reward += result.reward
        
        print(f"Step {step:3d}: T={obs.temperature:.2f}, r={obs.radius:.2f}, " + 
              f"reward={result.reward:+.2f}")
        
        if result.done:
            print(f"Episode done! Total reward: {total_reward:+.1f}")
            break

finally:
    env.close()
```

### Docker Quick Start

```python
# Automatic container management
env = HeatTreatmentSchedulerEnv.from_docker_image(
    "heat_treatment_scheduler_env:latest"
)
try:
    result = env.reset()
    result = env.step(HeatTreatmentSchedulerAction(action_num=2))  # Hold temp
finally:
    env.close()
```

## Installation & Setup

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- OpenEnv framework
- NumPy, FastAPI, Uvicorn

### Quick Installation

```bash
# Clone or navigate to the project directory
cd heat_treatment_scheduler

# Install dependencies using uv
uv sync

# Start the server
uv run --project . server --port 8000

# Or using Docker
docker build -t heat_treatment_scheduler_env:latest -f server/Dockerfile .
docker run -p 8000:8000 heat_treatment_scheduler_env:latest
```

## Physics Model

### Thermal Regimes

The environment models four distinct temperature regimes affecting precipitate growth:

#### 1. Frozen Phase (T < 400°C)

- **Growth Rate**: dr/dt = 0
- **Characteristic**: No atomic diffusion, zero precipitate growth
- **Agent Implication**: Heating below 400°C costs energy with no progress

#### 2. Controlled Growth Phase (400°C ≤ T ≤ 750°C)

- **Growth Rate**: dr/dt = k(T) × (1 - r/R_max)
- **Characteristic**: Diffusion-controlled growth with saturation effect
- **Implication**: **THE SWEET SPOT** - optimal operating window
- **Physics**: Growth velocity decreases as radius approaches target (natural saturation)

#### 3. Ostwald Ripening Phase (750°C < T ≤ 1100°C)

- **Growth Rate**: dr/dt = k(T) × (r/R_max)
- **Characteristic**: Grain coarsening, positive feedback growth
- **Implication**: Failure mode - over-coarsened material becomes brittle
- **Physics**: Large precipitates grow at expense of small ones

#### 4. Melting Phase (T > 1100°C)

- **Growth Rate**: dr/dt = 0 (episode terminates immediately)
- **Characteristic**: Material begins melting
- **Implication**: Catastrophic process failure

### Arrhenius Equation

Temperature-dependent growth rate constant:

```mathematics
k(T) = A × exp(-E / (R × (T + 273.15)))
```

Parameters:

- A = 50 [reactions/second]
- E = 120,000 [J/mol]
- R = 8.314 [J/(mol·K)]
- T [°C]

**Effect**: Exponential growth rate increase with temperature

## Action Space

Six discrete temperature control actions:

| Action | Temperature Change | Description |
|--------|-------------------|-------------|
| 0 | -50°C | Aggressive cooling |
| 1 | -10°C | Gentle cooling |
| 2 | 0°C | Hold temperature steady |
| 3 | +10°C | Gentle heating |
| 4 | +50°C | Aggressive heating |
| 5 | N/A | Terminate episode |

## Observation Space

All observations are **normalized** to [0, 1] for stable neural network training:

| Field | Description | Range |
|-------|-------------|-------|
| `time` | Elapsed time / TIME_MAX (180,000s) | [0, 1] |
| `temperature` | Oven temp / 1200°C | [0, 1] |
| `radius` | Current radius / 15nm | [0, 1] |
| `target_radius` | Target radius / 15nm | [0, 1] |
| `radius_error` | (radius - target) / 15nm | [-1, 1] |
| `temperature_phase` | Categorical regime indicator | {0, 1, 2} |
| `remaining_time` | Time left / TIME_MAX | [0, 1] |
| `done` | Episode terminated | {true, false} |
| `reward` | Step reward (dense shaping) | ℝ |

## Reward Function

Carefully designed multi-component reward to guide optimal behavior:

### Step Reward (Dense)

```
step_reward = -0.1 × error - 0.01 × error²
```

Where error = |radius - target_radius|

- Linear term: Constant guidance toward target
- Quadratic term: Strong penalty near failure

### Temperature Penalty

```
temp_penalty = 0.001 × T [°C]
```

Each degree Celsius costs 0.001 reward; encourages energy efficiency.

### Time Penalty

```
time_penalty = 0.00028 × dt_effective [seconds]
```

1 hour (3600s) step ≈ 1.0 penalty; encourages quick completion.

### Terminal Rewards

| Condition | Bonus/Penalty |
|-----------|---------------|
| **Success** (R_MIN ≤ r ≤ R_MAX) | +200 base + Gaussian proximity bonus |
| **Overcoarsening** (r > R_MAX) | -100 |
| **Melting** (T ≥ 1100°C) | -200 |
| **Other Failure** | -50 |

### High Temperature Penalty

```
If T > 1000°C and not done:
  extra_penalty = (T - 1000) × 0.05 per step
```

Discourages prolonged operation near melting point.

## Boundary Conditions

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| Time | 0 | 180,000 | seconds (50 hours) |
| Temperature | 0 | 1,200 | °C |
| Radius | 0 | 30 | nm (clipped) |
| Target Radius | 10 | 15 | nm |

## Configuration

### Difficulty Levels

Control stochasticity for curriculum learning:

```python
from heat_treatment_scheduler.server import AgentGrade

AgentGrade.EASY:      # σ_T=2°C, σ_r=1%, σ_t=2%   (Clean environment)
AgentGrade.MEDIUM:    # σ_T=4°C, σ_r=3%, σ_t=5%   (Realistic variability)
AgentGrade.HARD:      # σ_T=7°C, σ_r=5%, σ_t=8%   (Challenging conditions)
```

**Recommendation**: Train curriculum style: EASY → MEDIUM → HARD

### Environment Initialization

```python
from heat_treatment_scheduler.server import HeatTreatmentSchedulerEnvironment, AgentGrade

env = HeatTreatmentSchedulerEnvironment(
    t=0.0,              # Initial time [seconds]
    T=350.0,            # Initial temperature [°C]
    r=0.5,              # Initial radius [nm]
    r_target=12.5,      # Target radius [nm]
    difficulty=AgentGrade.MEDIUM
)
obs = env.reset()
```

## Server API

### HTTP Endpoints

#### POST /reset

Reset environment to initial state.

**Response:**

```json
{
  "observation": {
    "time": 0.0,
    "temperature": 0.29,
    "radius": 0.033,
    "target_radius": 0.83,
    "radius_error": -0.8,
    "temperature_phase": 0.0,
    "remaining_time": 1.0
  }
}
```

#### POST /step

Execute action in environment.

**Request:**

```json
{"action_num": 3}
```

**Response:**

```json
{
  "observation": {...},
  "reward": -1.23,
  "done": false
}
```

#### GET /state

Get raw environment state.

**Response:**

```json
{
  "episode_id": "uuid-string",
  "step_count": 5,
  "time": 18000.0,
  "temperature": 500.0,
  "radius": 5.2,
  "target_radius": 12.5
}
```

#### WS /ws

WebSocket endpoint for persistent sessions (recommended for many steps).

## Notable Physics Features

### The "Parking Brake" Effect

An elegant emergent property of the physics:

In controlled growth phase (400-750°C):

```
dr/dt = k(T) × (1 - r/R_max)
```

As r approaches R_max, the saturation factor (1 - r/R_max) → 0.

With the clamp `dr = max(dr, 0.0)`:

**Result**: Agent can hold precipitate at perfect size (R_max) with zero growth at 750°C, effectively "parking" without triggering ripening failure. This mirrors real-world saturation perfectly.

## Training Tips

### Reward Scaling

The environment uses physics-aware reward scaling:

- Step penalties accumulate (~1.0 per hour of operation)
- Success bonus is substantial (+200) but requires precise control
- Temperature cost encourages efficiency

These form a challenging but solvable task for modern RL agents.

### Optimal Strategy

Learned policies typically exhibit:

1. Quick heating to growth phase (3-5 steps)
2. Careful growth management (50-100 steps)
3. Hold temperature near target radius (saturation exploitation)
4. Timely termination when target achieved

### Curriculum Learning

Recommended progression:

```
1. EASY: 5-10 episodes (learn basics)
   ↓
2. MEDIUM: 10-20 episodes (realistic noise)
   ↓
3. HARD: Unlimited (challenging conditions)
```

## File Structure

```
heat_treatment_scheduler/
├── __init__.py                  # Package exports
├── client.py                    # Client (EnvClient subclass)
├── models.py                    # Pydantic data models
├── inference.py                 # Example inference script
├── openenv.yaml                 # OpenEnv configuration
├── pyproject.toml               # Dependencies
├── README.md                    # This file
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI application
    ├── heat_treatment_scheduler_environment.py  # Core simulation engine
    ├── Dockerfile               # Container definition
    └── requirements.txt
```

## Building Docker Image

```bash
docker build -t heat_treatment_scheduler_env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

Deploy your environment as a public/private space:

```bash
# Authenticate first
huggingface-cli login

# Push from project root
openenv push

# Or with options
openenv push --namespace my-org --private
```

## Common Issues & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Temperature oscillations | Reward not penalizing swings | Increase temp cost or add smoothing |
| Stuck in frozen phase | Below 400°C too safe | Adjust initial temp or growth bonus |
| Immediate termination | Early exit more rewarding | Increase success bonus |
| Melting catastrophe | High temp too rewarding | Increase melting penalty |
| Slow convergence | Large RL task | Use curriculum learning |

## Contributing

When modifying:

1. **Preserve Physics**: Keep Arrhenius equation and phase transitions intact
2. **Test Boundaries**: Verify behavior at 400°C, 750°C, 1100°C
3. **Comment Changes**: Update inline documentation
4. **Validate Reward**: Ensure terminal conditions properly reflect outcomes
5. **Check API**: Ensure clients can parse all observations

## License

© Meta Platforms, Inc. and affiliates.  
Licensed under the BSD-style license. See LICENSE file for details.

## Support

For detailed physics explanations, see inline comments in:

- `heat_treatment_scheduler_environment.py` - Core physics and reward
- `models.py` - Full observation/state/action definitions
- `client.py` - Protocol details
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action

**HeatTreatmentSchedulerAction**: Discrete temperature control command

- `action_num` (int, 0-5) - Temperature control command:
  - 0: Aggressive cooling (-50°C)
  - 1: Gentle cooling (-10°C)
  - 2: Hold temperature (0°C)
  - 3: Gentle heating (+10°C)
  - 4: Aggressive heating (+50°C)
  - 5: Terminate episode

### Observation

**HeatTreatmentSchedulerObservation**: Normalized state information for agent training

- `time` (float, [0,1]) - Normalized elapsed time
- `temperature` (float, [0,1]) - Normalized oven temperature
- `radius` (float, [0,1]) - Normalized current precipitate radius
- `target_radius` (float, [0,1]) - Normalized target radius
- `radius_error` (float, [-1,1]) - Error from target radius
- `temperature_phase` (float, {0, 1, 2}) - Current thermal regime (0=Frozen, 1=Growth, 2=Ripening)
- `remaining_time` (float, [0,1]) - Normalized time remaining
- `done` (bool) - Episode completion flag
- `reward` (float) - Step reward value
- `metadata` (dict) - Additional information

### Reward

Carefully shaped multi-component reward:

- **Step Reward**: Dense error-based guidance: `-0.1 × error - 0.01 × error²`
- **Temperature Cost**: Energy penalty proportional to temperature
- **Time Penalty**: Encourages efficient completion
- **Terminal Bonuses**: +200 for success, penalties for failures

## Advanced Usage

### Connecting to an Existing Server

If you already have a Heat Treatment Scheduler environment server running:

```python
from heat_treatment_scheduler import HeatTreatmentSchedulerEnv, HeatTreatmentSchedulerAction

# Connect to existing server
env = HeatTreatmentSchedulerEnv(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
obs = result.observation

# Take action (gentle heating)
action = HeatTreatmentSchedulerAction(action_num=3)
result = env.step(action)

print(f"Temperature: {obs.temperature:.2f}, Radius: {obs.radius:.2f}, Reward: {result.reward:.2f}")
```

Note: When connecting to an existing server, `env.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from heat_treatment_scheduler import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerEnv

# Connect with context manager (auto-connects and closes)
with HeatTreatmentSchedulerEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation
    print(f"Target radius: {obs.target_radius:.2f}")
    
    # Run episode with low latency WebSocket connection
    for step in range(50):
        action = HeatTreatmentSchedulerAction(action_num=3)  # Gentle heating
        result = env.step(action)
        obs = result.observation
        
        if result.done:
            print(f"Episode complete in {step} steps")
            break
```

The client uses WebSocket connections for:

- **Lower latency**: Persistent connection, no HTTP overhead per step
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential environment steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to allow more concurrent environments:

```python
# In server/app.py - increase concurrent sessions
app = create_app(
    HeatTreatmentSchedulerEnvironment,
    HeatTreatmentSchedulerAction,
    HeatTreatmentSchedulerObservation,
    env_name="heat_treatment_scheduler",
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from heat_treatment_scheduler import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(agent_id: int):
    with HeatTreatmentSchedulerEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        total_reward = 0.0
        
        for step in range(100):
            # Simple agent: alternate between heating and holding
            action_num = 3 if step % 2 == 0 else 2
            action = HeatTreatmentSchedulerAction(action_num=action_num)
            result = env.step(action)
            total_reward += result.reward
            
            if result.done:
                break
        
        return agent_id, total_reward

# Run multiple episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
    for agent_id, total_reward in results:
        print(f"Agent {agent_id}: Total Reward = {total_reward:.2f}")
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```python
from heat_treatment_scheduler.server import HeatTreatmentSchedulerEnvironment, AgentGrade
from heat_treatment_scheduler import HeatTreatmentSchedulerAction

# Create environment directly
env = HeatTreatmentSchedulerEnvironment(
    t=0.0,
    T=300.0,
    r=0.0,
    r_target=12.5,
    difficulty=AgentGrade.EASY
)

obs = env.reset()
print(f"Initial observation: time={obs.time}, temp={obs.temperature}, radius={obs.radius}")

# Take a step
action = HeatTreatmentSchedulerAction(action_num=3)
obs = env.step(action)
print(f"After heating: temp={obs.temperature}, reward={obs.reward}")
```

This verifies that:

- Environment resets correctly with initial state
- Step executes actions and updates state properly
- Physics calculations work as expected
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
# Install dependencies
uv sync

# Start with auto-reload for development
uv run --project . server

# Or use uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
heat_treatment_scheduler/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # HeatTreatmentSchedulerEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── heat_treatment_scheduler_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
