# Theme Selection

- Primary: Long-Horizon Planning & Instruction Following

- Secondary: World Modeling (Physical Systems)

## 1. The Problem Statement

Modern aerospace and industrial manufacturing relies on complex, multi-stage heat treatments (e.g., T6 tempering) for High-Entropy Alloys and Superalloys. Current AI agents struggle with continuous physical control involving severe delayed dynamics. The challenge is to evaluate an agent's ability to follow complex, multi-step natural language thermal recipes over long time horizons (up to 50 hours), while predictively managing non-linear thermodynamic constraints such as thermal mass inertia (lag) and dynamic oxidation insulation.

## 2. The Environment

The Heat Treatment Digital Twin (V2). It is a configuration-driven, continuous **Semi-Markov Decision Process (SMDP)**. Instead of discrete, game-like steps, the environment is powered by a *continuous ODE solver* (SciPy `solve_ivp`). It dynamically loads intrinsic material properties (e.g., Arrhenius kinetics for growth and oxidation) and extrinsic hardware geometries. The environment perfectly simulates the thermodynamic reality that furnace air temperature changes instantly, but material core temperature curves asymptotically based on specific heat capacity and mass.

## 3. Capabilities of the Agent

The agent operates as a Metallurgical Process Controller.

- **Semantic Perception**: It interprets multi-stage natural language processing recipes and contextual material properties.

- **Continuous State Monitoring**: It reads real-time normalized telemetry (elapsed time, material core temperature, precipitate radius, error margins).

- **Variable-Time Control (SMDP)**: It outputs decoupled action-duration pairs (`[Temperature_Action, Duration_Minutes]`). This allows the agent to make micro-adjustments (1 minute) during critical phase transitions, or macro-holds (hours) during steady-state baking.

## 4. Tasks to be Performed

Agents must execute complex thermal recipes on entirely different alloys and hardware configurations without prior hardcoding.

- **Example Task**: "Execute a T6 treatment on Titanium Ti-6Al-4V in a massive industrial casting. First, solutionize by holding above 1000°C for 2 hours. Next, rapidly quench below 200°C. Finally, execute an artificial aging phase to grow nanoprecipitates to exactly 22.5nm."

- **The Core Challenge**: Because of the massive thermal inertia, the agent must learn "Predictive Braking"—cutting the furnace heat long before the material reaches the target temperature to prevent residual heat from causing catastrophic Ostwald Ripening.

## 5. Reward Model & Evaluation Logic

The evaluation utilizes a dense, multi-component reward structure designed to prevent local optima:

- **Terminal Proximity Bonus**: High positive reward (+100 to +200) Gaussian-scaled based on the final radius's proximity to the target radius, but only if the episode terminates safely.

- **Catastrophic Failure Penalties**: Massive penalties for melting the material (-200) or over-coarsening beyond the maximum radius threshold (-100).

- **Efficiency Shaping**: Continuous time and energy penalties (penalizing high furnace temperatures) to force the agent to find the most efficient thermodynamic trajectory rather than taking infinitely long safe routes.

## 6. Post-Training & Self-Improvement Strategy

The architecture natively supports generating high-quality offline datasets.

1. **Trajectory Generation**: We will run standard frontier LLMs (e.g., Llama-3-70B, GPT-4o) against the environment to generate thousands of thermal trajectories.

2. **Preference Optimization**: Using the dense reward model, trajectories will be ranked. Failed trajectories (overshooting due to thermal lag) vs. Successful trajectories (proper predictive braking) will be paired.

3. **Self-Improvement**: We will apply *Direct Preference Optimization (DPO)* or *Supervised Fine-Tuning (SFT)* on an open-source, small language model (e.g., Llama-3-8B) using these paired trajectories. The goal is to prove that a small, self-improved agent can internalize continuous differential equations and outperform a zero-shot frontier model in managing physical lag.
