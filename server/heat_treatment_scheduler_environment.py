# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Heat Treatment Scheduler Environment Implementation.

This module implements a reinforcement learning environment that simulates the precipitation
hardening process in metal alloys. The agent learns to control oven temperature to achieve
a target nanoprecipitate size while managing time and energy constraints.

Physics Model Overview:
    The environment simulates precipitate growth through four distinct thermal regimes:
    
    1. FROZEN PHASE (T < 400°C):
       - Atomic diffusion is negligible
       - Precipitate growth rate: dr/dt = 0
       - Material remains in initial microstructure
       
    2. CONTROLLED GROWTH PHASE (400°C ≤ T ≤ 750°C):
       - Diffusion-controlled growth dominates
       - Growth rate follows Arrhenius equation + diffusion saturation
       - Formula: dr/dt = k(T) * (1 - r/R_max)
       - This is the target operating window for optimal precipitation
       - Rate-limiting factor: atomic diffusion in the solid
       
    3. OSTWALD RIPENING PHASE (750°C < T ≤ 1100°C):
       - Grain coarsening becomes dominant
       - Large precipitates grow at expense of smaller ones
       - Growth rate: dr/dt = k(T) * (r/R_max)
       - Material becomes brittle and loses mechanical properties
       - Failure mode if radius exceeds R_max
       
    4. MELTING PHASE (T > 1100°C):
       - Material begins melting
       - Crystalline structure dissolves
       - Immediate episode termination with severe penalty

Mathematical Formulation:

    Arrhenius Equation (Temperature-dependent rate constant):
        k(T) = A * exp(-E / (R * (T + 273.15)))
        
        where:
            k(T) = rate constant [reactions/second]
            A = pre-exponential factor [reactions/second]
            E = activation energy [J/mol]
            R = universal gas constant [8.314 J/(mol·K)]
            T = temperature [°C]
            273.15 = conversion from Celsius to Kelvin

    Radius Evolution (Discrete time step):
        r(t+1) = r(t) + dt * f(r, T)
        
        where:
            dt = time step duration [seconds]
            f(r, T) = growth rate function (temperature-dependent)

Example Usage:
    >>> env = HeatTreatmentSchedulerEnvironment(
    ...     t=0.0,           # Start at 0 seconds
    ...     T=300.0,         # Start at 300°C
    ...     r=1.0,           # Initial radius 1 nm
    ...     r_target=12.5,   # Target radius 12.5 nm
    ...     difficulty=AgentGrade.EASY
    ... )
    >>> obs = env.reset()
    >>> action = HeatTreatmentSchedulerAction(action_num=3)  # Heat by 10°C
    >>> obs = env.step(action)
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from enum import IntEnum
from typing import Any
import numpy as np
import random

try:
    from ..models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation, HeatTreatmentSchedulerState, R_MAX, R_MIN, TEMP_MAX, TEMP_MIN, TIME_MAX, TIME_MIN
    from ..logging_config import get_logger
except ImportError:
    from models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation, HeatTreatmentSchedulerState, R_MAX, R_MIN, TEMP_MAX, TEMP_MIN, TIME_MAX, TIME_MIN
    from logging_config import get_logger

# Module logger
logger = get_logger(__name__)
logger.debug("Heat Treatment Scheduler environment module loaded")


class AgentGrade(IntEnum):
    """
    Difficulty level enumeration controlling noise in the environment.
    
    Higher difficulty = more stochasticity in temperature, growth, and timing.
    Useful for curriculum learning: start with EASY, progress to HARD.
    """
    EASY = 1    # Low noise: suitable for learning basics
    MEDIUM = 2  # Moderate noise: realistic variability
    HARD = 3    # High noise: challenging real-world conditions


class HeatTreatmentSchedulerEnvironment(Environment):
    """
    Reinforcement learning environment for heat treatment scheduling.

    This environment simulates the precipitation hardening process in metal alloys,
    a critical manufacturing process for creating high-strength materials used in
    aerospace, automotive, and industrial applications.

    Core Concepts:
        - State: (time, temperature, radius) tuple representing process conditions
        - Action: Discrete temperature control {-50, -10, 0, +10, +50, term}
        - Observation: Normalized state metrics for neural network training
        - Reward: Dense shaped reward guiding agent toward target radius

    Task Description:
        The agent controls oven temperature to grow nanoprecipitates to a target radius
        within a material sample. Success requires:
        1. Reaching target radius: R_MIN ≤ r ≤ R_MAX
        2. Avoiding overcoarsening: r should not exceed R_MAX
        3. Avoiding melting: T should not exceed ~1100°C
        4. Minimizing time: Reaching target quickly (energy efficiency)
        5. Stable control: Smooth temperature changes preferred over rapid swings

    State Space:
        - time: Elapsed seconds [0, TIME_MAX]
        - temperature: Oven temperature [TEMP_MIN, TEMP_MAX]
        - radius: Precipitate radius [0, R_MAX * 2]
        - All transmitted as normalized observations [0, 1]

    Action Space:
        - 0: Aggressive cooling (-50°C)
        - 1: Gentle cooling (-10°C)
        - 2: Hold temperature (0°C change)
        - 3: Gentle heating (+10°C)
        - 4: Aggressive heating (+50°C)
        - 5: Terminate episode (signal completion)

    Reward Function:
        - Step reward: Dense error-based reward (encouraged at each step)
        - Terminal reward: Bonus for successful completion
        - Penalties: Temperature cost, time cost, boundaries violations

    Notable Features:
        - Physics-aware reward: Penalizes wasted energy (high temperature cost)
        - Time-aware penalty: Scaled by actual elapsed time (dt with jitter)
        - Saturation dynamics: Natural growth slowdown as radius approaches target
        - Stochastic environment: Noise configurable by difficulty level

    SUPPORTS_CONCURRENT_SESSIONS:
        Set to True, allowing multiple WebSocket clients to each have their own
        environment instance on the server.
    """

    # ======================== CONCURRENCY SUPPORT ========================
    # Enable concurrent WebSocket sessions.
    # Set to True if environment properly isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # ======================== PHYSICAL CONSTANTS ========================
    # These are material and process dependent constants for the physics model.

    # Universal Gas Constant
    R: float = 8.314
    """
    Universal gas constant [J/(mol·K)].
    
    Used in Arrhenius equation to convert temperature to activation rates.
    Standard value: 8.314 J/(mol·K)
    """

    # Pre-exponential Factor (Arrhenius)
    A: float = 1000
    """
    Pre-exponential factor in Arrhenius equation [reactions/second].
    
    Material and diffusion mechanism dependent. Represents frequency of
    atomic collision attempts leading to precipitate growth.
    
    Typical range in metallurgy: 10^7 - 10^13 reactions/second
    Our simplified value: 1000 reactions/second
    """

    # Activation Energy
    E: float = 120_000
    """
    Molar activation energy for precipitation [J/mol].
    
    The energy barrier atoms must overcome to participate in precipitate growth.
    Typical range for precipitation in alloys: 50 - 300 kJ/mol
    Our value: 120 kJ/mol (representative of aluminum-copper alloys)
    
    Effect: Higher E → slower growth at low temperatures but faster growth at high T
    """

    # ======================== NOISE CONFIGURATION ========================
    # Stochasticity scaling by difficulty level (curriculum learning friendly)

    NOISE_CONFIG = {
        AgentGrade.EASY: {
            "temp": 2.0,      # Temperature fluctuation std dev [°C]
            "growth": 0.01,   # Growth rate noise [fraction: 1% std dev]
            "jitter": 0.02,   # Time step jitter [fraction: 2% std dev]
        },
        AgentGrade.MEDIUM: {
            "temp": 4.0,      # 4°C fluctuations (more realistic furnace variation)
            "growth": 0.03,   # 3% growth uncertainty
            "jitter": 0.05,   # 5% timing jitter
        },
        AgentGrade.HARD: {
            "temp": 7.0,      # Large temperature swings possible (worst-case furnace)
            "growth": 0.05,   # 5% growth uncertainty (material variability)
            "jitter": 0.08,   # 8% timing variation (process uncertainty)
        }
    }

    def __init__(self, t=0.0, T=37.0, r=0.0, r_target=12.5, difficulty=AgentGrade.EASY) -> None:
        """
        Initialize the Heat Treatment Scheduler environment.

        Creates a new environment instance with specified initial conditions
        and physics parameters. Each environment simulation is deterministic
        given a seed, enabling reproducible experiments.

        Args:
            t: Initial elapsed time [seconds]. Defaults to 0.0.
            T: Initial oven temperature [°C]. Defaults to 37.0 (frozen phase).
               Note: Typically agents heat to 300-400°C range in training.
            r: Initial precipitate radius [nm]. Defaults to 0.0.
            r_target: Target precipitate radius [nm]. Defaults to 12.5.
                      Should be in range [R_MIN, R_MAX] = [10.0, 15.0].
            difficulty: AgentGrade level determining noise magnitude.
                       Defaults to EASY.
                - EASY (1): Great for learning environment dynamics
                - MEDIUM (2): Realistic physics with moderate stochasticity
                - HARD (3): Challenging conditions resembling factory variability

        Example:
            >>> # Using defaults (cold start)
            >>> env = HeatTreatmentSchedulerEnvironment()
            >>> obs = env.reset()
            
            >>> # Custom initial conditions
            >>> env = HeatTreatmentSchedulerEnvironment(
            ...     t=0.0,
            ...     T=350.0,
            ...     r=0.5,
            ...     r_target=12.5,
            ...     difficulty=AgentGrade.MEDIUM
            ... )
            >>> obs = env.reset()                                        
        """
        logger.debug(f"Initializing HeatTreatmentSchedulerEnvironment: t={t}, T={T}, r={r}, r_target={r_target}, difficulty={difficulty.name}")
        
        # Store initial conditions (used for reset())
        self.init_t = t  # in seconds (s)
        self.init_T = T  # in degrees Celsius (C)
        self.init_r = r  # in nanometers (nm)
        self.r_target = r_target  # in nanometers (nm)
        self._reset_count = 0
        self.dt = 3600.0
        """
        Discrete time step size [seconds]. Default: 3600 (one hour).
        
        At each step, the environment advances time by approximated dt.
        This represents one heating/cooling cycle in the manufacturing process.
        """
        
        # Configure noise based on difficulty level
        config = self.NOISE_CONFIG[difficulty]
        self.sigma_T = config["temp"]      # Temperature noise std dev [°C]
        self.sigma_r = config["growth"]    # Growth rate noise multiplier
        self.sigma_t = config["jitter"]    # Time jitter as fraction of dt
        """
        Noise parameters extracted from NOISE_CONFIG based on difficulty.
        - sigma_T: Temperature sensor/furnace noise [°C]
        - sigma_r: Material variability in growth rate
        - sigma_t: Process timing uncertainty
        """
        logger.debug(f"Noise config: sigma_T={self.sigma_T}, sigma_r={self.sigma_r}, sigma_t={self.sigma_t}")

        # Initialize the environment
        self.reset()


    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any) -> HeatTreatmentSchedulerObservation:
        """
        Reset the environment to initial conditions and start a new episode.

        This method is called at the beginning of each training episode to restore
        the environment to a clean state. It handles seeding for reproducibility,
        initializes/restores state variables, resets the step counter, and returns
        the initial observation.

        Args:
            seed: Random seed for reproducibility. If provided, numpy and
                  Python random modules are seeded with this value.
            episode_id: Optional unique identifier for this episode. If None,
                       a new UUID is generated for tracking.
            **kwargs: Additional keyword arguments (ignored, for API compatibility)

        Returns:
            HeatTreatmentSchedulerObservation: Initial normalized observation
                                              with done=False and reward=0.0

        Example:
            >>> env = HeatTreatmentSchedulerEnvironment(0, 350, 0.5, 12.5)
            >>> obs = env.reset(seed=42)  # Reproducible randomness
            >>> print(obs.temperature)  # Normalized temperature
            >>> print(obs.radius)       # Normalized radius (0-1)
        """
        # ---- SEEDING FOR REPRODUCIBILITY ----
        # If a seed is provided, set random number generators for reproducibility
        if seed is not None:
            self.SEED = seed
            random.seed(self.SEED)
            np.random.seed(self.SEED)
        # -----------------------------------

        self._reset_count += 1
        self.step_count = 0
        self.step_count = 0  # Reset step counter for new episode

        # Restore initial conditions
        self.t = self.init_t
        self.T = self.init_T
        self.r = self.init_r

        # Create fresh state object for this episode
        self._state = self._get_state(episode_id)
        logger.info(f"Environment reset #{self._reset_count}: episode_id={self._state.episode_id}, step_count=0")
        
        # Return initial observation (no action taken yet)
        return self._get_obs(done=False, reward=0.0)
    

    def _get_state(self, episode_id: str | None = None) -> HeatTreatmentSchedulerState:
        """
        Create a State object representing the current raw (unnormalized) environment state.

        State differs from Observation:
        - State: Raw physical units (seconds, Celsius, nanometers)
        - Observation: Normalized values [0, 1] for neural network training

        This method is called internally to maintain the internal state tracking
        required by the OpenEnv framework. Returns raw state with step_count from
        the environment's direct attribute self.step_count.

        Args:
            episode_id: Unique identifier for current episode. If None, generates new UUID.

        Returns:
            HeatTreatmentSchedulerState: Object with raw state values (t, T, r, r_target)
                                        plus metadata (episode_id, step_count)
        """
        return HeatTreatmentSchedulerState(
            episode_id=episode_id or str(uuid4()),
            step_count=getattr(self, "step_count", 0),
            time=self.t,
            temperature=self.T,
            radius=self.r,
            target_radius=self.r_target
        )
    

    def _get_obs(self, done, reward) -> HeatTreatmentSchedulerObservation:
        """
        Construct the normalized observation from raw state variables.

        This method normalizes all Observations to [0, 1] range for stable
        neural network training. Normalization is crucial for:
        - Preventing gradient explosion/vanishing
        - Ensuring features have similar scales
        - Improving convergence speed
        - Enabling generalization across different scales

        Normalization Formula:
            x_normalized = x_raw / x_max

        Logic for temperature_phase (categorical as float):
            - 0.0 if T < 400°C (Frozen: no growth)
            - 1.0 if 400 ≤ T ≤ 750°C (Growth: controlled growth)
            - 2.0 if T > 750°C (Ripening: coarsening, dangerous)

        Args:
            done: Episode termination flag (bool)
            reward: Reward scalar for this step (float)

        Returns:
            HeatTreatmentSchedulerObservation: All values normalized to [0, 1]
        """
        # Determine temperature phase for categorical encoding
        if self.T < 400:
            phase = 0.0  # Frozen: no growth
        elif self.T <= 750:
            phase = 1.0  # Growth: controlled growth phase
        else:
            phase = 2.0  # Ripening: ostwald ripening (dangerous)

        # Normalize time: t / TIME_MAX
        normalized_time = self.t / TIME_MAX

        # Construct and return normalized observation
        return HeatTreatmentSchedulerObservation(
            time=normalized_time,
            temperature=self.T / TEMP_MAX,
            radius=self.r / R_MAX,
            target_radius=self.r_target / R_MAX,
            radius_error=(self.r - self.r_target) / R_MAX,
            temperature_phase=phase,
            remaining_time=max(0.0, (1-normalized_time)),
            done=done,
            reward=reward
        )

    
    def step(self, action: HeatTreatmentSchedulerAction, timeout_s: float | None=None, **kwargs: Any) -> HeatTreatmentSchedulerObservation:
        """
        Execute one step in the heat treatment process simulation.

        This is the main simulation loop that applies the agent's action and computes
        the next state using the physics model. The step involves:
        1. Validate the action
        2. Update time: t → t + dt (with noise)
        3. Update temperature: T → T + dT (with noise)
        4. Compute precipitate growth based on Arrhenius equation
        5. Check termination conditions
        6. Compute reward using dense reward shaping
        7. Return observation and metrics

        State Transitions:
            Input state: (t, T, r) - time, temperature, radius
            
            Action: dT ∈ {-50, -10, 0, +10, +50} or TERMINATE
            
            Time update:
                noise_t ~ N(0, sigma_t), clipped to [-0.5, 0.5]
                dt_effective = dt * (1 + noise_t)
                t_new = t + dt_effective
            
            Temperature update:
                noise_T ~ N(0, sigma_T)
                T_new = clip(T + dT + noise_T, TEMP_MIN, TEMP_MAX)
                
                Critical thresholds:
                - T ≥ 1100°C: Material melting starts (episode ends)
                - T > 1200°C: Hard limit (fully clipped)
            
            Growth update:
                dr/dt = f(r, T) computed via Arrhenius equation
                noise_r ~ N(0, sigma_r), clipped to [-0.3, 0.3]
                dr = dr/dt * dt_effective * (1 + noise_r)
                r_new = clip(r + max(dr, 0.0), 0, R_MAX * 2)
            
            Output state: (t_new, T_new, r_new)

        Episode Termination Conditions:
            1. Agent action = 5 (TERMINATION_SIGNAL)
            2. t ≥ TIME_MAX (50 hours elapsed)
            3. T ≥ 1100°C (Material melting)
            4. r > R_MAX (Ostwald ripening failure)

        Args:
            action: HeatTreatmentSchedulerAction with action_num in [0, 5]
            timeout_s: Optional timeout for environment step (unused)
            **kwargs: Additional parameters (for API compatibility)

        Returns:
            HeatTreatmentSchedulerObservation: Normalized observation with reward and done flag

        Raises:
            ValueError: If action_num is not in ACTION_MAP (invalid action)

        Example:
            >>> env = HeatTreatmentSchedulerEnvironment(0, 350, 0.5, 12.5)
            >>> env.reset()
            >>> action = HeatTreatmentSchedulerAction(action_num=3)  # Heat by 10°C
            >>> obs = env.step(action)
            >>> print(f"Reward: {obs.reward}, Done: {obs.done}")
        """
        # Increment step counter
        # self._state.step_count += 1
        self.step_count += 1

        # ---- ACTION VALIDATION ----
        # Check that the action is valid and recognized
        if action.action_num not in HeatTreatmentSchedulerAction.ACTION_MAP:
            raise ValueError(f"Invalid action: {action.action_num}. Must be 0-5.")
        
        logger.debug(f"Step {self.step_count}: action_num={action.action_num}")
        
        # ---- TERMINATION CONDITIONS ----
        # Check if agent explicitly terminates episode or max time reached
        if action.action_num == HeatTreatmentSchedulerAction.TERMINATION_SIGNAL or self.t >= TIME_MAX:
            logger.info(f"Episode terminated: action={action.action_num}, t={self.t:.0f}s")
            return self._get_obs(done=True, reward=self._get_reward(done=True, dt_effective=0.0))
        
        # ---- TIME UPDATE: t → t + dt ----
        # Add small random jitter to simulate real-world timing variation
        noise_t = np.random.normal(0, self.sigma_t)  # ~N(0, sigma_t)
        noise_t = np.clip(noise_t, -0.5, 0.5)  # Limit to ±50% of dt
        dt_effective = self.dt * (1 + noise_t)  # 3600*(1+noise_t) seconds
        self.t += dt_effective  # Update elapsed time

        # ---- TEMPERATURE UPDATE: T → T + dT ----
        # Get action temperature change from mapping
        dT = HeatTreatmentSchedulerAction.ACTION_MAP.get(action.action_num)

        # Add Gaussian noise for realistic furnace/sensor noise
        noise_T = np.random.normal(0, self.sigma_T)  # ~N(0, sigma_T) °C
        # Update temperature with action + noise, clipped to [TEMP_MIN, TEMP_MAX]
        self.T = np.clip(self.T + dT + noise_T, TEMP_MIN, TEMP_MAX)
        
        # ---- EARLY TERMINATION: MELTING CHECK ----
        # If temperature exceeds melting point, material is destroyed
        if self.T >= 1100:
            logger.warning(f"Step {self.step_count}: Melting occurred at T={self.T:.1f}°C")
            return self._get_obs(done=True, reward=self._get_reward(done=True, dt_effective=dt_effective))
        
        # ---- PRECIPITATE GROWTH: dr/dt → dr ----
        # Compute growth rate based on Arrhenius equation and temperature phase
        dr_dt = self._get_growth_rate(self.r, self.T)
        
        # Add material variability noise (manufacturing tolerance)
        noise_r = np.random.normal(0, self.sigma_r)  # ~N(0, sigma_r)
        noise_r = np.clip(noise_r, -0.3, 0.3)  # Limit to ±30%
        # Delta radius = growth_rate * time * (1 + noise)
        dr = dr_dt * dt_effective * (1 + noise_r) 
        # Ensure growth is non-negative (no shrinkage in this model)
        dr = max(dr, 0.0)  # Growth only, no negative dr
        
        # Update radius with safety margin
        self.r = np.clip(self.r + dr, 0.0, R_MAX * 2)

        # ---- TERMINATION CONDITION: OVERCOARSENING CHECK ----
        # If radius exceeds target range, material is overcoarsened
        done = self.t >= TIME_MAX or self.r > R_MAX
        
        if self.r > R_MAX:
            logger.warning(f"Step {self.step_count}: Overcoarsening at r={self.r:.3f} nm (max={R_MAX})")
        
        # Compute reward for this step (dense shaping + terminal bonuses)
        reward = self._get_reward(done=done, dt_effective=dt_effective)

        # ---- UPDATE INTERNAL STATE ----
        # Refresh state object with new values
        self._state = self._get_state(self._state.episode_id)

        logger.debug(f"Step {self.step_count}: T={self.T:.1f}°C, r={self.r:.3f} nm, reward={reward:.2f}, done={done}")
        
        # Return normalized observation and metrics
        return self._get_obs(done=done, reward=reward)
    
        # ** QUIRK
        # A Cool Physics Quirk: The "Parking Brake"
        # This isn't a bug, but rather an elegant side-effect of the math.
        # Look at the diffusion-controlled growth (400°C to 750°C):
        # dr_dt = k * (1 - (r / R_MAX))
        # If the agent heats the oven to 700°C, the precipitates will grow quickly. 
        # But as r approaches R_MAX (15.0), the term (1 - (r / R_MAX)) approaches 0.
        # Because dr = max(dr, 0.0), if the radius hits 15.001, the growth rate becomes negative, which gets clamped to exactly 0.0.
        # The Result: At exactly 750°C, the precipitates will hit 15.0nm and stop growing entirely. 
        # The agent can effectively "park" the alloy at the perfect size without triggering the Ostwald ripening penalty,
        #  provided it doesn't cross the 750°C threshold. This actually mirrors the real-world concept of "Saturation" perfectly. 


    @property
    def state(self) -> HeatTreatmentSchedulerState:
        """
        Get the current environment state (raw, unnormalized values).

        The state property provides access to the internal state representation,
        which differs from observations. State is useful for:
        - Debugging (raw physical values in familiar units)
        - Logging and analysis (Celsius, seconds, nanometers)
        - UI visualization (no denormalization needed)
        - Physics validation (checking boundary conditions)

        Returns:
            HeatTreatmentSchedulerState with:
            - episode_id: Unique identifier for current episode (from _state)
            - step_count: Number of steps taken in current episode (from self.step_count)
            - time: Elapsed time (seconds)
            - temperature: Oven temperature (Celsius)
            - radius: Precipitate radius (nanometers)
            - target_radius: Target radius (nanometers)
        """
        return self._state
    
    
    def _get_growth_rate(self, r, T) -> float:
        """
        Compute the instantaneous growth rate dr/dt based on temperature regime.

        The growth rate is the most physically important component of the environment.
        It models how nanoprecipitate radius changes over time based on the Arrhenius
        equation and the thermodynamic phase of the material.

        Physics Model (Three Temperature Regimes):

            1. FROZEN PHASE (T < 400°C):
                Atomic diffusion is severely limited. Precipitate growth is essentially
                zero. Atoms lack enough thermal energy to move through the solid matrix.
                Formula: dr/dt = 0
                Physical Interpretation: No growth

            2. CONTROLLED GROWTH PHASE (400°C ≤ T ≤ 750°C):
                Diffusion-controlled growth dominates. Atoms have enough thermal energy
                to diffuse through the matrix but not enough to trigger Ostwald ripening.
                This is the SWEET SPOT for precipitation hardening.
                
                Formula: dr/dt = k(T) * (1 - r/R_max)
                
                Components:
                - k(T) = A * exp(-E / (R * (T + 273.15)))  [Arrhenius equation]
                - (1 - r/R_max) = saturation factor
                
                As r → R_max, the growth slows dramatically (saturation effect).
                This provides natural encouragement for the agent to stop heating near target.

            3. OSTWALD RIPENING PHASE (750°C < T ≤ 1100°C):
                High-temperature grain coarsening dominates. Larger precipitates grow
                at the expense of smaller ones, leading to loss of mechanical properties.
                This is FAILURE MODE to avoid.
                
                Formula: dr/dt = k(T) * (r/R_max)
                
                The growth rate increases with radius (feedback effect).
                Large precipitates grow faster → material becomes brittle.

            4. MELTING PHASE (T > 1100°C):
                Material begins to melt. Structure breaks down. Growth rate stops.
                Formula: dr/dt = 0 (but episode terminates immediately)

        Arrhenius Equation Explanation:
            k(T) = A * exp(-E / (R * (T + 273.15)))
            
            - k(T): Temperature-dependent reaction rate [reactions/second]
            - A: Pre-exponential factor (attempt frequency) [reactions/second]
            - E: Activation energy (energy barrier) [J/mol]
            - R: Universal gas constant = 8.314 [J/(mol·K)]
            - T+273.15: Absolute temperature [Kelvin]
            
            Behavior:
            - As T increases exponentially: k(T) increases exponentially
            - Higher E: Slower growth at low T, faster growth at high T
            - Effect of noise: Small T variations → large k(T) variations

        Args:
            r: Current precipitate radius [nanometers]
            T: Current temperature [Celsius]

        Returns:
            Growth rate dr/dt [nanometers/second]

        Example:
            >>> env = HeatTreatmentSchedulerEnvironment(0, 350, 0.5, 12.5)
            >>> # At 300°C (frozen phase): no growth
            >>> dr_dt = env._get_growth_rate(5.0, 300)
            >>> assert dr_dt == 0
            
            >>> # At 500°C (growth phase): positive growth
            >>> dr_dt = env._get_growth_rate(5.0, 500)
            >>> assert dr_dt > 0
            
            >>> # At 850°C (ripening phase): growth increases with radius
            >>> dr_dt = env._get_growth_rate(10.0, 850)
            >>> assert dr_dt > 0
        """
        # ---- ARRHENIUS EQUATION ----
        # Compute temperature-dependent rate constant (in Kelvin)
        k = self.A * np.exp(-self.E / (self.R * (T + 273.15)))
        
        # ---- PHASE-DEPENDENT GROWTH ----
        if T < 400:
            # FROZEN PHASE: No atomic diffusion, zero growth
            return 0.0
        
        elif T <= 750:
            # CONTROLLED GROWTH PHASE: Diffusion-limited growth with saturation
            # As r approaches R_max, saturation factor (1 - r/R_max) approaches 0
            # This creates a natural dampening effect
            return k * (1.0 - (r / R_MAX))
        
        elif T <= 1100:
            # OSTWALD RIPENING PHASE: Grain coarsening (dangerous)
            # Large precipitates grow even faster (positive feedback)
            return k * (r / R_MAX)
        
        else:
            # MELTING PHASE: No growth (episode ends immediately anyway)
            return 0.0


    def _get_reward(self, done, dt_effective) -> float:
        """
        Compute the reward signal for this step.

        The reward function is carefully designed to guide the agent toward the optimal
        policy while providing informative feedback at every step. The reward combines:
        1. Dense step reward: Guides toward target radius
        2. Penalties: Discourage excessive temperature and time
        3. Terminal bonuses: Incentivize successful completion

        Reward Components:

            1. ERROR-BASED STEP REWARD (Dense Shaping):
                step_reward = -0.1 * error - 0.01 * error^2
                
                Where error = |r - r_target|
                
                - First term: Linear penalty (encourages progress)
                - Second term: Quadratic penalty (strong penalty near failure)
                - Activated every step (dense reward for learning)
                
                Examples:
                - error = 0 nm: step_reward = 0
                - error = 1 nm: step_reward = -0.1 - 0.01 = -0.11
                - error = 5 nm: step_reward = -0.5 - 0.25 = -0.75
                - error = 10 nm: step_reward = -1.0 - 1.0 = -2.0

            2. TEMPERATURE PENALTY (Energy Cost):
                temp_penalty = 0.001 * T
                
                Penalizes heating to high temperatures (energy inefficiency).
                - Each degree Celsius costs 0.001 reward
                - 500°C step costs -0.5 reward
                - 1000°C step costs -1.0 reward
                
                Encourages the agent to use the sweet spot (400-750°C) efficiently.

            3. TIME PENALTY (Energy/Throughput Cost):
                time_penalty = 0.00028 * dt_effective
                
                Penalizes long episodes (throughput and energy).
                - Normal 3600s step: 0.00028 * 3600 ≈ 1.0 penalty
                - With noise, penalizes realistic variation
                - Encourages reaching target radius quickly
                
                Physics-aware: Scales with actual elapsed time (not fixed per step).

            4. TERMINAL REWARD (Episode Completion):
                Applied only when done=True
                
                a) OVERCOARSENING FAILURE (r > R_MAX):
                    Penalty = -100
                    Discourages the most common failure mode
                
                b) MELTING FAILURE (T ≥ 1100):
                    Penalty = -200
                    Strongest penalty: material destroyed
                
                c) SUCCESS (R_MIN ≤ r ≤ R_MAX):
                    bonus = Base(100) + Proximity(100 * exp(-error²/10))
                    Example:
                    - Perfect (error=0): +100 + 100*exp(0) = +200
                    - Close (error=1): +100 + 100*exp(-0.1) ≈ +190
                    - Target range (error=5): +100 + 100*exp(-2.5) ≈ +108
                
                d) OTHER FAILURES:
                    Penalty = -50
                    (e.g., time limit exceeded but radius still too small)

            5. HIGH TEMPERATURE PENALTY (Risk Mitigation):
                If T > 1000 and not done:
                    Extra penalty = (T - 1000) * 0.05 per step
                    
                Discourages prolonged heating near melting point.
                - 1050°C: -2.5 extra penalty
                - 1100°C: -5.0 extra penalty

        Total Reward Examples:

            Optimal step (T=500°C, r approaching target, normal dt):
                step_reward: -0.05 (tiny error)
                temp_penalty: -0.5
                time_penalty: -1.0
                total: -0.05 - 0.5 - 1.0 = -1.55 (negative but well-guided)

            Failed episode (overcoarsened):
                step_reward: -2.0 (error=10nm)
                temp_penalty: -0.8
                time_penalty: -1.0
                terminal_penalty: -100
                total: -103.8 (strong discouragement)

            Successful episode (perfect radius):
                step_reward: 0
                temp_penalty: -0.5
                time_penalty: -1.0
                terminal_bonus: +200
                total: +198.5 (strong encouragement)

        Args:
            done: Whether episode ended
            dt_effective: Actual time elapsed this step (with noise) [seconds]

        Returns:
            Scalar reward value (float)
        """
        # ---- COMPUTE ERROR FROM TARGET ----
        error = abs(self.r - self.r_target)
        
        # ---- STEP REWARD: DENSE SHAPING ----
        # Guide agent toward target radius at each step
        step_reward = -0.1 * error - 0.01 * (error ** 2)

        # ---- TEMPERATURE PENALTY: ENERGY COST ----
        # Discourage excessive heating (energy efficiency)
        temp_penalty = 0.001 * self.T

        # ---- TIME PENALTY: THROUGHPUT COST ----
        # Penalize long episodes using actual elapsed time
        # Weight 0.00028 means ~1.0 penalty per 3600 seconds (1 hour)
        # This makes the penalty scale with realistic timing
        time_penalty = 0.00028 * dt_effective

        # ---- BASE REWARD ----
        reward = step_reward - temp_penalty - time_penalty

        # ---- TERMINAL REWARDS & PENALTIES ----
        if done:
            if self.r > R_MAX:
                # FAILURE: Ostwald ripening - precipitate too coarse
                return reward - 100
            
            if self.T >= 1100:
                # FAILURE: Material melting - catastrophic failure
                return reward - 200
            
            if R_MIN <= self.r <= R_MAX:
                # SUCCESS: Achieved target radius within tolerance
                # Bonus scales with proximity to target (Gaussian centered at target)
                target_reward = 100.0 * np.exp(-(error ** 2) / 10.0)
                # Base success bonus (+100) + proximity bonus (up to +100)
                return reward + target_reward + 100.0
            
            # FAILURE: Ended before reaching target
            # (e.g., time limit expired with radius still too small)
            return reward - 50
        
        # ---- CONTINUOUS PENALTY FOR HIGH TEMPERATURE ----
        # Extra penalty if approaching melting point (even if not done yet)
        if self.T > 1000:
            # Penalty increases linearly above 1000°C
            # At 1100°C: penalty = 5.0 per step
            reward -= (self.T - 1000) * 0.05
        
        # ---- RETURN FINAL REWARD ----
        return reward
