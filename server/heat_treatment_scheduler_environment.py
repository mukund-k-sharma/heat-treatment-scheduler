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
    The environment simulates precipitate growth through distinct thermal regimes based on
    the alloy's melting temperature.
    
    The furnace temperature is controlled by the agent, and the material's core temperature
    follows continuous differential equations based on Newton's Law of Cooling, with an
    insulation factor from surface oxidation that builds up at high temperatures.
    
    Precipitate growth follows Arrhenius kinetics, and its rate varies depending on the 
    material's current temperature phase (Frozen, Controlled Growth, Ostwald Ripening, or Melting).

Example Usage:
    >>> env = HeatTreatmentSchedulerEnvironment(
    ...     t=0.0,           # Start at 0 seconds
    ...     T=300.0,         # Start at 300°C
    ...     r=1.0,           # Initial radius 1 nm
    ...     difficulty=AgentGrade.EASY,
    ...     alloy_key="Al_96_Cu_4",
    ...     hardware_key="industrial_standard"
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
from scipy.integrate import solve_ivp

try:
    from ..models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation, HeatTreatmentSchedulerState, ALLOY_REGISTRY, HARDWARE_REGISTRY, TIME_MAX, TIME_MIN
    from ..logging_config import get_logger
except ImportError:
    from models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation, HeatTreatmentSchedulerState, ALLOY_REGISTRY, HARDWARE_REGISTRY, TIME_MAX, TIME_MIN
    from logging_config import get_logger

# Module logger
logger = get_logger(__name__)
logger.debug("Heat Treatment Scheduler environment module loaded")


class AgentGrade(IntEnum):
    """
    Difficulty level enumeration controlling furnace temperature noise (σ_T).
    
    Higher difficulty = more stochasticity in furnace temperature readings.
    Useful for curriculum learning: start with EASY, progress to HARD.
    """
    EASY = 1    # Low noise: suitable for learning basics
    MEDIUM = 2  # Moderate noise: realistic variability
    HARD = 3    # High noise: challenging real-world conditions

class HeatTreatmentSchedulerEnvironment(Environment):
    """
    Heat Treatment Scheduler Environment for precipitation hardening.
    
    This environment simulates the thermodynamics and kinetics of nanoprecipitate growth
    in metal alloys using continuous differential equations. The agent controls the 
    furnace temperature, while the material's core temperature and precipitate radius 
    evolve based on the underlying physics.
    
    Physics Model Details:
    ----------------------
    1. Heat Transfer (Newton's Law of Cooling):
       The furnace temperature (T_furnace) changes instantaneously when the agent takes
       an action, but the material core temperature (T_material) follows a continuous
       differential equation:
       
           dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / (m * C_p)
           
       Where:
         - m (mass) is calculated dynamically from the alloy's density_g_cm3.
         - C_p is the specific heat capacity from materials.json.
         - h(t) is the Heat Transfer Coefficient, which decays over time as surface
           oxidation builds up at high temperatures, acting as an insulator.

    2. Oxidation Kinetics (Arrhenius):
       The oxidation factor (ox) grows based on the material's core temperature:
       
           d(ox)/dt = A_ox * exp(-E_ox / (R * (T_material + 273.15))) * (0.8 - ox)
           
       Where:
         - A_ox = Pre-exponential factor for oxidation.
         - E_ox = Activation energy for oxidation.
         - R = Universal Gas Constant.
       The term (0.8 - ox) acts as a saturation term, capping the insulation effect
       at 80%. As the oxide layer thickens, its growth slows down.

    3. Precipitate Growth (Arrhenius + Phase Thresholds):
       The growth rate depends on the temperature-dependent reaction rate k(T):
       
           k(T) = A * exp(-E / (R * (T_material + 273.15)))
           
       The actual growth rate dr/dt varies depending on the current thermal regime,
       which is defined relative to the alloy's melting temperature (T_melt):
       
       a) Frozen Phase (T < 0.35 * T_melt):
          Atomic diffusion is severely limited. Material remains in initial microstructure.
          dr/dt = 0
          
       b) Controlled Growth Phase (0.35 * T_melt <= T <= 0.68 * T_melt):
          Diffusion-controlled growth dominates. Rate-limiting factor: atomic diffusion in the solid. This is the optimal window.
          dr/dt = k(T) * (1 - r/R_max)
          The (1 - r/R_max) term is a saturation factor. As the radius approaches
          R_max, growth slows dramatically, allowing the agent to "park" at the target.
          
       c) Ostwald Ripening Phase (0.68 * T_melt < T <= T_melt):
          High-temperature grain coarsening dominates. Material becomes brittle and loses mechanical properties. Larger precipitates grow at
          the expense of smaller ones. This is a failure mode.
          dr/dt = k(T) * (r/R_max)
          
       d) Melting Phase (T > T_melt):
          Material breaks down. Crystalline structure dissolves. Episode terminates.
          dr/dt = 0
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    R: float = 8.314 # Universal Gas Constant
    
    def __init__(self, t=0.0, T=20.0, r=0.0, difficulty=AgentGrade.EASY, alloy_key="Al_96_Cu_4", hardware_key="industrial_standard"):
        """
        Initializes the environment state and dynamically loads physical properties.
        
        Args:
            t: Initial elapsed time (seconds).
            T: Initial furnace and material temperature (°C).
            r: Initial precipitate radius (nanometers).
            difficulty: Controls environmental stochasticity via AgentGrade.
            alloy_key: Key for the metal alloy to simulate. Must match an entry in
                       `materials.json` (e.g., "Al_96_Cu_4", "Ti_6Al_4V"). This dynamically
                       loads all intrinsic physical properties (melting temp, density, etc.).
            hardware_key: Key for the hardware setup to simulate. Must match an entry in
                          `hardware.json` (e.g., "industrial_standard", "lab_scale"). This
                          dynamically loads extrinsic geometric and thermodynamic properties.
        """

        logger.info(f"Initializing v2 environment: Alloy : {alloy_key}, Hardware: {hardware_key}")

        # Load alloy properties dynamically
        if alloy_key not in ALLOY_REGISTRY:
            raise ValueError(f"Alloy {alloy_key} not found in registry.")
        # Load lab/hardware properties dynamically
        if hardware_key not in HARDWARE_REGISTRY:
            raise ValueError(f"Hardware setup {hardware_key} not found in registry.")
        
        self.alloy = ALLOY_REGISTRY[alloy_key]
        self.hardware = HARDWARE_REGISTRY[hardware_key]

        # Physics constants from Registry
        self.A = self.alloy.A
        self.E = self.alloy.E
        self.C_p = self.alloy.specific_heat_capacity

        self.density_g_cm3 = self.alloy.density_g_cm3
        self.base_h = self.hardware.base_h

        # Geometric / Thermodynamic Calculation
        # Volume of cylinder = pi * r^2 * h
        volume_m3 = np.pi * (self.hardware.radius_m ** 2) * self.hardware.height_m

        # Mass : volume (m^3) * density (convert g/cm^3 to kg/m^3) 
        self.mass_kg = (self.density_g_cm3 * 1000) * volume_m3

        # Surface area: 2 * pi * r * (r + h)
        self.surface_area_m2 = 2 * np.pi * self.hardware.radius_m * (self.hardware.radius_m + self.hardware.height_m)

        # Average radius, based on alloy's min and max radius
        self.r_target = (self.alloy.r_target_min + self.alloy.r_target_max) / 2.0

        # Noise config
        self.difficulty = difficulty
        self.sigma_T = {
            AgentGrade.EASY: 1.0,
            AgentGrade.MEDIUM: 2.0,
            AgentGrade.HARD: 3.0
        }[difficulty]

        # Initial state information
        self.init_t = t
        self.init_T_furnace = T
        self.init_T_material = T
        self.init_r = r
        self._reset_count = 0

        self.reset()

    
    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: Any):
        """
        Reset the environment to its initial state for a new episode.

        Restores all state variables (time, temperatures, radius, oxidation) to their
        initial values and returns the initial normalized observation.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Optional UUID string to identify this episode.
            **kwargs: Additional keyword arguments (for interface compatibility).

        Returns:
            HeatTreatmentSchedulerObservation with initial state (done=False, reward=0.0).
        """
        # =============== SEED =================
        if seed is not None:
            self.SEED = seed
            np.random.seed(self.SEED)
            random.seed(self.SEED)
        # ======================================
        
        self._reset_count += 1
        self.step_count = 0

        self.t = self.init_t
        self.T_furnace = self.init_T_furnace
        self.T_material = self.init_T_material
        self.r = self.init_r
        self.oxidation_factor = 0.0 # Builds up over time at high heat

        self._state = self._get_state(episode_id)

        return self._get_obs(done=False, reward=0.0)

    
    def _get_state(self, episode_id : str | None = None) -> HeatTreatmentSchedulerState:
        """Construct a raw (unnormalized) state snapshot for UI, logging, and debugging."""
        return HeatTreatmentSchedulerState(
            episode_id=episode_id or str(uuid4()),
            step_count=getattr(self, "step_count", 0),
            time=self.t,
            temperature=self.T_material, # State tracks material core, not furnace air temp
            radius=self.r,
            target_radius=self.r_target
        )


    def _get_obs(self, done=False, reward=0.0) -> HeatTreatmentSchedulerObservation:
        """
        Build a normalized observation for the agent.

        All values are normalized to [0, 1] (or [-1, 1] for radius_error) for
        stable neural network training. Also computes the categorical temperature
        phase indicator (Frozen=0, Growth=1, Ripening=2).

        Args:
            done: Whether the episode has ended.
            reward: The reward for the current step.

        Returns:
            HeatTreatmentSchedulerObservation with normalized state values.
        """
        # Determine temperature phase based on alloy melting point
        if self.T_material < (self.alloy.temp_melt * 0.35):
            phase = 0.0  # Frozen: no growth
        elif self.T_material <= (self.alloy.temp_melt * 0.68):
            phase = 1.0  # Growth: controlled growth phase
        else:
            phase = 2.0  # Ripening: ostwald ripening (dangerous)

        # Normalize time: t / TIME_MAX
        normalized_time = self.t / TIME_MAX

        # Construct and return normalized observation
        return HeatTreatmentSchedulerObservation(
            time=normalized_time,
            temperature=self.T_material / self.alloy.temp_max,
            radius=self.r / self.alloy.r_max_clip,
            target_radius=self.r_target / self.alloy.r_max_clip,
            radius_error=(self.r - self.r_target) / self.alloy.r_max_clip,
            temperature_phase=phase,
            remaining_time=max(0.0, (1 - normalized_time)),
            done=done,
            reward=reward
        )


    def step(self, action : HeatTreatmentSchedulerAction, timeout_s: float | None = None, **kwargs: Any) -> HeatTreatmentSchedulerObservation:
        """
        Executes a single time step in the environment.
        
        This method applies the chosen temperature change action to the furnace,
        and advances the simulation by the action's specified duration. The
        underlying physics (heat transfer and precipitate growth) are simulated
        continuously over this duration using SciPy's ODE solver.
        
        Args:
            action: The action to take, specifying the temperature change and duration.
            timeout_s: Optional timeout for the step execution.
            **kwargs: Additional keyword arguments.
            
        Returns:
            An observation containing the new state of the environment.
        """

        self.step_count += 1

        if action.action_num == HeatTreatmentSchedulerAction.TERMINATION_SIGNAL:
            return self._get_obs(done=True, reward=self._get_reward(done=True, duration_sec=0.0))

        dT_furnace = HeatTreatmentSchedulerAction.ACTION_MAP.get(action.action_num)
        duration_sec = action.duration_minutes * 60.0

        # Update Furnace Air temperature (instantaneous + noise)
        noise_T = np.random.normal(0, self.sigma_T)
        self.T_furnace = np.clip(self.T_furnace + dT_furnace + noise_T, self.init_T_furnace, self.alloy.temp_max) # type: ignore
        

        # Solving differential equation over the duration:
        t_span = (self.t, self.t + duration_sec)
        y0 = [self.T_material, self.r, self.oxidation_factor]

        try:
            solution = solve_ivp(
                fun=self._physics_derivatives,
                t_span=t_span,
                y0=y0,
                method='RK45', # Explicit Runge-Kutta method of order 5(4) 
                max_step=120  # solve it every 2 min (120 s)
            )

            # Extract final state
            self.T_material = solution.y[0][-1]
            self.r = max(0.0, solution.y[1][-1])  # Clamp: ODE solver numerical noise can produce tiny negatives
            self.oxidation_factor = min(0.8, solution.y[2][-1])
        except Exception as e:
            logger.error(f"ODE solver failed : {e}")
            return self._get_obs(done=True, reward=-500)

        # Update time
        self.t += duration_sec
        
        done = self.t >= TIME_MAX or self.r > self.alloy.r_max_clip or self.T_material >= self.alloy.temp_melt
        reward = self._get_reward(done=done, duration_sec=duration_sec)
        self._state = self._get_state(self._state.episode_id)

        return self._get_obs(done=done, reward=reward)


    def _get_reward(self, done=False, duration_sec=0.0):
        """
        Compute the dense, multi-component reward for the current step.

        Components:
        1. Proximity shaping: -0.1·|error| - 0.01·error² (always applied)
        2. Energy penalty: -0.001·T_material·(Δt/3600)
        3. Time penalty: -0.00028·Δt
        4. Warning zone: extra penalty when T_material > T_melt - 100°C

        Terminal bonuses/penalties:
        - Success (r in target window): +100 + Gaussian proximity bonus
        - Over-coarsened (r > r_target_max): -100
        - Melted (T >= T_melt): -200
        - Other (timeout, etc.): -50

        Args:
            done: Whether the episode is ending on this step.
            duration_sec: Duration of the action in seconds.

        Returns:
            float: The scalar reward value.
        """
        error = abs(self.r - self.r_target)
        step_reward = -0.1 * error - 0.01 * (error ** 2)

        temp_penalty = 0.001 * self.T_material * (duration_sec / 3600.0)
        time_penalty = 0.00028 * duration_sec

        reward = step_reward - temp_penalty - time_penalty

        if done:
            if self.r > self.alloy.r_target_max:
                return reward - 100
            
            if self.T_material >= self.alloy.temp_melt:
                return reward - 200
            
            if self.alloy.r_target_min <= self.r <= self.alloy.r_target_max:
                target_reward = 100.0 * np.exp(-(error ** 2) / 10.0)
                return reward + target_reward + 100.0

            return reward - 50
        
    
        warning_temp = self.alloy.temp_melt - 100.0
        if self.T_material > warning_temp:
            reward -= (self.T_material - warning_temp) * 0.05 * (duration_sec / 3600.0)

        return reward
    

    def _physics_derivatives(self, t, y):
        """
        Calculates the continuous differential equations for the simulation.
        
        This function is solved by SciPy over the duration of a step. It computes
        the rate of change for material temperature, precipitate radius, and
        oxidation factor.
        
        Physics Model:
        1. Heat Transfer (Newton's Law of Cooling):
           dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / (m * C_p)
           where h(t) decays as oxidation builds up.
           
        2. Precipitate Growth (Arrhenius Equation):
           k(T) = A * exp(-E / (R * (T_material + 273.15)))
           - Frozen Phase (T < 0.35 * T_melt): dr/dt = 0
           - Controlled Growth (T <= 0.68 * T_melt): dr/dt = k(T) * (1 - r/R_max)
           - Ostwald Ripening (T <= T_melt): dr/dt = k(T) * (r/R_max)
           - Melting Phase (T > T_melt): dr/dt = 0
           
        3. Oxidation Rate:
           d(ox)/dt = A_ox * exp(-E_ox / (R * (T_material + 273.15))) * (0.8 - ox)
           Capped at 80% insulation.

        Args:
            t: Current time in seconds.
            y: Current state vector [T_material, radius, oxidation_factor].
            
        Returns:
            List of derivatives [dT_material/dt, dr/dt, d_ox/dt].
        """

        T_material, r, ox = y

        # 1. Heat Transfer (Newton's Law of Cooling) - Furnace
        # dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / m * C_p
        h_effective = self.base_h * (1.0 - ox)
        thermal_rate = (h_effective * self.surface_area_m2) / (self.mass_kg * self.C_p)
        dT_mat_dt = thermal_rate * (self.T_furnace - T_material)

        # 2. Precipitate growth (Arrhenius Equation)
        # k(T) = A * exp(-E / (R * (T + 273.15)))
        k = self.A * np.exp(-self.E / (self.R * (T_material + 273.15)))

        frozen_threshold = self.alloy.temp_melt * 0.35
        ripening_threshold = self.alloy.temp_melt * 0.68

        # Based on material temperature, we need to calculate dr_dt, which varies as:
        #   Frozen Phase : dr/dt = 0
        #   Controlled Growth : dr/dt = k(T) * (1 - r/R_max)
        #   Ostwald ripening : dr/dt = k(T) * (r/R_max)
        #   melting phase : dr/dt = 0 (an episode terminates)

        if T_material < frozen_threshold:
            dr_dt = 0.0
        elif T_material <= ripening_threshold:
            dr_dt = k * (1.0 - (r / self.alloy.r_target_max))
        elif T_material <= self.alloy.temp_melt:
            dr_dt = k * (r / self.alloy.r_target_max)
        else:
            dr_dt = 0.0

        # 3. Oxidation Rate (Arrhenius)
        # Calculates how fast the insulating layer builds based on current core temperature
        # d(ox)/dt = A_ox * exp(-E_ox / (R * (T_material + 273.15))) * (0.8 - ox)
        k_ox = self.alloy.A_ox * np.exp(-self.alloy.E_ox / (self.R * (T_material + 273.15)))

        # Saturation: THe oxide layer caps at 0.8 (80% insulation)
        d_ox_dt = k_ox * (0.8 - ox) if ox < 0.8 else 0.0

        return [dT_mat_dt, max(dr_dt, 0.0), max(d_ox_dt, 0.0)]
    
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
