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

Now I am moving away from simple discrete jump T = T + action. Instead, I am implementing
Newton's Law of Cooling combining with Oxidation Insulation Factor. The furnace temperature (T_furnace)
jumps instantly, but the material's core temperature (T_material) follows a continuous differential equation:
dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / m * C_p

where:
    m (mass) is calculated dynamically from the alloy's density_g_cm3
    C_p is pull directly from "materials.json"
    h(t) (Heat Transfer Coefficient) decays over time as surface oxidation builds up at hight 
            temperature, acting as insulator

            
Now we formulate oxidation factor using Arrhenius kinetics as:
    d(ox)/dt = A_ox * exp(-E_ox / (R * T_material)) * (0.8 - ox)

    where:
        A_ox = Pre-exponential factor
        E_ox = Activation energy
        R = Universal Gas constant


        To prevent it from growing to infinity, we multiply it by a saturation term $(0.8 - ox)$, meaning the thicker the oxide layer gets, the slower it grows.


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
    Difficulty level enumeration controlling noise in the environment.
    
    Higher difficulty = more stochasticity in temperature, growth, and timing.
    Useful for curriculum learning: start with EASY, progress to HARD.
    """
    EASY = 1    # Low noise: suitable for learning basics
    MEDIUM = 2  # Moderate noise: realistic variability
    HARD = 3    # High noise: challenging real-world conditions

class HeatTreatmentSchedulerEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    R: float = 8.314 # Universal Gas Constant
    
    def __init__(self, t=0.0, T=20.0, r=0.0, difficulty=AgentGrade.EASY, alloy_key="Al_96_Cu_4", hardware_key="industrial_standard"):

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

        self.density_m3 = self.alloy.density_g_cm3
        self.base_h = self.hardware.base_h

        # Geometric / Thermodynamic Calculation
        # Volumn of cylinder = pi * r^2 * h
        volumn_m3 = np.pi * (self.hardware.radius_m ** 2) * self.hardware.height_m

        # Mass : volumn (m^3) * density (convert g/cm^3 to kg/m^3) 
        self.mass_kg = (self.density_m3 * 1000) * volumn_m3

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
        return HeatTreatmentSchedulerState(
            episode_id=episode_id or str(uuid4()),
            step_count=getattr(self, "step_count", 0),
            time=self.t,
            temperature=self.T_material, # State tracks material core, not furnace air temp
            radius=self.r,
            target_radius=self.r_target
        )


    def _get_obs(self, done=False, reward=0.0) -> HeatTreatmentSchedulerObservation:
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
        Implementing Newton's Law of Cooling combining with Oxidation Insulation Factor. 
        The furnace temperature (T_furnace) jumps instantly, but the material's core temperature (T_material) follows a continuous differential equation:

        dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / m * C_p

        where:
            m (mass) is calculated dynamically from the alloy's density_g_cm3
            C_p is pull directly from "materials.json"
            h(t) (Heat Transfer Coefficient) decays over time as surface oxidation builds up at hight 
                    temperature, acting as insulator

        And

        Physics Model (Four Temperature Regimes):

            1. FROZEN PHASE :
                Atomic diffusion is severely limited. Precipitate growth is essentially
                zero. Atoms lack enough thermal energy to move through the solid matrix.
                Formula: dr/dt = 0
                Physical Interpretation: No growth

            2. CONTROLLED GROWTH PHASE - (0.35 * alloy's melting tempearture):
                Diffusion-controlled growth dominates. Atoms have enough thermal energy
                to diffuse through the matrix but not enough to trigger Ostwald ripening.
                This is the SWEET SPOT for precipitation hardening.
                
                Formula: dr/dt = k(T) * (1 - r / R_max)
                
                Components:
                - k(T) = A * exp(-E / (R * (T + 273.15)))  [Arrhenius equation]
                - (1 - r/R_max) = saturation factor
                
                As r → R_max, the growth slows dramatically (saturation effect).
                This provides natural encouragement for the agent to stop heating near target.

            3. OSTWALD RIPENING PHASE - (0.68 * alloy's melting temperature):
                High-temperature grain coarsening dominates. Larger precipitates grow
                at the expense of smaller ones, leading to loss of mechanical properties.
                This is FAILURE MODE to avoid.
                
                Formula: dr/dt = k(T) * (r / R_max)
                
                The growth rate increases with radius (feedback effect).
                Large precipitates grow faster → material becomes brittle.

            4. MELTING PHASE:
                Material begins to melt. Structure breaks down. Growth rate stops.
                Formula: dr/dt = 0 (but episode terminates immediately)

        Arrhenius Equation Explanation:
            k(T) = A * exp(-E / (R * (T + 273.15)))
            
            - k(T): Temperature-dependent reaction rate [reactions/second]
            - A: Pre-exponential factor (attempt frequency) [reactions/second]
            - E: Activation energy (energy barrier) [J/mol]
            - R: Universal gas constant = 8.314 [J/(mol·K)]
            - T+273.15: Absolute temperature [Kelvin]

        
        These two differential equations are needs to be solved continuously between the time span:
        1. dT_material/dt = h(t) * A_surface * (T_furnace - T_material) / m * C_p
        2. k(T) = A * exp(-E / (R * (T + 273.15)))
            and based on material temperature, we need to calculate dr_dt, which varies:
                Frozen Phase : dr/dt = 0
                Controlled Growth : dr/dt = k(T) * (1 - r/R_max)
                Ostwald ripening : dr/dt = k(T) * (r/R_max)
                melting phase : dr/dt = 0 (and episode terminates)

        
        Now we formulate oxidation factor using Arrhenius kinetics as:
            d(ox)/dt = A_ox * exp(-E_ox / (R * T_material)) * (0.8 - ox)

            where:
                A_ox = Pre-exponential factor
                E_ox = Activation energy
                R = Universal Gas constant


             To prevent it from growing to infinity, we multiply it by a saturation term $(0.8 - ox)$, meaning the thicker the oxide layer gets, the slower it grows.

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
            self.r = solution.y[1][-1]
            self.oxidation_factor = min(0.8, solution.y[2][-1])
        except Exception as e:
            logger.error(f"ODE solver failed : {e}")
            return self._get_obs(done=True, reward=-500) # TODO: Catastrophic physics failure

        # Update time
        self.t += duration_sec
        
        done = self.t >= TIME_MAX or self.r > self.alloy.r_max_clip or self.T_material >= self.alloy.temp_melt
        reward = self._get_reward(done=done, duration_sec=duration_sec)
        self._state = self._get_state(self._state.episode_id)

        return self._get_obs(done=done, reward=reward)


    def _get_reward(self, done=False, duration_sec=0.0):
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
        The continuous differential equations solved by SciPy.
        y[0] = T_material
        y[1] = radius
        y[2] = oxidation_factor
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

        forzen_threshold = self.alloy.temp_melt * 0.35
        ripening_threshold = self.alloy.temp_melt * 0.68

        # Based on material temperature, we need to calculate dr_dt, which varies as:
        #   Frozen Phase : dr/dt = 0
        #   Controlled Growth : dr/dt = k(T) * (1 - r/R_max)
        #   Ostwald ripening : dr/dt = k(T) * (r/R_max)
        #   melting phase : dr/dt = 0 (an episode terminates)

        if T_material < forzen_threshold:
            dr_dt = 0.0
        elif T_material <= ripening_threshold:
            dr_dt = k * (1.0 - (r / self.alloy.r_target_max))
        elif T_material <= self.alloy.temp_melt:
            dr_dt = k * (r / self.alloy.r_target_max)
        else:
            dr_dt = 0.0

        # 3. Orixation Rate (Arrhenius)
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
