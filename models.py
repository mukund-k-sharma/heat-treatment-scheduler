# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Heat Treatment Scheduler Environment.

This module defines the observation and action spaces, as well as boundary conditions
for the heat treatment scheduling task. It includes Pydantic models for type safety
and validation.

Physics Context:
    The heat treatment process controls nanoprecipitate growth in metal alloys through
    precise temperature management. The environment tracks three coupled state variables:
    - Time (t): Elapsed time in the furnace (seconds)
    - Temperature (T_material): Core temperature of the material (Celsius), which lags
      behind the furnace air temperature due to thermal inertia
    - Radius (r): Average radius of nanoprecipitates (nanometers)

    The physics is governed by four thermal regimes relative to the alloy's melting point.
"""

import json
from pathlib import Path

from pydantic import BaseModel
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import ClassVar, Dict

try:
    from .logging_config import get_logger
except (ImportError, ValueError, SystemError):
    from logging_config import get_logger

# Module logger
logger = get_logger(__name__)
logger.debug("Heat Treatment Scheduler models module loaded")


# ======================== MATERIAL REGISTRY ========================

class AlloyProperties(BaseModel):
    """
    Defines the physical constants and boundary limits for a specific material
    """
    name: str = Field(description="Display name of the alloy")
    composition_wt: Dict[str, float] = Field(description="Weight percentage of elemental composition")
    density_g_cm3: float = Field(description="Material density in g/cm^3")
    specific_heat_capacity : float = Field(description="Specific heat capacity (C_p) in J/(kg * K)")
    A: float = Field(description="Pre-exponential factor for Arrhenius equation (reactions/s)")
    E: float = Field(description="Activation energy for Arrhenius equation (J/mol)")
    temp_melt: float = Field(description="Melting temperature in Celsius (catastrophic failure point)")
    temp_max : float = Field(description="Absolute max temperature for neural network normalization")
    r_target_min: float = Field(description="Minimum target radius for success (nm)")
    r_target_max : float = Field(description="Maximum target radius for success (nm)")
    r_max_clip: float = Field(description="Absolute max radius for neural network normalization")

    # Oxidation kinetics
    A_ox: float = Field(description="Pre-exponential factor for Oxidation (1/s)")
    E_ox: float = Field(description="Activation energy for Oxidation (J/mol).")


def load_alloy_registry() -> Dict[str, AlloyProperties]:
    """
    Loads and validates the materials configuration file (`materials.json`).
    
    This configuration file contains the physical properties (density, specific heat, 
    melting temperature, Arrhenius constants, oxidation constants) for various 
    metal alloys, allowing the environment to simulate different materials dynamically.
    """

    config_path = Path(__file__).parent / "materials.json"

    try:
        with open(config_path, "r") as f:
            data = json.load(f)

        registry = {}

        # Parsing: Series --> AlloyKey --> Properties
        for series_name, alloys in data.items():
            for key, props in alloys.items():
                registry[key] = AlloyProperties(**props)

        logger.info(f"Successfully loaded {len(registry)} materials into the registry.")
        return registry
    except FileNotFoundError:
        logger.error(f"Could not find materials config at {config_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to parse materials config : {e}")
        raise



# Initialize the global registry
ALLOY_REGISTRY = load_alloy_registry()

#========================= HARDWARE PROPERTIES ===============================

class HardwareProperties(BaseModel):
    """
    Define the extrinsic geometric and thermodynamic properties of the setup.
    """
    # Geometric properties (billet)
    name : str = Field(description="Display name of hardware setup.")
    radius_m: float = Field(description="Radius of the cylindrical billet in meters.", gt=0.0)
    height_m: float = Field(description="Height of cylindrical billet in meters", gt=0.0)

    # Thermodynamic properties (billet)
    base_h: float = Field(description="Base convective heat transfer coefficient. (W/m^2 * K)", gt=0.0)

    # General
    description: str = Field(description="Contextual description of the thermal dynamics.")


def load_hardware_registry() -> Dict[str, HardwareProperties]:
    """
    Loads and validates the hardware configuration file (`hardware.json`).
    
    This configuration file defines different furnace geometries and properties 
    (like radius, height, and the base convective heat transfer coefficient `base_h`), 
    allowing the environment to simulate various real-world hardware setups ranging 
    from small lab samples to massive industrial castings.
    """
    config_path = Path(__file__).parent / "hardware.json"

    try:
        with open(config_path, "r") as f:
            data = json.load(f)

        return {k: HardwareProperties(**v) for k, v in data.items()}
    except Exception as e:
        logger.error(f"Failed to parse hardware config: {e}")
        raise


HARDWARE_REGISTRY = load_hardware_registry()


# ======================== GLOBAL BOUNDARY CONDITIONS ========================
# These constants define the valid ranges for the heat treatment process.
# Values outside these ranges result in episode termination or state clipping.

TIME_MIN: float = 0.0
"""Minimum elapsed time (seconds). Typically unused as time only increases."""

TIME_MAX: float = 180_000.0 # 50 hours
"""
Maximum allowed time in oven: 50 hours = 180,000 seconds.
Episode ends if time exceeds this value.
"""


# ======================== ACTION SPACE ========================

class HeatTreatmentSchedulerAction(Action):
    """
    Action space for the Heat Treatment Scheduler environment.
    
    The agent controls the oven by selecting discrete temperature changes.
    Six possible actions representing different heating strategies.
    
    Attributes:
        action_num: Integer 0-5 selecting one of six discrete actions.
                   This is the primary interface for agent-environment interaction.
    
    Raises:
        ValueError: If action_num is not in range [0, 5]
    """
    
    # Instance variable holding the selected action
    action_num: int = Field(description="0-5 discrete temperature control actions", le=5)
    # Hold a temperature: min = 1 minute (micro-adjustments) and max: 10 hours per step
    duration_minutes: float = Field(default=60.0, description="Duration to hold this temperature state in minutes.", ge=1.0, le=600.0)

    
    # ---- Action Mapping ----
    # Maps action_num to temperature change (dT in °C)
    ACTION_MAP : ClassVar[dict[int, int | None]] = {
        0: -50,  # Aggressive cooling: decrease oven temperature by 50°C
        1: -10,  # Gentle cooling: decrease oven temperature by 10°C
        2: 0,    # Maintain: hold the current temperature steady
        3: 10,   # Gentle heating: increase oven temperature by 10°C
        4: 50,   # Aggressive heating: increase oven temperature by 50°C
        5: None  # Termination: end the episode (no temperature change)
    }
    
    TERMINATION_SIGNAL: ClassVar[int] = 5
    """Agent action to signal end of episode. Used for graceful termination."""
    
    # ---- Physics Context ----
    # Temperature regimes and their effects on precipitate growth:
    #
    # Below 0.35 * temp_melt (Frozen Phase):
    #   - Atomic diffusion is negligible
    #   - Precipitate radius remains essentially constant
    #   - Growth rate dr/dt ≈ 0
    #
    # 0.35 * temp_melt to 0.68 * temp_melt (Controlled Growth Phase):
    #   - Diffusion-controlled growth regime
    #   - Growth rate increases with temperature (Arrhenius equation)
    #   - dr/dt = k(T) * (1 - r/R_max)
    #   - Growth slows as radius approaches R_max (saturation effect)
    #   - **SWEET SPOT**: Agent can reach target radius here
    #
    # 0.68 * temp_melt to temp_melt (Ostwald Ripening Phase):
    #   - High-temperature grain coarsening begins
    #   - Larger precipitates grow at expense of smaller ones
    #   - Growth rate: dr/dt = k(T) * (r/R_max)
    #   - Material becomes brittle and overcoarsened
    #   - Episode fails if radius exceeds R_max
    #
    # Above temp_melt (Melting/Destruction Phase):
    #   - Material begins melting and crystalline structure breaks down
    #   - Large negative reward and episode termination
    #   - Represent catastrophic process failure

# ======================== OBSERVATION SPACE ========================

class HeatTreatmentSchedulerObservation(Observation):
    """
    Observation returned by the environment at each step.
    
    The environment is fully observable: the agent receives all relevant state
    information. All values are normalized to [0, 1] range for stable neural network
    training. Normalization improves convergence and gradient stability.
    
    Attributes:
        time: Normalized elapsed time (t / TIME_MAX). Range: [0, 1]
        temperature: Normalized material core temperature (T / alloy.temp_max). Range: [0, 1]
        radius: Normalized current precipitate radius (r / alloy.r_max_clip). Range: [0, 1]
        target_radius: Normalized target radius (r_target / alloy.r_max_clip). Range: [0, 1]
        radius_error: Deviation from target: (r - r_target) / alloy.r_max_clip. Range: [-1, 1]
        temperature_phase: Categorical phase indicator. Values:
            - 0.0: Frozen phase (T < 0.35 * temp_melt) - no growth
            - 1.0: Growth phase (0.35 * temp_melt ≤ T ≤ 0.68 * temp_melt) - controlled growth
            - 2.0: Ripening phase (T > 0.68 * temp_melt) - dangerous coarsening
        remaining_time: Normalized time left: (TIME_MAX - t) / TIME_MAX. Range: [0, 1]
        done: Boolean flag. True if episode ended (success or failure).
        reward: Scalar reward value. Shaped to guide agent toward optimal policy.
        metadata: Dictionary for additional information (not directly used in training).
    """
    
    # Core state observations (normalized to [0, 1])
    time: float = Field(description="Normalized elapsed time in oven: t / TIME_MAX")
    temperature: float = Field(description="Normalized material core temperature: T / alloy.temp_max")
    radius: float = Field(description="Normalized average radius of nanoprecipitates: r / alloy.r_max_clip")
    target_radius: float = Field(description="Normalized target precipitate radius: r_target / alloy.r_max_clip")
    radius_error: float = Field(description="Distance from target (normalized): (r - r_target) / alloy.r_max_clip")
    
    # Phase indicator for interpretability
    temperature_phase: float = Field(
        default=0,
        description=(
            "Temperature regime indicator (categorical as float):\n"
            "  0.0 = Frozen phase (T < 0.35 * temp_melt)\n"
            "  1.0 = Growth phase (0.35 * temp_melt ≤ T ≤ 0.68 * temp_melt)\n"
            "  2.0 = Ripening phase (T > 0.68 * temp_melt)"
        )
    )
    
    remaining_time: float = Field(description="Normalized time remaining: (1 - (t / TIME_MAX))")

# ======================== STATE SPACE ========================

class HeatTreatmentSchedulerState(State):
    """
    Raw, unnormalized environment state for UI, physics calculations, and debugging.
    
    While observations are normalized for neural network training, internal state
    is maintained in raw physical units. This separation allows:
    - Accurate physics simulations
    - Proper application of constraints and boundaries
    - Clear interpretation for human visualization
    - Efficient numerical computation
    
    Attributes:
        time: Elapsed time in seconds. Range: [TIME_MIN, TIME_MAX]
        temperature: Material core temperature in degrees Celsius.
        radius: Current precipitate radius in nanometers. Range: [0, ∞)
        target_radius: Desired precipitate radius in nanometers.
    """
    
    time: float = Field(description="Current elapsed time in the oven (seconds)")
    temperature: float = Field(description="Current material core temperature (degrees Celsius)")
    radius: float = Field(description="Current average radius of nanoprecipitates (nanometers)", ge=0.0) # Radius must be non-negative 
    target_radius: float = Field(description="Target radius for nanoprecipitates (nanometers)")
