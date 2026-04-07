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
    precise temperature management. The environment tracks three key variables:
    - Time (t): Elapsed time in the oven (seconds)
    - Temperature (T): Current oven temperature (Celsius)
    - Radius (r): Average radius of nanoprecipitates (nanometers)

    The physics is governed by four regimes based on temperature.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import ClassVar

from .logging_config import get_logger

# Module logger
logger = get_logger(__name__)
logger.debug("Heat Treatment Scheduler models module loaded")

# ======================== BOUNDARY CONDITIONS ========================
# These constants define the valid ranges for the heat treatment process.
# Values outside these ranges result in episode termination or state clipping.

TIME_MIN: float = 0.0
"""Minimum elapsed time (seconds). Typically unused as time only increases."""

TIME_MAX: float = 180_000.0
"""
Maximum allowed time in oven: 50 hours = 180,000 seconds.
Episode ends if time exceeds this value.
"""

TEMP_MIN: float = 0.0
"""Minimum oven temperature (Celsius). Temperature is clipped to this lower bound."""

TEMP_MAX: float = 1_200.0
"""
Maximum oven temperature (Celsius). Temperature is clipped to this upper bound.
Critical: Above ~1100°C, material begins melting and is destroyed.
"""

R_MIN: float = 10.0
"""Minimum target precipitate radius (nanometers). Success range lower bound."""

R_MAX: float = 15.0
"""
Maximum target precipitate radius (nanometers). Success range upper bound.
Episode ends if radius exceeds this value (Ostwald ripening failure).
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
    action_num: int = Field(
        ...,
        description="0-5 discrete temperature control actions",
        ge=0,
        le=5
    )
    
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
    # Below 400°C (Frozen Phase):
    #   - Atomic diffusion is negligible
    #   - Precipitate radius remains essentially constant
    #   - Growth rate dr/dt ≈ 0
    #
    # 400°C to 750°C (Controlled Growth Phase):
    #   - Diffusion-controlled growth regime
    #   - Growth rate increases with temperature (Arrhenius equation)
    #   - dr/dt = k(T) * (1 - r/R_max)
    #   - Growth slows as radius approaches R_max (saturation effect)
    #   - **SWEET SPOT**: Agent can reach target radius here
    #
    # 750°C to 1100°C (Ostwald Ripening Phase):
    #   - High-temperature grain coarsening begins
    #   - Larger precipitates grow at expense of smaller ones
    #   - Growth rate: dr/dt = k(T) * (r/R_max)
    #   - Material becomes brittle and overcoarsened
    #   - Episode fails if radius exceeds R_max
    #
    # Above 1100°C (Melting/Destruction Phase):
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
        temperature: Normalized oven temperature (T / TEMP_MAX). Range: [0, 1]
        radius: Normalized current precipitate radius (r / R_MAX). Range: [0, 1]
        target_radius: Normalized target radius (r_target / R_MAX). Range: [0, 1]
        radius_error: Deviation from target: (r - r_target) / R_MAX. Range: [-1, 1]
        temperature_phase: Categorical phase indicator. Values:
            - 0.0: Frozen phase (T < 400°C) - no growth
            - 1.0: Growth phase (400 ≤ T ≤ 750°C) - controlled growth
            - 2.0: Ripening phase (T > 750°C) - dangerous coarsening
        remaining_time: Normalized time left: (TIME_MAX - t) / TIME_MAX. Range: [0, 1]
        done: Boolean flag. True if episode ended (success or failure).
        reward: Scalar reward value. Shaped to guide agent toward optimal policy.
        metadata: Dictionary for additional information (not directly used in training).
    """
    
    # Core state observations (normalized to [0, 1])
    time: float = Field(
        ...,
        description="Normalized elapsed time in oven: t / TIME_MAX"
    )
    
    temperature: float = Field(
        ...,
        description="Normalized oven temperature: T / TEMP_MAX"
    )
    
    radius: float = Field(
        ...,
        description="Normalized average radius of nanoprecipitates: r / R_MAX"
    )
    
    # Goal and error tracking
    target_radius: float = Field(
        ...,
        description="Normalized target precipitate radius: r_target / R_MAX"
    )
    
    radius_error: float = Field(
        ...,
        description="Distance from target (normalized): (r - r_target) / R_MAX"
    )
    
    # Phase indicator for interpretability
    temperature_phase: float = Field(
        default=0,
        description=(
            "Temperature regime indicator (categorical as float):\n"
            "  0.0 = Frozen phase (T < 400°C)\n"
            "  1.0 = Growth phase (400°C ≤ T ≤ 750°C)\n"
            "  2.0 = Ripening phase (T > 750°C)"
        )
    )
    
    # Time pressure signal
    remaining_time: float = Field(
        ...,
        description="Normalized time remaining: (TIME_MAX - t) / TIME_MAX"
    )

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
        temperature: Oven temperature in degrees Celsius. Range: [TEMP_MIN, TEMP_MAX]
        radius: Current precipitate radius in nanometers. Range: [0, ∞)
        target_radius: Desired precipitate radius in nanometers.
    """
    
    time: float = Field(
        ...,
        description="Current elapsed time in the oven (seconds)"
    )
    
    temperature: float = Field(
        ...,
        description="Current oven temperature (degrees Celsius)"
    )
    
    radius: float = Field(
        ...,
        description="Current average radius of nanoprecipitates (nanometers)",
        ge=0.0  # Radius must be non-negative
    )
    
    target_radius: float = Field(
        ...,
        description="Target radius for nanoprecipitates (nanometers)"
    )
