# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Heat Treatment Scheduler Environment Package.

This package provides a reinforcement learning environment for simulating and optimizing
the heat treatment process of materials, specifically focusing on precipitation hardening.

The environment simulates the growth of nanoprecipitates in a metal alloy through controlled
heating. The goal is for an agent to learn optimal temperature control strategies to achieve
a target precipitate size while managing time and energy constraints.

Core Components:
    - HeatTreatmentSchedulerEnv: Client interface for interacting with the environment
    - HeatTreatmentSchedulerAction: Action space definition (temperature adjustments)
    - HeatTreatmentSchedulerObservation: Observation space (current state metrics)

Physics Model:
    The environment implements a four-phase growth model relative to the alloy's melting 
    point (T_melt), which includes continuous thermodynamics:
    1. Frozen Phase (<0.35 * T_melt): No growth, atomic diffusion is negligible.
    2. Growth Phase (0.35-0.68 * T_melt): Diffusion-controlled growth following Arrhenius equation.
    3. Ripening Phase (0.68-1.0 * T_melt): Ostwald ripening with grain coarsening failure.
    4. Melting Phase (>= T_melt): Material destruction and catastrophic failure.

    Additionally, the model simulates thermal mass (heating/cooling lag) via Newton's Law 
    of Cooling and insulation buildup via oxidation kinetics.

Example:
    >>> from heat_treatment_scheduler import HeatTreatmentSchedulerEnv, HeatTreatmentSchedulerAction
    >>> with HeatTreatmentSchedulerEnv(base_url="http://localhost:8000") as env:
    ...     obs = env.reset()
    ...     # Increase temp by 10°C and hold for 60 minutes
    ...     action = HeatTreatmentSchedulerAction(action_num=3, duration_minutes=60.0) 
    ...     result = env.step(action)
"""

try:
    from .logging_config import get_logger
    from .client import HeatTreatmentSchedulerEnv
    from .models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation
except (ImportError, ValueError, SystemError):
    from logging_config import get_logger
    from client import HeatTreatmentSchedulerEnv
    from models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation

# Module logger
logger = get_logger(__name__)
logger.debug("Heat Treatment Scheduler package initialized")

__all__ = [
    "HeatTreatmentSchedulerAction",
    "HeatTreatmentSchedulerObservation",
    "HeatTreatmentSchedulerEnv",
    "get_logger",
]
