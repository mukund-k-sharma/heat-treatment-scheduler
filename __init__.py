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
    The environment implements a four-phase growth model:
    1. Frozen Phase (<400°C): No growth, zero precipitate growth rate
    2. Growth Phase (400-750°C): Diffusion-controlled growth following Arrhenius equation
    3. Ripening Phase (>750°C): Ostwald ripening with exponential coarsening
    4. Melting (>1100°C): Material destruction

Example:
    >>> from heat_treatment_scheduler import HeatTreatmentSchedulerEnv, HeatTreatmentSchedulerAction
    >>> with HeatTreatmentSchedulerEnv(base_url="http://localhost:8000") as env:
    ...     obs = env.reset()
    ...     action = HeatTreatmentSchedulerAction(action_num=3)  # Increase temp by 10°C
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
