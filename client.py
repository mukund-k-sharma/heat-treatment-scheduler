# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Heat Treatment Scheduler Environment Client.

This module provides the client-side interface for interacting with the
HeatTreatmentSchedulerEnvironment running on a remote server. It handles
WebSocket communication, action encoding, and observation decoding.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation
    from .logging_config import get_logger
except (ImportError, ValueError, SystemError):
    from models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation
    from logging_config import get_logger

# Module logger
logger = get_logger(__name__)


class HeatTreatmentSchedulerEnv(
    EnvClient[HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation, State]
):
    """
    Client for interacting with the Heat Treatment Scheduler Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency. Each client instance
    has its own dedicated environment session on the server.

    The environment simulates the heat treatment process of a material alloy, where the
    agent learns to control oven temperature to achieve a target precipitate size. The
    client handles all communication details, allowing the user to focus on the agent logic.

    Initialization:
        Can be initialized in two ways:
        1. With a server URL: HeatTreatmentSchedulerEnv(base_url="http://localhost:8000")
        2. Using Docker: HeatTreatmentSchedulerEnv.from_docker_image("heat_treatment_scheduler_env:latest")

    Example with server:
        >>> # Connect to a running server
        >>> with HeatTreatmentSchedulerEnv(base_url="http://localhost:8000") as client:
        ...     # Reset the environment to get initial observation
        ...     result = client.reset()
        ...     obs = result.observation
        ...
        ...     # Take an action (e.g., increase temperature by 10°C)
        ...     result = client.step(HeatTreatmentSchedulerAction(action_num=3))
        ...     print(f"Reward: {result.reward}, Done: {result.done}")

    Example with Docker:
        >>> # Automatically start Docker container and connect
        >>> client = HeatTreatmentSchedulerEnv.from_docker_image(\"heat_treatment_scheduler_env:latest\")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(HeatTreatmentSchedulerAction(action_num=2))  # Hold temp
        ... finally:
        ...     client.close()  # Cleanup resources

    Context Manager:
        The client works as a context manager (with statement) for automatic
        resource cleanup. Recommended for production code.
    """

    def _step_payload(self, action: HeatTreatmentSchedulerAction) -> Dict:
        """
        Convert HeatTreatmentSchedulerAction to JSON payload for step message.

        This method serializes the action object into a dictionary suitable for
        JSON encoding when sending to the server. It extracts the discrete action
        number and maps it to the corresponding temperature change.

        Args:
            action: HeatTreatmentSchedulerAction instance containing action_num (0-5)

        Returns:
            Dictionary with structure:
            {
                "action_num": <int 0-5>
            }
            
        Example:
            >>> action = HeatTreatmentSchedulerAction(action_num=3)  # +10°C
            >>> payload = client._step_payload(action)
            >>> assert payload == {"action_num": 3}
        """
        payload = {
            "action_num": action.action_num,
        }
        logger.debug(f"Step payload created: action={action.action_num}")
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[HeatTreatmentSchedulerObservation]:
        """
        Parse server response into StepResult[HeatTreatmentSchedulerObservation].

        This method deserializes the JSON response from the server step() call
        and constructs the StepResult object with normalized observation values.
        All observation values come pre-normalized from the server.

        Server Response Format:
            {
                "observation": {
                    "time": <float 0-1>,
                    "temperature": <float 0-1>,
                    "radius": <float 0-1>,
                    "target_radius": <float 0-1>,
                    "radius_error": <float -1 to 1>,
                    "temperature_phase": <float 0-2>,
                    "remaining_time": <float 0-1>,
                    "metadata": {<dict>}
                },
                "reward": <float>,
                "done": <bool>
            }

        Args:
            payload: JSON response data from server step() endpoint

        Returns:
            StepResult with populated HeatTreatmentSchedulerObservation and metrics
            
        Raises:
            KeyError: If required fields are missing from payload
            ValueError: If observation/reward values are invalid
        """
        obs_data = payload.get("observation", {})
        observation = HeatTreatmentSchedulerObservation(
            time=obs_data.get("time", 0.0),
            temperature=obs_data.get("temperature", 0.0),
            radius=obs_data.get("radius", 0.0),
            target_radius=obs_data.get("target_radius", 0.0),
            radius_error=obs_data.get("radius_error", 0.0),
            temperature_phase=obs_data.get("temperature_phase", 0.0),
            remaining_time=obs_data.get("remaining_time", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        logger.debug(f"Step result parsed: reward={payload.get('reward'):.2f}, done={payload.get('done', False)}")
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        This method deserializes state information from the server for introspection.
        State includes episode metadata (ID, step count) but not normalized values
        used for training.

        Server Response Format:
            {
                "episode_id": <str UUID>,
                "step_count": <int>,
                "time": <float raw seconds>,
                "temperature": <float raw Celsius>,
                "radius": <float raw nanometers>,
                "target_radius": <float raw nanometers>
            }

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count for session tracking
            
        Example:
            >>> state = client._parse_state({"episode_id": "abc-123", "step_count": 42})
            >>> print(state.episode_id)  # "abc-123"
            >>> print(state.step_count)  # 42
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
