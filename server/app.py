# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Heat Treatment Scheduler Environment.

This module creates an HTTP server that exposes the HeatTreatmentSchedulerEnvironment
over HTTP and WebSocket endpoints, compatible with OpenEnv's EnvClient.

The server uses OpenEnv's create_app factory to automatically generate:
- REST endpoints for environment interaction
- WebSocket endpoints for persistent sessions
- Automatic API documentation (Swagger UI)
- JSON schema generation for actions and observations

Endpoints:
    POST /reset
        Reset the environment to initial state.
        Returns: ResetResult with initial observation

    POST /step
        Execute an action in the environment.
        Request body: {"action_num": int, "duration_minutes": float}
        Returns: StepResult with observation, reward, done flag

    GET /state
        Query the current environment state (episode_id, step_count).
        Returns: State object

    GET /schema
        Get JSON schemas for actions and observations.
        Returns: {"action": {...}, "observation": {...}}

    WS /ws
        WebSocket endpoint for persistent client sessions.
        Enables efficient multi-step interactions with lower latency.
...

Deployment:
    Development (with auto-reload):
        uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
        
    Production (with multiple workers):
        uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
        
    Direct execution:
        uv run --project . server
        python -m heat_treatment_scheduler.server.app

Docker Usage:
    Build: docker build -t heat_treatment_scheduler_env:latest .
    Run:   docker run -p 8000:8000 heat_treatment_scheduler_env:latest

Configuration:
    max_concurrent_envs: Maximum number of simultaneous WebSocket sessions.
                        Increase if supporting multiple concurrent clients.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation
    from .heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment
    from ..logging_config import get_logger
except (ImportError, ValueError, SystemError, ModuleNotFoundError):
    # Fallback for Docker/non-package context
    from models import HeatTreatmentSchedulerAction, HeatTreatmentSchedulerObservation
    from heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment
    from logging_config import get_logger

# Module logger
logger = get_logger(__name__)
logger.info("Heat Treatment Scheduler FastAPI server initializing")


# ======================== APP FACTORY ========================
# Create the FastAPI application with automatic endpoint generation
# The create_app factory handles all HTTP/WebSocket boilerplate
app = create_app(
    HeatTreatmentSchedulerEnvironment,  # Environment class
    HeatTreatmentSchedulerAction,       # Action type
    HeatTreatmentSchedulerObservation,  # Observation type
    env_name="heat_treatment_scheduler", # Environment identifier
    max_concurrent_envs=8,   # GRPO needs 4 concurrent sessions (num_generations=4)
                             # Set to 8 for headroom
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution of the server.

    This function enables running the server without Docker or explicit uvicorn calls.
    It's useful for:
    - Local development and testing
    - Debugging the environment in isolation
    - Integration with other Python tools
    - Simple deployment scenarios

    Usage:
        # Run directly in Python
        >>> from heat_treatment_scheduler.server.app import main
        >>> main()  # Starts on http://localhost:8000
        
        # Or via command line
        uv run --project . server
        uv run --project . server --port 8001
        python -m heat_treatment_scheduler.server.app

    Args:
        host: Host/IP address to bind to. Default: "0.0.0.0" (all interfaces)
              Use "127.0.0.1" for localhost only
        port: Port number to listen on. Default: 8000
              Common alternatives: 5000, 8001, 8080

    For production deployments with high concurrency, use uvicorn directly:
        uvicorn heat_treatment_scheduler.server.app:app --workers 4 --host 0.0.0.0 --port 8000

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


# ======================== CLI ENTRY POINT ========================
if __name__ == "__main__":
    # Allow port configuration via command-line argument
    import argparse

    parser = argparse.ArgumentParser(
        description="Heat Treatment Scheduler Environment Server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()
    main(host=args.host, port=args.port)
