# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Heat Treatment Scheduler Environment Server Components.

This package contains the server-side implementation of the heat treatment scheduler
environment, including the core physics simulation and HTTP/WebSocket API endpoints.

Main Components:
    HeatTreatmentSchedulerEnvironment: Core simulation engine with physics model
    FastAPI application (app.py): HTTP server providing REST and WebSocket APIs
"""

try:
    # Try relative imports first (normal Python package structure)
    from ..logging_config import get_logger
    from .heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment
except (ImportError, ValueError, SystemError):
    # Fallback for when relative imports fail:
    # - ImportError: Module not found
    # - ValueError: "attempted relative import beyond top-level package"
    # - SystemError: Relative import issues in special contexts (Docker, etc.)
    try:
        from logging_config import get_logger
        from heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment
    except ImportError:
        # Last resort: try importing from parent directory
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from logging_config import get_logger
        from heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment

# Module logger
logger = get_logger(__name__)
logger.debug("Server components module loaded")

__all__ = ["HeatTreatmentSchedulerEnvironment"]
