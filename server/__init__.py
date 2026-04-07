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
    # Fallback for when relative imports fail in Docker/special contexts:
    # - ImportError: Module not found
    # - ValueError: "attempted relative import beyond top-level package"
    # - SystemError: Relative import issues in special contexts (Docker, etc.)
    import sys
    import os
    
    # Add both parent directory (for logging_config) and current directory
    # (for heat_treatment_scheduler_environment) to sys.path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    for dir_path in [parent_dir, current_dir]:
        if dir_path not in sys.path:
            sys.path.insert(0, dir_path)
    
    # Now attempt imports with modified path
    from logging_config import get_logger
    from heat_treatment_scheduler_environment import HeatTreatmentSchedulerEnvironment

# Module logger
logger = get_logger(__name__)
logger.debug("Server components module loaded")

__all__ = ["HeatTreatmentSchedulerEnvironment"]
