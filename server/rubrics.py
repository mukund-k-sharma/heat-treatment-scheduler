# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Composable Rubrics for Heat Treatment Reward Computation.

Decomposes the monolithic reward function into modular, inspectable rubric
components using OpenEnv's Rubric framework (RFC 004). Each rubric evaluates
one independent axis of agent performance:

    rubric = HeatTreatmentRubric(alloy)
    reward = rubric(action, observation)

    # Introspect per-component scores during training:
    for name, r in rubric.named_rubrics():
        wandb.log({f"rubric/{name}": r.last_score})

Architecture:
    HeatTreatmentRubric (WeightedSum, 4 components)
    ├── proximity   (40%) — How close is the radius to target?
    ├── efficiency  (20%) — Energy and time cost minimization
    ├── safety      (20%) — Avoid melting / over-coarsening zones
    └── terminal    (20%) — Episode outcome bonus/penalty
"""

import numpy as np
from typing import Any

from openenv.core.rubrics import Rubric, WeightedSum

try:
    from ..models import ALLOY_REGISTRY, AlloyProperties
except ImportError:
    from models import ALLOY_REGISTRY, AlloyProperties


class ProximityRubric(Rubric):
    """Rewards the agent for minimizing the radius error.

    Dense per-step signal: higher reward when current radius is close to
    the target. Uses a combination of linear and quadratic terms for
    smooth gradient landscape.

    Score range: [-1.0, 1.0]
        1.0 = exactly at target
        0.0 = moderately far
       -1.0 = very far (error ≥ r_max_clip)
    """

    def __init__(self, alloy: AlloyProperties):
        super().__init__()
        self.r_target = (alloy.r_target_min + alloy.r_target_max) / 2.0
        self.r_max_clip = alloy.r_max_clip
        self.r_target_min = alloy.r_target_min
        self.r_target_max = alloy.r_target_max

    def forward(self, action: Any, observation: Any) -> float:
        """Compute proximity score from observation."""
        # Denormalize radius from observation
        r = observation.radius * self.r_max_clip
        error = abs(r - self.r_target)
        error = min(error, self.r_max_clip * 2)

        # Linear + quadratic penalty, normalized to [-1, 1]
        raw = -0.1 * error - 0.01 * (error ** 2)
        # Normalize: worst case is ~-0.1*60 - 0.01*3600 ≈ -42
        # Best case is 0. Map to [-1, 1]
        max_penalty = 0.1 * self.r_max_clip + 0.01 * (self.r_max_clip ** 2)
        score = 1.0 + (raw / max_penalty) if max_penalty > 0 else 1.0
        return float(np.clip(score, -1.0, 1.0))


class EfficiencyRubric(Rubric):
    """Penalizes excessive energy consumption and time usage.

    Encourages the agent to reach the target quickly and with minimal
    energy expenditure. Energy is proportional to temperature × time.

    Score range: [0.0, 1.0]
        1.0 = no energy/time cost (instantaneous, cold)
        0.0 = maximum energy/time budget exhausted
    """

    def __init__(self, alloy: AlloyProperties):
        super().__init__()
        self.temp_max = alloy.temp_max

    def forward(self, action: Any, observation: Any) -> float:
        """Compute efficiency score from observation."""
        # Use normalized time remaining as a proxy for time efficiency
        remaining = getattr(observation, 'remaining_time', 1.0)

        # Temperature-based energy penalty: higher temp = more energy
        temp_norm = getattr(observation, 'temperature', 0.0)
        energy_factor = 1.0 - (temp_norm * 0.3)  # Up to 30% penalty at max temp

        score = remaining * energy_factor
        return float(np.clip(score, 0.0, 1.0))


class SafetyRubric(Rubric):
    """Penalizes dangerous operation near melting/ripening thresholds.

    Acts as a soft constraint: score degrades as material temperature
    approaches the melting point. This teaches the agent to avoid
    catastrophic failure zones proactively.

    Score range: [0.0, 1.0]
        1.0 = safe operation (T_material well below danger zone)
        0.0 = at or above melting temperature
    """

    def __init__(self, alloy: AlloyProperties):
        super().__init__()
        self.temp_melt = alloy.temp_melt
        self.temp_max = alloy.temp_max
        # Warning zone starts 100°C below melting
        self.warning_temp = self.temp_melt - 100.0

    def forward(self, action: Any, observation: Any) -> float:
        """Compute safety score from observation."""
        # Denormalize temperature
        T_material = observation.temperature * self.temp_max

        if T_material >= self.temp_melt:
            return 0.0  # Catastrophic: melted

        if T_material > self.warning_temp:
            # Linear decay from 1.0 at warning_temp to 0.0 at temp_melt
            margin = (self.temp_melt - T_material) / 100.0
            return float(np.clip(margin, 0.0, 1.0))

        return 1.0  # Safe


class TerminalRubric(Rubric):
    """Evaluates episode outcome quality at termination.

    Only produces a meaningful score when the episode is done.
    Returns a neutral score (0.5) during ongoing episodes.

    Score range: [0.0, 1.0]
        1.0 = radius in target window, close to center
        0.5 = episode still running (neutral)
        0.0 = melted or over-coarsened
    """

    def __init__(self, alloy: AlloyProperties):
        super().__init__()
        self.r_target = (alloy.r_target_min + alloy.r_target_max) / 2.0
        self.r_target_min = alloy.r_target_min
        self.r_target_max = alloy.r_target_max
        self.r_max_clip = alloy.r_max_clip
        self.temp_melt = alloy.temp_melt
        self.temp_max = alloy.temp_max

    def forward(self, action: Any, observation: Any) -> float:
        """Compute terminal outcome score."""
        if not getattr(observation, 'done', False):
            return 0.5  # Neutral score during episode

        r = observation.radius * self.r_max_clip
        T_material = observation.temperature * self.temp_max

        # Catastrophic failures
        if T_material >= self.temp_melt:
            return 0.0  # Melted

        if r > self.r_target_max:
            return 0.1  # Over-coarsened

        # Success: radius in target window
        if self.r_target_min <= r <= self.r_target_max:
            error = abs(r - self.r_target)
            # Gaussian bonus: peaks at 1.0 when error = 0
            precision = float(np.exp(-(error ** 2) / 10.0))
            return 0.5 + 0.5 * precision  # [0.5, 1.0]

        # Partial: didn't reach target but didn't catastrophically fail
        return 0.3


class HeatTreatmentRubric(Rubric):
    """Composable rubric for heat treatment reward computation.

    Combines four independent evaluation axes using WeightedSum.
    Each component can be inspected individually for training diagnostics.

    Weights:
        proximity  = 0.40 — Primary signal: radius accuracy
        efficiency = 0.15 — Secondary: minimize energy and time
        safety     = 0.25 — Critical: avoid melting/ripening zones
        terminal   = 0.20 — Outcome: episode success/failure bonus

    Usage:
        rubric = HeatTreatmentRubric(alloy)
        env = HeatTreatmentSchedulerEnvironment(rubric=rubric, ...)

        # In training loop, introspect:
        for name, r in env.rubric.named_rubrics():
            print(f"{name}: {r.last_score}")

    Introspection paths:
        rubric.proximity.last_score
        rubric.efficiency.last_score
        rubric.safety.last_score
        rubric.terminal.last_score
    """

    def __init__(self, alloy: AlloyProperties):
        super().__init__()
        self.proximity = ProximityRubric(alloy)
        self.efficiency = EfficiencyRubric(alloy)
        self.safety = SafetyRubric(alloy)
        self.terminal = TerminalRubric(alloy)

        self._scorer = WeightedSum(
            rubrics=[self.proximity, self.efficiency, self.safety, self.terminal],
            weights=[0.40, 0.15, 0.25, 0.20],
        )

    def forward(self, action: Any, observation: Any) -> float:
        """Compute weighted sum of all rubric components.

        Returns a score in [0.0, 1.0] which is then rescaled to the
        environment's reward range (±500) by the environment's step().
        """
        return self._scorer(action, observation)

    def reset(self) -> None:
        """Reset all child rubric state for a new episode."""
        for child in self.children():
            child.reset()
