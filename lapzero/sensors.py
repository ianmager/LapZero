from __future__ import annotations

import math
from typing import Iterable, List

from . import config
from .geometry import raycast
from .track import Point, Segment


def ray_angles(heading: float) -> List[float]:
    """RAY_COUNT angles, evenly spread across RAY_SPREAD_DEGREES, centered on heading."""
    if config.RAY_COUNT == 1:
        return [heading]
    spread = math.radians(config.RAY_SPREAD_DEGREES)
    step = spread / (config.RAY_COUNT - 1)
    start = heading - spread / 2
    return [start + i * step for i in range(config.RAY_COUNT)]


def cast_rays(origin: Point, heading: float, segments: Iterable[Segment]) -> List[float]:
    """Distance readings for each ray, capped at RAY_MAX_DISTANCE. These plus
    speed are the neural net's inputs in a later stage.
    """
    segments = list(segments)
    return [
        raycast(origin, angle, config.RAY_MAX_DISTANCE, segments)
        for angle in ray_angles(heading)
    ]
