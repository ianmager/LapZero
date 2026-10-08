from __future__ import annotations

import math
from typing import Iterable, List

from . import config, geometry
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


def cast_rays(origin: Point, heading: float, obstacles: Iterable[Segment] | geometry.SegmentGrid) -> List[float]:
    """Distance readings for each ray, capped at RAY_MAX_DISTANCE. These plus
    speed are the neural net's inputs.
    """
    if isinstance(obstacles, geometry.SegmentGrid):
        cast = obstacles.raycast
    else:
        segments = list(obstacles)
        cast = lambda origin, angle, distance: raycast(origin, angle, distance, segments)
    return [cast(origin, angle, config.RAY_MAX_DISTANCE) for angle in ray_angles(heading)]
