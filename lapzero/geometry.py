from __future__ import annotations

import math
from typing import Iterable

from .track import Point, Segment


def closest_point_on_segment(p: Point, a: Point, b: Point) -> Point:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return a
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return (ax + t * dx, ay + t * dy)


def distance_point_to_segment(p: Point, a: Point, b: Point) -> float:
    return math.dist(p, closest_point_on_segment(p, a, b))


def circle_hits_any_segment(center: Point, radius: float, segments: Iterable[Segment]) -> bool:
    return any(distance_point_to_segment(center, a, b) <= radius for a, b in segments)
