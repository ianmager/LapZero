from __future__ import annotations

import math
from typing import Iterable, Optional

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


def segment_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
    """Intersection point of segments p1-p2 and p3-p4, or None if they don't cross."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if denom == 0:
        return None  # parallel (or overlapping) — ignore for raycasting purposes
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denom
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def raycast(origin: Point, angle: float, max_distance: float, segments: Iterable[Segment]) -> float:
    """Distance from origin to the nearest segment hit by a ray at `angle`,
    capped at `max_distance` if nothing is hit within range.
    """
    end = (origin[0] + math.cos(angle) * max_distance, origin[1] + math.sin(angle) * max_distance)
    closest = max_distance
    for a, b in segments:
        hit = segment_intersection(origin, end, a, b)
        if hit is not None:
            closest = min(closest, math.dist(origin, hit))
    return closest
