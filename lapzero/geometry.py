from __future__ import annotations

import math
from typing import Iterable, Optional

from . import config
from .track import Point, Segment, Track


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


def window_border_segments() -> list[Segment]:
    """The four edges of the window, treated as walls."""
    width = float(config.SCREEN_WIDTH - 1)
    height = float(config.SCREEN_HEIGHT - 1)
    return [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]


def obstacle_segments(track: Track) -> list[Segment]:
    """Drawn walls plus the window border."""
    return [*track.segments(), *window_border_segments()]


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


class SegmentGrid:
    """Walls bucketed into cells so a ray or a car only tests nearby segments."""

    def __init__(self, segments: Iterable[Segment], cell: float = 80.0) -> None:
        self.cell = cell
        self.buckets: dict[tuple[int, int], list[Segment]] = {}
        for segment in segments:
            (x1, y1), (x2, y2) = segment
            for ix in range(int(min(x1, x2) // cell), int(max(x1, x2) // cell) + 1):
                for iy in range(int(min(y1, y2) // cell), int(max(y1, y2) // cell) + 1):
                    self.buckets.setdefault((ix, iy), []).append(segment)

    def circle_hits(self, center: Point, radius: float) -> bool:
        seen: set[Segment] = set()
        for segment in self._near(center[0], center[1], radius):
            if segment in seen:
                continue
            seen.add(segment)
            if distance_point_to_segment(center, segment[0], segment[1]) <= radius:
                return True
        return False

    def raycast(self, origin: Point, angle: float, max_distance: float) -> float:
        ox, oy = origin
        dx, dy = math.cos(angle), math.sin(angle)
        if dx == 0.0 and dy == 0.0:
            return max_distance
        cell = self.cell
        ix, iy = int(ox // cell), int(oy // cell)
        step_x, step_y = (1 if dx > 0 else -1), (1 if dy > 0 else -1)
        t_delta_x = abs(cell / dx) if dx else math.inf
        t_delta_y = abs(cell / dy) if dy else math.inf
        t_max_x = ((ix + (dx > 0)) * cell - ox) / dx if dx else math.inf
        t_max_y = ((iy + (dy > 0)) * cell - oy) / dy if dy else math.inf
        end = (ox + dx * max_distance, oy + dy * max_distance)
        closest = max_distance
        seen: set[Segment] = set()
        traveled = 0.0
        while traveled <= closest:
            for segment in self.buckets.get((ix, iy), ()):
                if segment in seen:
                    continue
                seen.add(segment)
                hit = segment_intersection(origin, end, segment[0], segment[1])
                if hit is not None:
                    closest = min(closest, math.dist(origin, hit))
            if t_max_x < t_max_y:
                traveled, t_max_x, ix = t_max_x, t_max_x + t_delta_x, ix + step_x
            elif t_max_y < t_max_x:
                traveled, t_max_y, iy = t_max_y, t_max_y + t_delta_y, iy + step_y
            else:
                traveled = t_max_x
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                ix += step_x
                iy += step_y
        return closest

    def _near(self, x: float, y: float, radius: float):
        cell = self.cell
        for ix in range(int((x - radius) // cell), int((x + radius) // cell) + 1):
            for iy in range(int((y - radius) // cell), int((y + radius) // cell) + 1):
                yield from self.buckets.get((ix, iy), ())
