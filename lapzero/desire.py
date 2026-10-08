from __future__ import annotations

import math
from collections import deque

from . import config, geometry
from .track import Point, Segment


def signed_angle(origin: Point, heading: float, target: Point) -> float:
    """Positive when `target` is to the right of `heading`."""
    desired = math.atan2(target[1] - origin[1], target[0] - origin[0])
    return (desired - heading + math.pi) % (2 * math.pi) - math.pi


def can_see_finish(origin: Point, heading: float, finish: Point, segments: list[Segment]) -> bool:
    """True when the finish sits in the forward ray fan and no wall blocks it."""
    distance = math.dist(origin, finish)
    if distance > config.RAY_MAX_DISTANCE or distance < 1.0:
        return distance <= config.FINISH_RADIUS
    if abs(signed_angle(origin, heading, finish)) > math.radians(config.RAY_SPREAD_DEGREES / 2):
        return False
    angle = math.atan2(finish[1] - origin[1], finish[0] - origin[0])
    hit = segments.raycast(origin, angle, distance) if hasattr(segments, "raycast") else geometry.raycast(
        origin, angle, distance, segments
    )
    return hit >= distance - 2.0


class RouteMap:
    """Open-ground grid. Cars follow it around walls instead of aiming through them."""

    def __init__(self, walls: geometry.SegmentGrid) -> None:
        self.cell = config.EXPLORE_CELL
        self.clearance = config.RACER_RADIUS + 14.0
        self.walls = walls
        self.neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._finish: tuple[Point, set[tuple[int, int]]] | None = None
        self._build()

    def target(self, origin: Point, heading: float, finish: Point) -> tuple[Point, float]:
        """Next point on the open route to the finish, and how soon that route bends (0 to 1)."""
        start = self._nearest_open(origin)
        if start is None:
            return _ahead(origin, heading), 1.0
        path = self._path(start, self._finish_cells(finish))
        if path is None or len(path) < 2:
            return _ahead(origin, heading), 1.0
        return self._aim(path)

    def toward(self, origin: Point, heading: float, destination: tuple[int, int]) -> tuple[Point, float]:
        """Next point along the open route to a cell this car already chose."""
        start = self._nearest_open(origin)
        if start is None or destination not in self.neighbors:
            return _ahead(origin, heading), 1.0
        path = self._path(start, {destination})
        if path is None or len(path) < 2:
            return _ahead(origin, heading), 1.0
        return self._aim(path)

    def explore(
        self,
        origin: Point,
        bearing: float,
        explored: set[tuple[int, int]],
        claimed: set[tuple[int, int]],
    ) -> tuple[tuple[int, int] | None, Point, float]:
        """A far open cell in this car's own direction, and the next point toward it.

        `claimed` cells are already spoken for by other cars, so the pack splits.
        """
        start = self._nearest_open(origin)
        if start is None:
            return None, _ahead(origin, bearing), 1.0
        parent, dist, _found = self._search(start)
        reach = 8
        best_cell: tuple[int, int] | None = None
        best_score = math.inf
        for cell, steps in dist.items():
            if cell == start or cell in explored:
                continue
            center = self._center(cell)
            turn = abs(signed_angle(origin, bearing, center))
            score = turn * 900.0 + abs(steps - reach) * 3.0
            if cell in claimed:
                score += 8000.0
            if score < best_score:
                best_score = score
                best_cell = cell
        if best_cell is None:
            return None, _ahead(origin, bearing), 1.0
        path = []
        cursor: tuple[int, int] | None = best_cell
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        if len(path) < 2:
            return best_cell, _ahead(origin, bearing), 1.0
        point, bend = self._aim(path)
        return best_cell, point, bend

    def _aim(self, path: list[tuple[int, int]]) -> tuple[Point, float]:
        until_turn = _cells_until_turn(path)
        aim_index = max(1, min(2, until_turn, len(path) - 1))
        if until_turn <= 1:
            bend = 1.0
        elif until_turn == 2:
            bend = 0.75
        elif until_turn == 3:
            bend = 0.4
        else:
            bend = 0.0
        return self._center(path[aim_index]), bend

    def _build(self) -> None:
        cols = math.ceil(config.SCREEN_WIDTH / self.cell)
        rows = math.ceil(config.SCREEN_HEIGHT / self.cell)
        open_cells = {
            (gx, gy)
            for gy in range(rows)
            for gx in range(cols)
            if self._point_clear(self._center((gx, gy)))
        }
        for cell in open_cells:
            links = []
            for step in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + step[0], cell[1] + step[1])
                if nxt in open_cells and self._passage_clear(cell, nxt):
                    links.append(nxt)
            self.neighbors[cell] = links

    def _search(
        self, start: tuple[int, int], goals: set[tuple[int, int]] | None = None
    ) -> tuple[dict[tuple[int, int], tuple[int, int] | None], dict[tuple[int, int], int], tuple[int, int] | None]:
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        dist = {start: 0}
        found = None
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if goals and cell in goals and cell != start:
                found = cell
                break
            for nxt in self.neighbors.get(cell, ()):
                if nxt in parent:
                    continue
                parent[nxt] = cell
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
        return parent, dist, found

    def _path(self, start: tuple[int, int], goals: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if not goals:
            return None
        parent, _dist, found = self._search(start, goals)
        if found is None:
            return None
        path = []
        cursor: tuple[int, int] | None = found
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return path

    def _finish_cells(self, finish: Point) -> set[tuple[int, int]]:
        if self._finish is not None and self._finish[0] == finish:
            return self._finish[1]
        span = config.FINISH_RADIUS + self.cell * 0.5
        near = {cell for cell in self.neighbors if math.dist(self._center(cell), finish) <= span}
        cells = near or {min(self.neighbors, key=lambda cell: math.dist(self._center(cell), finish))}
        self._finish = (finish, cells)
        return cells

    def _nearest_open(self, origin: Point) -> tuple[int, int] | None:
        cell = (int(origin[0] // self.cell), int(origin[1] // self.cell))
        if cell in self.neighbors:
            return cell
        if not self.neighbors:
            return None
        return min(self.neighbors, key=lambda item: math.dist(origin, self._center(item)))

    def _center(self, cell: tuple[int, int]) -> Point:
        return ((cell[0] + 0.5) * self.cell, (cell[1] + 0.5) * self.cell)

    def _point_clear(self, point: Point) -> bool:
        return _on_screen(point) and not self.walls.circle_hits(point, self.clearance)

    def _passage_clear(self, left: tuple[int, int], right: tuple[int, int]) -> bool:
        ax, ay = self._center(left)
        bx, by = self._center(right)
        return all(
            self._point_clear((ax + (bx - ax) * t, ay + (by - ay) * t)) for t in (0.33, 0.5, 0.67)
        )


def _cells_until_turn(path: list[tuple[int, int]]) -> int:
    """How many cells ahead the route next changes direction."""
    for index in range(1, len(path) - 1):
        if _step(path[index - 1], path[index]) != _step(path[index], path[index + 1]):
            return index
    return len(path)


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (b[0] - a[0], b[1] - a[1])


def _ahead(origin: Point, heading: float) -> Point:
    reach = config.EXPLORE_CELL * 3
    return (origin[0] + math.cos(heading) * reach, origin[1] + math.sin(heading) * reach)


def drive_toward(
    ray_distances: list[float], goal_angle: float, network_output, bend: float = 0.0
) -> tuple[float, float]:
    """Steer toward a goal point and slow down before a wall or a corner.

    The network only nudges the result, and only on a straight, so a random
    brain cannot shove the car through the turn it is trying to make.
    """
    scale = 1.0 / config.RAY_MAX_DISTANCE
    mid = len(ray_distances) // 2
    ahead = ray_distances[mid] * scale
    avoid = ((ray_distances[-1] - ray_distances[0]) * 1.2 + (ray_distances[-2] - ray_distances[1]) * 0.6) * scale
    steer = _clip(goal_angle / (math.pi / 4))
    if ahead < 0.2:
        steer = _clip(steer * 0.85 + avoid * 0.7)
        if abs(steer) < 0.45:
            steer = 0.85 if avoid >= 0.0 else -0.85
    else:
        steer = _clip(steer + avoid * 0.12)
    steer = _clip(steer + 0.06 * float(network_output[0]))
    if abs(goal_angle) > 0.12 and abs(steer) < 0.45:
        steer = 0.45 if goal_angle > 0.0 else -0.45
    straight = max(0.0, 1.0 - bend) * max(0.0, 1.0 - abs(goal_angle) / (math.pi / 3))
    room = min(1.0, ahead / 0.5)
    # Stay above the throttle deadzone. At zero speed the car cannot steer.
    limit = 0.38 + 0.55 * straight * room
    if bend >= 0.7:
        limit = min(limit, 0.42)
    elif bend >= 0.4:
        limit = min(limit, 0.55)
    throttle = min(1.0, max(0.38, limit + 0.18 * max(0.0, float(network_output[1])) * straight * room))
    return (steer, -0.4 if ahead < 0.12 else throttle)


def _clip(value: float) -> float:
    return -1.0 if value < -1.0 else 1.0 if value > 1.0 else value


def _on_screen(point: Point) -> bool:
    return 0.0 <= point[0] < config.SCREEN_WIDTH and 0.0 <= point[1] < config.SCREEN_HEIGHT
