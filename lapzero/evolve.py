from __future__ import annotations

import math

import numpy as np

from . import config, geometry
from .desire import RouteMap, can_see_finish, drive_toward, signed_angle
from .network import Network, controls_from_output, make_inputs
from .racer import Racer
from .sensors import cast_rays
from .track import Point, Track


def _cell(point: Point) -> tuple[int, int]:
    return (int(point[0] // config.EXPLORE_CELL), int(point[1] // config.EXPLORE_CELL))


class Agent:
    def __init__(self, network: Network, origin: Point, heading: float) -> None:
        self.network = network
        self.spawn_heading = heading
        self.racer = Racer(origin[0], origin[1], heading)
        self.alive = True
        self.finished = False
        self.finish_time: float | None = None
        self.visited = {_cell(origin)}
        self.discovered = 0
        self.stall = 0.0
        self.bend = 0.0
        self.best_goal_dist = math.inf
        self.fitness = 0.0
        self.goal: Point | None = None
        self.destination: tuple[int, int] | None = None
        self.rays: list[float] = []

    def score(self, time_limit: float, finish: Point, finish_known: bool) -> float:
        """Before anyone sees the finish, more new ground is better. After that,
        closer to the finish wins, and a faster finish wins outright.
        """
        if self.finished and self.finish_time is not None:
            return 100_000.0 + time_limit / max(self.finish_time, 0.05)
        if finish_known:
            return 10_000.0 - math.dist(self.racer.position, finish)
        return float(self.discovered)


class Evolution:
    """One population, stepped in real time, replaced when the generation ends."""

    def __init__(self, track: Track, rng: np.random.Generator | None = None) -> None:
        self.track = track
        self.rng = rng if rng is not None else np.random.default_rng()
        self.walls = geometry.SegmentGrid(geometry.obstacle_segments(track))
        self.generation = 1
        self.elapsed = 0.0
        self.history: list[float] = []
        self.time_history: list[float | None] = []
        self.best_time: float | None = None
        self.champion: Agent | None = None
        self.finish_known = False
        self.explored: set[tuple[int, int]] = {_cell(track.start)}
        self.routes = RouteMap(self.walls)
        self.agents: list[Agent] = []
        self._spawn()

    def step(self, dt: float) -> str | None:
        """Advance one fixed physics step. `dt` only matters to the caller, which
        may invoke this several times per frame to fast-forward.
        """
        del dt
        dt = 1.0 / config.FPS
        self.elapsed += dt
        for agent in self.agents:
            if agent.alive and not agent.finished:
                self._step_agent(agent, dt)
        still_going = any(agent.alive and not agent.finished for agent in self.agents)
        timed_out = self.elapsed >= config.GA_GENERATION_SECONDS
        if still_going and not timed_out:
            return None
        return self._end_generation()

    def leader(self) -> Agent:
        alive = [agent for agent in self.agents if agent.alive and not agent.finished]
        pool = alive or self.agents
        return max(pool, key=lambda agent: agent.fitness)

    def finished_count(self) -> int:
        return sum(agent.finished for agent in self.agents)

    def alive_count(self) -> int:
        return sum(agent.alive and not agent.finished for agent in self.agents)

    def _step_agent(self, agent: Agent, dt: float) -> None:
        if not agent.rays:
            agent.rays = cast_rays(agent.racer.position, agent.racer.heading, self.walls)
        if not self.finish_known and can_see_finish(
            agent.racer.position, agent.racer.heading, self.track.finish, self.walls
        ):
            self.finish_known = True
        agent.goal = self._goal_for(agent)
        network_output = agent.network.forward(make_inputs(agent.rays, agent.racer.speed))
        goal_angle = signed_angle(agent.racer.position, agent.racer.heading, agent.goal)
        output = drive_toward(agent.rays, goal_angle, network_output, agent.bend)
        agent.racer.step(dt, controls_from_output(output, analog=True))
        if self.walls.circle_hits(agent.racer.position, config.RACER_RADIUS):
            agent.alive = False
        elif self._reached_finish(agent):
            agent.finished = True
            agent.finish_time = self.elapsed
            agent.alive = False
        else:
            cell = _cell(agent.racer.position)
            if cell not in self.explored:
                self.explored.add(cell)
                agent.discovered += 1
            if cell not in agent.visited:
                agent.visited.add(cell)
            if self.finish_known:
                agent.stall = 0.0
            else:
                goal_dist = math.dist(agent.racer.position, agent.goal)
                if goal_dist < agent.best_goal_dist - 12.0:
                    agent.best_goal_dist = goal_dist
                    agent.stall = 0.0
                else:
                    agent.stall += dt
                    if agent.stall >= config.GA_STALL_SECONDS:
                        agent.alive = False
            agent.rays = cast_rays(agent.racer.position, agent.racer.heading, self.walls)
        agent.fitness = max(
            agent.fitness,
            agent.score(config.GA_GENERATION_SECONDS, self.track.finish, self.finish_known),
        )

    def _goal_for(self, agent: Agent) -> Point:
        if self.finish_known:
            return self._set_goal(agent, *self.routes.target(
                agent.racer.position, agent.racer.heading, self.track.finish
            ))
        if self._keeping_destination(agent):
            if agent.goal is not None and _cell(agent.racer.position) != _cell(agent.goal):
                return agent.goal
            return self._set_goal(agent, *self.routes.toward(
                agent.racer.position, agent.spawn_heading, agent.destination
            ))
        claimed = {
            other.destination
            for other in self.agents
            if other is not agent and other.destination is not None
        }
        destination, point, bend = self.routes.explore(
            agent.racer.position, agent.spawn_heading, self.explored, claimed
        )
        if destination is None:
            return self._set_goal(agent, *self.routes.target(
                agent.racer.position, agent.spawn_heading, self.track.finish
            ))
        agent.destination = destination
        return self._set_goal(agent, point, bend)

    def _keeping_destination(self, agent: Agent) -> bool:
        return (
            agent.destination is not None
            and agent.destination not in self.explored
            and _cell(agent.racer.position) != agent.destination
        )

    def _set_goal(self, agent: Agent, point: Point, bend: float) -> Point:
        agent.goal = point
        agent.bend = bend
        agent.best_goal_dist = math.inf
        return point

    def _reached_finish(self, agent: Agent) -> bool:
        return math.dist(agent.racer.position, self.track.finish) <= config.FINISH_RADIUS

    def _end_generation(self) -> str:
        best = max(agent.fitness for agent in self.agents)
        self.history.append(best)
        times = [agent.finish_time for agent in self.agents if agent.finish_time is not None]
        if times:
            fastest = min(times)
            self.time_history.append(fastest)
            record = min((agent for agent in self.agents if agent.finished), key=lambda agent: agent.finish_time)
            if self.best_time is None or fastest < self.best_time - 1e-3:
                self.best_time = fastest
                self.champion = record
        else:
            self.time_history.append(None)
        finished = self.finished_count()
        time_bit = f", best time {min(times):.1f}s" if times else f", explored {len(self.explored)} cells"
        summary = (
            f"Generation {self.generation} — best {best:.1f}, "
            f"finished {finished}/{len(self.agents)}{time_bit}"
        )
        self.generation += 1
        self.elapsed = 0.0
        self._spawn()
        return summary

    def _spawn(self) -> None:
        if self.history:
            starters = self._next_starters()
        else:
            starters = [
                (Network(self.rng), (index / config.GA_POPULATION) * math.tau)
                for index in range(config.GA_POPULATION)
            ]
        if self.finish_known:
            point, _bend = self.routes.target(self.track.start, 0.0, self.track.finish)
            aim = math.atan2(point[1] - self.track.start[1], point[0] - self.track.start[0])
            starters = [
                (network, aim + (index - len(starters) / 2) * 0.04)
                for index, (network, _heading) in enumerate(starters)
            ]
        else:
            starters = [
                (network, (index / len(starters)) * math.tau)
                for index, (network, _heading) in enumerate(starters)
            ]
        self.agents = []
        for index, (network, heading) in enumerate(starters):
            origin = self.track.start if self.finish_known else self._spread_origin(heading, index)
            agent = Agent(network, origin, heading)
            agent.rays = cast_rays(agent.racer.position, agent.racer.heading, self.walls)
            self.agents.append(agent)

    def _spread_origin(self, heading: float, index: int) -> Point:
        """Park each car on its own patch of open ground so they do not leave in a stack."""
        radius = 48.0 + (index % 5) * 22.0
        origin = (
            self.track.start[0] + math.cos(heading) * radius,
            self.track.start[1] + math.sin(heading) * radius,
        )
        margin = 36.0
        if not (
            margin <= origin[0] <= config.SCREEN_WIDTH - margin
            and margin <= origin[1] <= config.SCREEN_HEIGHT - margin
        ):
            return self.track.start
        if self.walls.circle_hits(origin, config.RACER_RADIUS + 6):
            return self.track.start
        return origin

    def _next_starters(self) -> list[tuple[Network, float]]:
        if self.champion is not None:
            return self._refine_starters()
        ranked = sorted(self.agents, key=lambda agent: agent.fitness, reverse=True)
        pool = ranked[:max(config.GA_ELITE, len(ranked) // 2)]
        starters = [(agent.network.clone(), agent.spawn_heading) for agent in ranked[:config.GA_ELITE]]
        while len(starters) < config.GA_POPULATION:
            if self.rng.random() < 0.25:
                starters.append((Network(self.rng), float(self.rng.uniform(0.0, math.tau))))
                continue
            parent = _tournament(pool, self.rng)
            child = parent.network.clone()
            child.set_genome(_mutate(child.genome(), self.rng, config.GA_MUTATION_RATE, config.GA_MUTATION_SCALE))
            heading = parent.spawn_heading + float(self.rng.normal(0.0, 0.3))
            starters.append((child, heading))
        return starters

    def _refine_starters(self) -> list[tuple[Network, float]]:
        """Most of the next generation is a small tweak of the lap record."""
        champion = self.champion
        starters = [(champion.network.clone(), champion.spawn_heading)]
        while len(starters) < config.GA_POPULATION:
            child = champion.network.clone()
            if self.rng.random() < 0.8:
                child.set_genome(_mutate(
                    child.genome(), self.rng, config.GA_FINE_MUTATION_RATE, config.GA_FINE_MUTATION_SCALE
                ))
                # Most children are asked to go a little faster. Ones that crash
                # don't replace the record; ones that survive become it.
                child.nudge_throttle(float(self.rng.uniform(0.04, 0.12)))
                heading = champion.spawn_heading + float(self.rng.normal(0.0, 0.04))
            else:
                child.set_genome(_mutate(
                    child.genome(), self.rng, config.GA_MUTATION_RATE, config.GA_MUTATION_SCALE
                ))
                heading = champion.spawn_heading + float(self.rng.normal(0.0, 0.12))
            starters.append((child, heading))
        return starters


def _tournament(pool: list[Agent], rng: np.random.Generator) -> Agent:
    picks = rng.choice(len(pool), size=min(config.GA_TOURNAMENT, len(pool)), replace=False)
    return max((pool[int(index)] for index in picks), key=lambda agent: agent.fitness)


def _mutate(genes: np.ndarray, rng: np.random.Generator, rate: float, scale: float) -> np.ndarray:
    genes = genes.copy()
    mask = rng.random(genes.shape) < rate
    count = int(mask.sum())
    if count:
        genes[mask] += rng.normal(0.0, scale, size=count)
    return np.clip(genes, -5.0, 5.0)
