from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config
from .racer import RacerControls


class Network:
    """Small feedforward net: ray distances and speed in, steering and throttle out.

    Weights only — no backprop. The genetic algorithm copies, crosses, and
    mutates `genome()`.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.layer_sizes = (config.RAY_COUNT + 1, config.NN_HIDDEN_SIZE, 2)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.randomize()

    def randomize(self) -> None:
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(self.layer_sizes, self.layer_sizes[1:]):
            scale = 1.0 / np.sqrt(n_in)
            self.weights.append(self.rng.normal(0.0, scale, size=(n_out, n_in)))
            self.biases.append(self.rng.normal(0.0, scale, size=(n_out,)))

    def forward(self, inputs: Sequence[float]) -> np.ndarray:
        x = np.asarray(inputs, dtype=float)
        for weight, bias in zip(self.weights, self.biases):
            x = np.tanh(weight @ x + bias)
        return x

    def genome(self) -> np.ndarray:
        parts = [weight.ravel() for weight in self.weights]
        parts += [bias.ravel() for bias in self.biases]
        return np.concatenate(parts)

    def set_genome(self, genes: Sequence[float]) -> None:
        genes = np.asarray(genes, dtype=float)
        index = 0
        for array in (*self.weights, *self.biases):
            size = array.size
            array[:] = genes[index:index + size].reshape(array.shape)
            index += size

    def nudge_throttle(self, amount: float) -> None:
        """Push the throttle output up. Used so lap-time search tries going faster."""
        self.biases[-1][1] += amount

    def clone(self) -> "Network":
        other = Network.__new__(Network)
        other.rng = self.rng
        other.layer_sizes = self.layer_sizes
        other.weights = [weight.copy() for weight in self.weights]
        other.biases = [bias.copy() for bias in self.biases]
        return other


def make_inputs(ray_distances: Sequence[float], speed: float) -> np.ndarray:
    """Normalize sensing into the net's input vector, each roughly in [-1, 1]."""
    rays = np.asarray(ray_distances, dtype=float) / config.RAY_MAX_DISTANCE
    return np.concatenate([rays, [speed / config.RACER_MAX_SPEED]])


class AimlessDriver:
    """Drives ahead at a modest speed and turns away when a wall is close.

    A head-on wall makes it commit to one turn direction until the way ahead
    is open again — flipping side to side never rotates the car in time.
    Which way it turns comes from the network, so new weights take a
    different path without overriding the avoidance.
    """

    def __init__(self, network: Network) -> None:
        self.network = network
        self.turning = 0  # -1 left, +1 right, 0 cruising

    def output(self, ray_distances: Sequence[float], speed: float) -> np.ndarray:
        rays = np.asarray(ray_distances, dtype=float) / config.RAY_MAX_DISTANCE
        # Only the forward ray counts as "blocked". Side rays stay short in a
        # corridor and must not be treated as a wall straight ahead.
        ahead = float(rays[len(rays) // 2])
        # Positive steer turns right, away from a closer left-hand wall.
        avoid = (rays[-1] - rays[0]) * 2.2 + (rays[-2] - rays[1]) * 1.4
        wander = float(self.network.forward(make_inputs(ray_distances, speed))[0])

        if self.turning != 0:
            if ahead > 0.72:
                self.turning = 0
            else:
                return np.array([float(self.turning), self._turn_throttle(ahead, speed)])

        steer = float(np.clip(avoid, -1.0, 1.0))
        if ahead < 0.48 and abs(avoid) < 0.75:
            self.turning = 1 if wander >= 0 else -1
            if rays[-1] > rays[0] + 0.08:
                self.turning = 1
            elif rays[0] > rays[-1] + 0.08:
                self.turning = -1
            return np.array([float(self.turning), self._turn_throttle(ahead, speed)])

        return np.array([steer, self._cruise_throttle(ahead, speed)])

    def _turn_throttle(self, ahead: float, speed: float) -> float:
        if ahead < 0.12:
            return -1.0
        if speed < 36.0:
            return 1.0
        if speed > 58.0:
            return -1.0
        return 0.0

    def _cruise_throttle(self, ahead: float, speed: float) -> float:
        cruise = min(config.AI_CRUISE_SPEED, ahead * config.RAY_MAX_DISTANCE * 0.55)
        if speed > cruise + 10.0:
            return -1.0
        if speed < cruise - 8.0:
            return 1.0
        return 0.0


def controls_from_output(output: Sequence[float], analog: bool = False) -> RacerControls:
    """Map steering/throttle in [-1, 1] onto racer controls.

    Analog mode keeps the magnitude: a larger throttle output is a higher
    target speed, and a larger steer output turns harder. Evolution uses
    that so later generations can actually go faster.
    """
    steer, throttle = float(output[0]), float(output[1])
    controls = RacerControls(
        throttle=throttle > config.NN_THROTTLE_DEADZONE,
        brake=throttle < -config.NN_THROTTLE_DEADZONE,
        steer_left=steer < -config.NN_STEER_DEADZONE,
        steer_right=steer > config.NN_STEER_DEADZONE,
    )
    if analog:
        if abs(throttle) > config.NN_THROTTLE_DEADZONE:
            controls.target_speed = throttle * config.RACER_MAX_SPEED
        if abs(steer) > config.NN_STEER_DEADZONE:
            controls.steer_power = abs(steer)
        else:
            controls.steer_power = 0.0
    return controls
