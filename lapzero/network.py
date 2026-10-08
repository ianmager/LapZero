from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config
from .racer import RacerControls


class Network:
    """Small feedforward net: ray distances + speed in, steering + throttle out.

    Weights only — no backprop. `randomize()` draws a fresh set; later the
    genetic algorithm will replace those weights directly.
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


def make_inputs(ray_distances: Sequence[float], speed: float) -> np.ndarray:
    """Normalize sensing into the net's input vector, each roughly in [-1, 1]."""
    rays = np.asarray(ray_distances, dtype=float) / config.RAY_MAX_DISTANCE
    return np.concatenate([rays, [speed / config.RACER_MAX_SPEED]])


def controls_from_output(output: Sequence[float]) -> RacerControls:
    """Map continuous steering/throttle in [-1, 1] onto the racer's on/off controls."""
    steer, throttle = float(output[0]), float(output[1])
    return RacerControls(
        throttle=throttle > config.NN_THROTTLE_DEADZONE,
        brake=throttle < -config.NN_THROTTLE_DEADZONE,
        steer_left=steer < -config.NN_STEER_DEADZONE,
        steer_right=steer > config.NN_STEER_DEADZONE,
    )
