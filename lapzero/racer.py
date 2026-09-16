from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .track import Point


@dataclass
class RacerControls:
    throttle: bool = False
    brake: bool = False
    steer_left: bool = False
    steer_right: bool = False


class Racer:
    """Momentum-based car: accelerates/decelerates rather than snapping to a
    speed, and turns tighter at low speed than at high speed.
    """

    def __init__(self, x: float, y: float, heading: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.heading = heading  # radians, 0 = facing +x
        self.speed = 0.0

    @property
    def position(self) -> Point:
        return (self.x, self.y)

    def reset(self, x: float, y: float, heading: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0

    def step(self, dt: float, controls: RacerControls) -> None:
        self._update_speed(dt, controls)
        self._update_heading(dt, controls)
        self.x += math.cos(self.heading) * self.speed * dt
        self.y += math.sin(self.heading) * self.speed * dt

    def _update_speed(self, dt: float, controls: RacerControls) -> None:
        if controls.throttle:
            self.speed += config.RACER_ACCELERATION * dt
        elif controls.brake:
            self.speed -= config.RACER_BRAKE_DECEL * dt
        elif self.speed > 0:
            self.speed = max(0.0, self.speed - config.RACER_FRICTION_DECEL * dt)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + config.RACER_FRICTION_DECEL * dt)
        self.speed = max(config.RACER_MIN_SPEED, min(config.RACER_MAX_SPEED, self.speed))

    def _update_heading(self, dt: float, controls: RacerControls) -> None:
        if self.speed == 0 or not (controls.steer_left or controls.steer_right):
            return
        speed_fraction = min(abs(self.speed) / config.RACER_MAX_SPEED, 1.0)
        turn_rate = config.RACER_MAX_TURN_RATE - speed_fraction * (
            config.RACER_MAX_TURN_RATE - config.RACER_MIN_TURN_RATE
        )
        if controls.steer_left:
            self.heading -= turn_rate * dt
        if controls.steer_right:
            self.heading += turn_rate * dt
