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
    # When set, speed eases toward this value instead of a held on/off throttle.
    # Manual driving leaves it unset.
    target_speed: float | None = None
    steer_power: float = 1.0


class Racer:
    """Momentum-based car: accelerates/decelerates rather than snapping to a
    speed, and turns tighter at low speed than at high speed.
    """

    def __init__(self, x: float, y: float, heading: float = 0.0, max_speed: float | None = None) -> None:
        self.x = x
        self.y = y
        self.heading = heading  # radians, 0 = facing +x
        self.speed = 0.0
        self.max_speed = config.RACER_MAX_SPEED if max_speed is None else max_speed

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
        if controls.target_speed is not None:
            target = max(config.RACER_MIN_SPEED, min(self.max_speed, controls.target_speed))
            if self.speed < target:
                self.speed = min(target, self.speed + config.RACER_ACCELERATION * dt)
            elif self.speed > target:
                self.speed = max(target, self.speed - config.RACER_BRAKE_DECEL * dt)
            return
        if controls.throttle:
            self.speed += config.RACER_ACCELERATION * dt
        elif controls.brake:
            self.speed -= config.RACER_BRAKE_DECEL * dt
        elif self.speed > 0:
            self.speed = max(0.0, self.speed - config.RACER_FRICTION_DECEL * dt)
        elif self.speed < 0:
            self.speed = min(0.0, self.speed + config.RACER_FRICTION_DECEL * dt)
        self.speed = max(config.RACER_MIN_SPEED, min(self.max_speed, self.speed))

    def _update_heading(self, dt: float, controls: RacerControls) -> None:
        power = controls.steer_power
        if self.speed == 0 or power <= 0 or not (controls.steer_left or controls.steer_right):
            return
        speed_fraction = min(abs(self.speed) / config.RACER_MAX_SPEED, 1.0)
        turn_rate = config.RACER_MAX_TURN_RATE - speed_fraction * (
            config.RACER_MAX_TURN_RATE - config.RACER_MIN_TURN_RATE
        )
        if controls.steer_left:
            self.heading -= turn_rate * power * dt
        if controls.steer_right:
            self.heading += turn_rate * power * dt
