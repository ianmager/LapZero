from __future__ import annotations

import math
from enum import Enum, auto
from typing import Optional

import pygame

from .config import MIN_STROKE_POINT_DISTANCE
from .track import Point, Track


class EditorMode(Enum):
    PLACE_START = auto()
    PLACE_FINISH = auto()
    READY = auto()       # start + finish placed click-drag to draw a wall
    DRAWING = auto()     # mouse button currently held, actively adding points


class TrackEditor:

    def __init__(self, track: Optional[Track] = None) -> None:
        self.track: Track = track if track is not None else Track()
        self._current_stroke: Optional[list[Point]] = None
        self._relocate_start = False
        self._relocate_finish = False

    @property
    def mode(self) -> EditorMode:
        if self._current_stroke is not None:
            return EditorMode.DRAWING
        if self._relocate_start or self.track.start is None:
            return EditorMode.PLACE_START
        if self._relocate_finish or self.track.finish is None:
            return EditorMode.PLACE_FINISH
        return EditorMode.READY

    @property
    def current_stroke(self) -> Optional[list[Point]]:
        return self._current_stroke

    def handle_event(self, event: "pygame.event.Event") -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_mouse_down(event.pos)
        elif event.type == pygame.MOUSEMOTION:
            self._on_mouse_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._on_mouse_up(event.pos)
        elif event.type == pygame.KEYDOWN:
            self._on_key_down(event.key)

    # mouse

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        mode = self.mode
        if mode is EditorMode.PLACE_START:
            self.track.start = pos
            self._relocate_start = False
        elif mode is EditorMode.PLACE_FINISH:
            self.track.finish = pos
            self._relocate_finish = False
        elif mode is EditorMode.READY:
            self._current_stroke = [pos]

    def _on_mouse_motion(self, pos: tuple[int, int]) -> None:
        if self._current_stroke is None:
            return
        if math.dist(self._current_stroke[-1], pos) >= MIN_STROKE_POINT_DISTANCE:
            self._current_stroke.append(pos)

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self._current_stroke is None:
            return
        if self._current_stroke[-1] != pos:
            self._current_stroke.append(pos)
        if len(self._current_stroke) >= 2:
            self.track.boundaries.append(self._current_stroke)
        self._current_stroke = None

    #keyboard

    def _on_key_down(self, key: int) -> None:
        if key == pygame.K_u:
            self.undo_last_stroke()
        elif key == pygame.K_c:
            self.track.clear_boundaries()
        elif key == pygame.K_r:
            self.reset()
        elif key == pygame.K_1:
            self._relocate_start = True
        elif key == pygame.K_2 and self.track.start is not None:
            self._relocate_finish = True

    #actions

    def undo_last_stroke(self) -> None:
        if self.track.boundaries:
            self.track.boundaries.pop()

    def reset(self) -> None:
        self.track.reset()
        self._current_stroke = None
        self._relocate_start = False
        self._relocate_finish = False
