from __future__ import annotations

import math
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pygame

from . import config, geometry
from .editor import EditorMode, TrackEditor
from .network import AimlessDriver, Network, controls_from_output
from .racer import Racer, RacerControls
from .sensors import cast_rays, ray_angles
from .track import Track

_MODE_HINTS = {
    EditorMode.PLACE_START: "Click to place the START point",
    EditorMode.PLACE_FINISH: "Click to place the FINISH point",
    EditorMode.READY: "Click and drag to draw obstacles",
    EditorMode.DRAWING: "Drawing obstacle...",
}

_HELP_TEXT = (
    "S save   L load   U undo stroke   C clear obstacles   "
    "R reset track   1 move start   2 move finish   T test drive   Esc quit"
)

_DRIVE_HELP_TEXT = (
    "Up/Down throttle+brake   Left/Right steer   A toggle AI   N new weights   Esc back to editor"
)


class FileMode(Enum):
    NONE = auto()
    SAVE = auto()
    LOAD = auto()


class AppMode(Enum):
    EDITING = auto()
    DRIVING = auto()


class App:
    """Top-level application: game loop + input routing + rendering."""

    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.big_font = pygame.font.SysFont("consolas", 22, bold=True)

        self.editor = TrackEditor()
        self.running = True

        self.mode = AppMode.EDITING
        self.racer: Optional[Racer] = None
        self.ray_distances: list[float] = []
        self.network: Optional[Network] = None
        self.driver: Optional[AimlessDriver] = None
        self.ai_enabled = False
        self.ai_output: Optional[tuple[float, float]] = None

        self.file_mode = FileMode.NONE
        self.input_text = ""
        self.status_message = ""
        self.status_timer = 0.0

        self.tracks_dir = Path(config.TRACKS_DIR)
        self.tracks_dir.mkdir(exist_ok=True)

    # main loop

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(config.FPS) / 1000.0
            self._handle_events()
            self._update(dt)
            self._draw()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif self.file_mode is not FileMode.NONE:
                self._handle_file_prompt_event(event)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._on_escape()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                self._toggle_drive_mode()
            elif self.mode is AppMode.DRIVING and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self._toggle_ai()
                elif event.key == pygame.K_n:
                    self._reroll_weights()
            elif self.mode is AppMode.EDITING:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self._begin_file_prompt(FileMode.SAVE)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                    self._begin_file_prompt(FileMode.LOAD)
                else:
                    self.editor.handle_event(event)

    def _on_escape(self) -> None:
        if self.mode is AppMode.DRIVING:
            self._toggle_drive_mode()
        else:
            self.running = False

    def _toggle_drive_mode(self) -> None:
        if self.mode is AppMode.DRIVING:
            self.mode = AppMode.EDITING
            self.racer = None
            self.ray_distances = []
            self.network = None
            self.driver = None
            self.ai_enabled = False
            self.ai_output = None
            return
        track = self.editor.track
        if not track.is_complete():
            self._set_status("Add a start, finish, and an obstacle before test driving")
            return
        self.racer = Racer(track.start[0], track.start[1], self._spawn_heading())
        self.network = Network()
        self.driver = AimlessDriver(self.network)
        self.ai_enabled = False
        self.ai_output = None
        self.ray_distances = cast_rays(self.racer.position, self.racer.heading, track.segments())
        self.mode = AppMode.DRIVING

    def _spawn_heading(self) -> float:
        track = self.editor.track
        return math.atan2(track.finish[1] - track.start[1], track.finish[0] - track.start[0])

    def _toggle_ai(self) -> None:
        self.ai_enabled = not self.ai_enabled
        self.ai_output = None
        self._set_status("AI control" if self.ai_enabled else "Manual control")

    def _reroll_weights(self) -> None:
        if self.network is None or self.racer is None:
            return
        self.network.randomize()
        self.driver = AimlessDriver(self.network)
        self.racer.reset(self.editor.track.start[0], self.editor.track.start[1], self._spawn_heading())
        self.ray_distances = cast_rays(
            self.racer.position, self.racer.heading, self.editor.track.segments()
        )
        self.ai_output = None
        self._set_status("New random weights")

    def _update(self, dt: float) -> None:
        if self.status_timer > 0:
            self.status_timer -= dt
            if self.status_timer <= 0:
                self.status_message = ""
        if self.mode is AppMode.DRIVING and self.racer is not None:
            self._update_racer(dt)

    def _update_racer(self, dt: float) -> None:
        self.racer.step(dt, self._current_controls())
        track = self.editor.track
        if geometry.circle_hits_any_segment(
            self.racer.position, config.RACER_RADIUS, track.segments()
        ):
            self.racer.reset(track.start[0], track.start[1], self._spawn_heading())
            self._set_status("Crashed! Resetting to start.")
        self.ray_distances = cast_rays(self.racer.position, self.racer.heading, track.segments())

    def _current_controls(self) -> RacerControls:
        if self.ai_enabled and self.driver is not None:
            output = self.driver.output(self.ray_distances, self.racer.speed)
            self.ai_output = (float(output[0]), float(output[1]))
            return controls_from_output(output)
        self.ai_output = None
        keys = pygame.key.get_pressed()
        return RacerControls(
            throttle=keys[pygame.K_UP],
            brake=keys[pygame.K_DOWN],
            steer_left=keys[pygame.K_LEFT],
            steer_right=keys[pygame.K_RIGHT],
        )

    # save / load text prompt

    def _begin_file_prompt(self, mode: FileMode) -> None:
        self.file_mode = mode
        self.input_text = config.DEFAULT_TRACK_NAME
        pygame.key.start_text_input()

    def _handle_file_prompt_event(self, event: "pygame.event.Event") -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._end_file_prompt()
            elif event.key == pygame.K_RETURN:
                self._confirm_file_prompt()
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
        elif event.type == pygame.TEXTINPUT:
            self.input_text += event.text

    def _end_file_prompt(self) -> None:
        self.file_mode = FileMode.NONE
        self.input_text = ""
        pygame.key.stop_text_input()

    def _confirm_file_prompt(self) -> None:
        name = self.input_text.strip() or config.DEFAULT_TRACK_NAME
        path = self.tracks_dir / f"{name}.json"
        if self.file_mode is FileMode.SAVE:
            self.editor.track.save(path)
            self._set_status(f"Saved track to {path}")
        elif self.file_mode is FileMode.LOAD:
            if path.exists():
                self.editor.track = Track.load(path)
                self._set_status(f"Loaded track from {path}")
            else:
                self._set_status(f"No track file found at {path}")
        self._end_file_prompt()

    def _set_status(self, message: str) -> None:
        self.status_message = message
        self.status_timer = 3.0

    def _existing_track_names(self) -> list[str]:
        return sorted(p.stem for p in self.tracks_dir.glob("*.json"))

    # drawing

    def _draw(self) -> None:
        self.screen.fill(config.BACKGROUND_COLOR)
        self._draw_grid()
        self._draw_boundaries()
        self._draw_start_finish()
        if self.mode is AppMode.DRIVING and self.racer is not None:
            self._draw_racer()
        self._draw_hud()
        if self.file_mode is not FileMode.NONE:
            self._draw_file_prompt()
        pygame.display.flip()

    def _draw_racer(self) -> None:
        racer = self.racer
        self._draw_rays()
        color = config.RACER_AI_COLOR if self.ai_enabled else config.RACER_COLOR
        pygame.draw.circle(self.screen, color, racer.position, config.RACER_RADIUS)
        nose = (
            racer.x + math.cos(racer.heading) * config.RACER_RADIUS * 1.8,
            racer.y + math.sin(racer.heading) * config.RACER_RADIUS * 1.8,
        )
        pygame.draw.line(self.screen, color, racer.position, nose, 2)

    def _draw_rays(self) -> None:
        racer = self.racer
        for angle, distance in zip(ray_angles(racer.heading), self.ray_distances):
            end = (
                racer.x + math.cos(angle) * distance,
                racer.y + math.sin(angle) * distance,
            )
            hit = distance < config.RAY_MAX_DISTANCE - 0.5
            color = config.RAY_HIT_COLOR if hit else config.RAY_COLOR
            pygame.draw.line(self.screen, color, racer.position, end, 1)
            if hit:
                pygame.draw.circle(self.screen, color, end, 3)

    def _draw_grid(self) -> None:
        for x in range(0, config.SCREEN_WIDTH, config.GRID_SPACING):
            pygame.draw.line(self.screen, config.GRID_COLOR, (x, 0), (x, config.SCREEN_HEIGHT))
        for y in range(0, config.SCREEN_HEIGHT, config.GRID_SPACING):
            pygame.draw.line(self.screen, config.GRID_COLOR, (0, y), (config.SCREEN_WIDTH, y))

    def _draw_boundaries(self) -> None:
        for stroke in self.editor.track.boundaries:
            if len(stroke) >= 2:
                pygame.draw.lines(self.screen, config.BOUNDARY_COLOR, False, stroke, config.BOUNDARY_LINE_WIDTH)
        stroke = self.editor.current_stroke
        if stroke and len(stroke) >= 2:
            pygame.draw.lines(self.screen, config.IN_PROGRESS_COLOR, False, stroke, config.BOUNDARY_LINE_WIDTH)

    def _draw_start_finish(self) -> None:
        track = self.editor.track
        if track.start is not None:
            pygame.draw.circle(self.screen, config.START_COLOR, track.start, config.POINT_RADIUS)
            self._draw_label(track.start, "START", config.START_COLOR)
        if track.finish is not None:
            pygame.draw.circle(self.screen, config.FINISH_COLOR, track.finish, config.POINT_RADIUS)
            self._draw_label(track.finish, "FINISH", config.FINISH_COLOR)

    def _draw_label(self, pos: tuple[float, float], text: str, color: tuple[int, int, int]) -> None:
        surf = self.font.render(text, True, color)
        self.screen.blit(surf, (pos[0] + config.POINT_RADIUS + 6, pos[1] - surf.get_height() // 2))

    def _draw_hud(self) -> None:
        lines = [self._mode_hint(), self._help_text()]
        if self.mode is AppMode.DRIVING and self.ray_distances:
            lines.append(self._ray_readout())
        y = 10
        for text in lines:
            surf = self.font.render(text, True, config.TEXT_COLOR)
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 4

        if self.status_message:
            surf = self.font.render(self.status_message, True, config.STATUS_COLOR)
            self.screen.blit(surf, (10, config.SCREEN_HEIGHT - surf.get_height() - 10))

    def _mode_hint(self) -> str:
        if self.mode is AppMode.DRIVING and self.racer is not None:
            who = "AI" if self.ai_enabled else "manual"
            hint = f"Test drive ({who}) — speed {self.racer.speed:.0f} px/s"
            if self.ai_output is not None:
                steer, throttle = self.ai_output
                hint += f"   steer {steer:+.2f}   throttle {throttle:+.2f}"
            return hint
        return _MODE_HINTS[self.editor.mode]

    def _help_text(self) -> str:
        return _DRIVE_HELP_TEXT if self.mode is AppMode.DRIVING else _HELP_TEXT

    def _ray_readout(self) -> str:
        return "Rays: " + "  ".join(f"{d:.0f}" for d in self.ray_distances)

    def _draw_file_prompt(self) -> None:
        label = "Save track as: " if self.file_mode is FileMode.SAVE else "Load track: "
        text = f"{label}{self.input_text}_"
        text_surf = self.big_font.render(text, True, config.TEXT_COLOR)

        existing = self._existing_track_names()
        hint = "Existing: " + ", ".join(existing) if existing else "No saved tracks yet"
        hint_surf = self.font.render(hint, True, config.TEXT_COLOR)

        box_height = text_surf.get_height() + hint_surf.get_height() + 28
        box_rect = pygame.Rect(0, 0, config.SCREEN_WIDTH, box_height)
        box_rect.centery = config.SCREEN_HEIGHT // 2

        pygame.draw.rect(self.screen, config.HUD_BG_COLOR, box_rect)
        pygame.draw.rect(self.screen, config.BOUNDARY_COLOR, box_rect, 2)
        self.screen.blit(text_surf, (20, box_rect.y + 10))
        self.screen.blit(hint_surf, (20, box_rect.y + 14 + text_surf.get_height()))
