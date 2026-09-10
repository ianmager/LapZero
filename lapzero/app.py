from __future__ import annotations

from enum import Enum, auto
from pathlib import Path
from typing import Optional

import pygame

from . import config
from .editor import EditorMode, TrackEditor
from .track import Track

_MODE_HINTS = {
    EditorMode.PLACE_START: "Click to place the START point",
    EditorMode.PLACE_FINISH: "Click to place the FINISH point",
    EditorMode.READY: "Click and drag to draw obstacles",
    EditorMode.DRAWING: "Drawing obstacle...",
}

_HELP_TEXT = (
    "S save   L load   U undo stroke   C clear obstacles   "
    "R reset track   1 move start   2 move finish   Esc quit"
)


class FileMode(Enum):
    NONE = auto()
    SAVE = auto()
    LOAD = auto()


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
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self._begin_file_prompt(FileMode.SAVE)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                self._begin_file_prompt(FileMode.LOAD)
            else:
                self.editor.handle_event(event)

    def _update(self, dt: float) -> None:
        if self.status_timer > 0:
            self.status_timer -= dt
            if self.status_timer <= 0:
                self.status_message = ""

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
        self._draw_hud()
        if self.file_mode is not FileMode.NONE:
            self._draw_file_prompt()
        pygame.display.flip()

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
        lines = [self._mode_hint(), _HELP_TEXT]
        y = 10
        for text in lines:
            surf = self.font.render(text, True, config.TEXT_COLOR)
            self.screen.blit(surf, (10, y))
            y += surf.get_height() + 4

        if self.status_message:
            surf = self.font.render(self.status_message, True, config.STATUS_COLOR)
            self.screen.blit(surf, (10, config.SCREEN_HEIGHT - surf.get_height() - 10))

    def _mode_hint(self) -> str:
        return _MODE_HINTS[self.editor.mode]

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
