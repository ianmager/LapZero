from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass
class Track:
    start: Optional[Point] = None
    finish: Optional[Point] = None
    boundaries: list[list[Point]] = field(default_factory=list)

    def is_complete(self) -> bool:
        """True once a start, a finish, and at least one boundary stroke exist."""
        return self.start is not None and self.finish is not None and bool(self.boundaries)

    def segments(self) -> Iterator[Segment]:
        """Yield every wall segment (pair of consecutive points) across all strokes."""
        for stroke in self.boundaries:
            for a, b in zip(stroke, stroke[1:]):
                yield (a, b)

    def clear_boundaries(self) -> None:
        self.boundaries.clear()

    def reset(self) -> None:
        self.start = None
        self.finish = None
        self.boundaries.clear()

    #serialization

    def to_dict(self) -> dict:
        return {
            "start": list(self.start) if self.start is not None else None,
            "finish": list(self.finish) if self.finish is not None else None,
            "boundaries": [[list(point) for point in stroke] for stroke in self.boundaries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Track":
        start = tuple(data["start"]) if data.get("start") is not None else None
        finish = tuple(data["finish"]) if data.get("finish") is not None else None
        boundaries = [[tuple(point) for point in stroke] for stroke in data.get("boundaries", [])]
        return cls(start=start, finish=finish, boundaries=boundaries)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Track":
        path = Path(path)
        with path.open("r") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
