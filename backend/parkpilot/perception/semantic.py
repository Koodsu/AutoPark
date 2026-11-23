"""Accumulated semantic map: painted-line landmarks and sign/paint/curb feature evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from parkpilot.core.geometry import FloatArray
from parkpilot.perception.camera import SensorFrame
from parkpilot.world.model import FeatureKind


@dataclass(slots=True)
class LineLandmark:
    id: str
    a: FloatArray
    b: FloatArray
    count: int = 1

    @property
    def mid(self) -> FloatArray:
        return np.asarray((self.a + self.b) / 2)

    @property
    def direction(self) -> FloatArray:
        d = self.b - self.a
        return np.asarray(d / max(float(np.hypot(*d)), 1e-9))

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.b - self.a)))


@dataclass(slots=True)
class FeatureEvidence:
    kind: FeatureKind
    label: str
    position: FloatArray
    count: int = 1


@dataclass(slots=True)
class SemanticMap:
    """Running averages of line observations and clustered feature sightings."""

    merge_radius: float = 1.0
    lines: dict[str, LineLandmark] = field(default_factory=dict)
    features: list[FeatureEvidence] = field(default_factory=list)

    def update(self, frame: SensorFrame) -> None:
        for obs in frame.lines:
            a, b = np.array(obs.a), np.array(obs.b)
            lm = self.lines.get(obs.line_id)
            if lm is None:
                self.lines[obs.line_id] = LineLandmark(obs.line_id, a, b)
                continue
            # Endpoints may come back swapped; align before averaging.
            if np.hypot(*(a - lm.a)) > np.hypot(*(b - lm.a)):
                a, b = b, a
            n = lm.count + 1
            lm.a = lm.a + (a - lm.a) / n
            lm.b = lm.b + (b - lm.b) / n
            lm.count = n
        for f in frame.features:
            pos = np.array(f.position)
            for ev in self.features:
                if (
                    ev.kind is f.kind
                    and ev.label == f.label
                    and math.hypot(*(ev.position - pos)) <= self.merge_radius
                ):
                    ev.count += 1
                    ev.position = ev.position + (pos - ev.position) / ev.count
                    break
            else:
                self.features.append(FeatureEvidence(f.kind, f.label, pos))
