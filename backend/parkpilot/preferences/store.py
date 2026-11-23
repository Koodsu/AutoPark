"""Preference repositories: user settings plus per-location favorite spots (C15).

When the user overrides the chosen spot at a location, the override is recorded as a
favorite position for that location. On the next visit, the ranking gives stalls near a
favorite a bonus. The interface is a ``Protocol`` so the JSON file store can later be
swapped for a database or a cloud profile without touching the autonomy stack.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from parkpilot.core.config import UserSettings


class Favorite(BaseModel):
    x: float
    y: float
    count: int = 1


class LocationPreference(BaseModel):
    location_id: str
    favorites: list[Favorite] = Field(default_factory=list)

    def positions(self) -> list[tuple[float, float]]:
        return [(f.x, f.y) for f in self.favorites]


class PreferenceDocument(BaseModel):
    """The whole persisted document (versioned for future migrations)."""

    version: int = 1
    settings: UserSettings = UserSettings()
    locations: dict[str, LocationPreference] = Field(default_factory=dict)


class PreferenceRepository(Protocol):
    def get_settings(self) -> UserSettings: ...

    def save_settings(self, settings: UserSettings) -> None: ...

    def get_location(self, location_id: str) -> LocationPreference: ...

    def record_override(self, location_id: str, x: float, y: float) -> LocationPreference: ...

    def clear_location(self, location_id: str) -> None: ...

    def list_locations(self) -> list[LocationPreference]: ...


MERGE_RADIUS = 1.5
"""Overrides within this distance of an existing favorite reinforce it (m)."""


class InMemoryPreferenceStore:
    def __init__(self, doc: PreferenceDocument | None = None) -> None:
        self._doc = doc or PreferenceDocument()

    def _changed(self) -> None:
        """Hook for persistent subclasses."""

    def get_settings(self) -> UserSettings:
        return self._doc.settings

    def save_settings(self, settings: UserSettings) -> None:
        self._doc.settings = settings
        self._changed()

    def get_location(self, location_id: str) -> LocationPreference:
        return self._doc.locations.get(location_id, LocationPreference(location_id=location_id))

    def record_override(self, location_id: str, x: float, y: float) -> LocationPreference:
        loc = self._doc.locations.setdefault(
            location_id, LocationPreference(location_id=location_id)
        )
        for fav in loc.favorites:
            if math.hypot(fav.x - x, fav.y - y) <= MERGE_RADIUS:
                n = fav.count + 1
                fav.x += (x - fav.x) / n
                fav.y += (y - fav.y) / n
                fav.count = n
                break
        else:
            loc.favorites.append(Favorite(x=x, y=y))
        loc.favorites.sort(key=lambda f: -f.count)
        self._changed()
        return loc

    def clear_location(self, location_id: str) -> None:
        self._doc.locations.pop(location_id, None)
        self._changed()

    def list_locations(self) -> list[LocationPreference]:
        return list(self._doc.locations.values())


class JsonFilePreferenceStore(InMemoryPreferenceStore):
    """Persists the document as JSON, writing atomically (temp file + rename)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        doc = None
        if path.is_file():
            doc = PreferenceDocument.model_validate_json(path.read_text(encoding="utf-8"))
        super().__init__(doc)

    def _changed(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self._doc.model_dump(mode="json"), indent=2, sort_keys=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".prefs-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data + "\n")
        Path(tmp).replace(self.path)
