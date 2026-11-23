from pathlib import Path

from parkpilot.core.config import DirectionPreference, Unlock, UserSettings
from parkpilot.preferences.store import (
    InMemoryPreferenceStore,
    JsonFilePreferenceStore,
    PreferenceRepository,
)


def _exercise(store: PreferenceRepository) -> None:
    assert store.get_location("lot").favorites == []
    store.record_override("lot", 10.0, 5.0)
    store.record_override("lot", 10.4, 5.2)  # reinforces the same favorite
    store.record_override("lot", 30.0, 5.0)
    loc = store.get_location("lot")
    assert [f.count for f in loc.favorites] == [2, 1]
    assert abs(loc.favorites[0].x - 10.2) < 1e-9
    assert loc.positions()[1] == (30.0, 5.0)
    store.save_settings(
        UserSettings(direction=DirectionPreference.REVERSE_IN, unlocks=frozenset({Unlock.PLACARD}))
    )
    assert store.get_settings().direction is DirectionPreference.REVERSE_IN
    assert [x.location_id for x in store.list_locations()] == ["lot"]


def test_in_memory_store() -> None:
    store = InMemoryPreferenceStore()
    _exercise(store)
    store.clear_location("lot")
    assert store.list_locations() == []


def test_json_store_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "prefs" / "prefs.json"
    store = JsonFilePreferenceStore(path)
    _exercise(store)
    reloaded = JsonFilePreferenceStore(path)
    assert reloaded.get_location("lot").favorites[0].count == 2
    assert reloaded.get_settings().unlocks == frozenset({Unlock.PLACARD})
    reloaded.clear_location("lot")
    assert JsonFilePreferenceStore(path).list_locations() == []
