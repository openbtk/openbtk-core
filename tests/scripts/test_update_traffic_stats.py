"""Tests for scripts/update_traffic_stats.py -- not part of the openbtk package,
but its merge/totals logic is exactly the kind of thing that silently double-counts
or drops a day if it's wrong, so it gets the same real tests as library code.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "update_traffic_stats.py"
)
_SPEC = importlib.util.spec_from_file_location("update_traffic_stats", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
stats = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = stats
_SPEC.loader.exec_module(stats)


def _daily(date: str, count: int, uniques: int) -> dict[str, object]:
    return {"timestamp": f"{date}T00:00:00Z", "count": count, "uniques": uniques}


class TestMerge:
    def test_a_new_date_is_added(self) -> None:
        history: dict[str, dict[str, int]] = {}
        stats._merge(history, [_daily("2026-09-08", 5, 2)])
        assert history == {"2026-09-08": {"count": 5, "uniques": 2}}

    def test_re_fetching_the_same_day_overwrites_rather_than_adds(self) -> None:
        history = {"2026-09-08": {"count": 5, "uniques": 2}}
        # GitHub's rolling window returns this date again on the next run, with
        # a since-updated count for "today". Re-merging must replace, not add.
        stats._merge(history, [_daily("2026-09-08", 9, 3)])
        assert history == {"2026-09-08": {"count": 9, "uniques": 3}}

    def test_a_day_that_scrolled_out_of_the_window_is_kept(self) -> None:
        history = {"2026-09-01": {"count": 3, "uniques": 1}}
        stats._merge(history, [_daily("2026-09-15", 4, 2)])
        assert history == {
            "2026-09-01": {"count": 3, "uniques": 1},
            "2026-09-15": {"count": 4, "uniques": 2},
        }


class TestTotals:
    def test_sums_counts_and_uniques_independently(self) -> None:
        history = {
            "2026-09-01": {"count": 3, "uniques": 1},
            "2026-09-02": {"count": 7, "uniques": 4},
        }
        assert stats._totals(history) == (10, 5)

    def test_empty_history_totals_to_zero(self) -> None:
        assert stats._totals({}) == (0, 0)


class TestBadge:
    def test_matches_the_shields_io_endpoint_schema(self) -> None:
        badge = stats._badge("views", "1,234", "blue")
        assert badge == {
            "schemaVersion": 1,
            "label": "views",
            "message": "1,234",
            "color": "blue",
        }


class TestLoadHistory:
    def test_missing_file_starts_empty(self, tmp_path: Path) -> None:
        assert stats._load_history(tmp_path / "history.json") == {
            "views": {},
            "clones": {},
        }

    def test_existing_file_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        saved = {"views": {"2026-09-01": {"count": 1, "uniques": 1}}, "clones": {}}
        path.write_text(json.dumps(saved), encoding="utf-8")
        assert stats._load_history(path) == saved


class TestMainEndToEnd:
    def test_a_full_run_merges_fetches_and_writes_badges(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = {
            "views": {"2026-09-01": {"count": 2, "uniques": 1}},
            "clones": {"2026-09-01": {"count": 1, "uniques": 1}},
        }
        (tmp_path / "history.json").write_text(json.dumps(existing), encoding="utf-8")

        fetched = {
            f"{stats.API_ROOT}/repos/acme/widgets/traffic/views": {
                "views": [_daily("2026-09-02", 5, 3)]
            },
            f"{stats.API_ROOT}/repos/acme/widgets/traffic/clones": {
                "clones": [_daily("2026-09-02", 10, 6)]
            },
        }
        monkeypatch.setattr(stats, "_get", lambda url, token: fetched[url])
        monkeypatch.setenv("TRAFFIC_TOKEN", "fake-token")
        monkeypatch.setenv("GITHUB_REPOSITORY", "acme/widgets")
        monkeypatch.setenv("TRAFFIC_DATA_DIR", str(tmp_path))

        assert stats.main() == 0

        history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
        assert history["views"]["2026-09-01"]["count"] == 2  # older day untouched
        assert history["views"]["2026-09-02"]["count"] == 5  # new day added

        views_badge = json.loads(
            (tmp_path / "badges" / "views.json").read_text(encoding="utf-8")
        )
        assert views_badge["message"] == "7"  # 2 (09-01) + 5 (09-02)
        clones_badge = json.loads(
            (tmp_path / "badges" / "clones.json").read_text(encoding="utf-8")
        )
        assert clones_badge["message"] == "11"  # 1 + 10

    def test_missing_env_vars_fail_fast_with_exit_code_2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TRAFFIC_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        assert stats.main() == 2
