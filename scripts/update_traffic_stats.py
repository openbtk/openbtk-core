#!/usr/bin/env python3
"""Fetch this repo's GitHub traffic stats and fold them into a running history.

GitHub's traffic API (``/traffic/views``, ``/traffic/clones``) only ever returns a
rolling 14-day window and needs a token with push access to the repo -- there is no
public, unauthenticated way to read it and no built-in way to see totals older than
14 days. This script is run on a schedule (see ``.github/workflows/traffic-stats.yml``)
against a checkout of the ``metrics-data`` branch, where it keeps ``history.json`` (one
entry per calendar day, upserted -- a day already recorded is only ever overwritten,
never double-counted) and regenerates a shields.io "endpoint badge" JSON file per
metric so the README can show a live total without embedding this repo's own secrets
or requiring a viewer to authenticate.

"Unique visitors" and "unique cloners" are GitHub's per-day unique counts, summed
across days -- the same person visiting on two different days counts twice. That is
what GitHub itself gives us; it is not a true all-time distinct-visitor count, and the
badge label says "visits" rather than "visitors" to avoid overclaiming.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://api.github.com"


def _get(url: str, token: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "openbtk-traffic-stats",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def _load_history(path: Path) -> dict[str, dict[str, dict[str, int]]]:
    if not path.exists():
        return {"views": {}, "clones": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge(history: dict[str, dict[str, int]], daily: list[dict[str, Any]]) -> None:
    for entry in daily:
        date = entry["timestamp"][:10]
        history[date] = {"count": entry["count"], "uniques": entry["uniques"]}


def _totals(history: dict[str, dict[str, int]]) -> tuple[int, int]:
    count = sum(day["count"] for day in history.values())
    uniques = sum(day["uniques"] for day in history.values())
    return count, uniques


def _badge(label: str, message: str, color: str) -> dict[str, str]:
    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def main() -> int:
    token = os.environ.get("TRAFFIC_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    data_dir = Path(os.environ.get("TRAFFIC_DATA_DIR", "."))
    if not token or not repo:
        print("TRAFFIC_TOKEN and GITHUB_REPOSITORY must be set", file=sys.stderr)
        return 2

    history_path = data_dir / "history.json"
    history = _load_history(history_path)

    try:
        views = _get(f"{API_ROOT}/repos/{repo}/traffic/views", token)
        clones = _get(f"{API_ROOT}/repos/{repo}/traffic/clones", token)
    except urllib.error.HTTPError as exc:
        print(f"GitHub traffic API request failed: {exc}", file=sys.stderr)
        return 1

    _merge(history["views"], views["views"])
    _merge(history["clones"], clones["clones"])
    history_path.write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    view_count, view_uniques = _totals(history["views"])
    clone_count, clone_uniques = _totals(history["clones"])

    badges_dir = data_dir / "badges"
    badges_dir.mkdir(exist_ok=True)
    badges = {
        "views.json": _badge("views", f"{view_count:,}", "blue"),
        "visits.json": _badge("visits", f"{view_uniques:,}", "blue"),
        "clones.json": _badge("clones", f"{clone_count:,}", "informational"),
        "cloners.json": _badge("clone visits", f"{clone_uniques:,}", "informational"),
    }
    for name, payload in badges.items():
        (badges_dir / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    print(
        f"views={view_count} visits={view_uniques} clones={clone_count} "
        f"clone_visits={clone_uniques}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
