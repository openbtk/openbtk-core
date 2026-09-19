"""In-process tests for the CLI's pure pieces: manifest comparison, the doctor
report, and the JSON/text branches of each command. (test_cli.py drives the
real command line, including through a subprocess; coverage cannot see
into a subprocess, so the branches are also exercised here.)"""

from __future__ import annotations

import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openbtk.cli import doctor, main
from openbtk.cli.commands import _compare, _redacted_paths
from openbtk.core.provenance import (
    ComponentProvenance,
    DataDigest,
    RunManifest,
    StepProvenance,
)

if TYPE_CHECKING:
    import pytest

    from openbtk.core.provenance import RunStatus

_ROOT = Path(__file__).resolve().parents[3]
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _json_from(out: str) -> Any:
    """The library's log lines are single-line JSON; a command's own JSON is
    indented, so it starts at the first line that is exactly ``{`` or ``[``."""
    lines = out.splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() in ("{", "["))
    return json.loads("".join(lines[start:]))


def _step(step_id: str, n_in: int, n_out: int) -> StepProvenance:
    return StepProvenance(
        step_id=step_id,
        component=ComponentProvenance(
            registry_key="x.y.z", class_name="C", package_version="0"
        ),
        records_in=n_in,
        records_out=n_out,
        status="success",
    )


def _manifest(
    *,
    steps: list[StepProvenance] | None = None,
    digests: list[DataDigest] | None = None,
    status: RunStatus = "success",
    config: dict[str, Any] | None = None,
) -> RunManifest:
    return RunManifest(
        run_id="r",
        status=status,
        config=config or {"name": "p", "steps": []},
        started_at=_NOW,
        steps=steps or [],
        input_digests=digests or [],
    )


class TestCompare:
    def test_identical_runs_have_no_differences(self) -> None:
        m = _manifest(
            steps=[_step("a", 0, 3)],
            digests=[DataDigest(uri="f", sha256="h", record_count=3)],
        )
        assert _compare(m, m) == []

    def test_each_kind_of_difference_is_named(self) -> None:
        old = _manifest(
            steps=[_step("a", 0, 3), _step("gone", 3, 3)],
            digests=[
                DataDigest(uri="same", sha256="h", record_count=3),
                DataDigest(uri="changed", sha256="h1", record_count=3),
                DataDigest(uri="count", sha256="h", record_count=3),
                DataDigest(uri="unhashed", sha256=None, record_count=3),
                DataDigest(uri="dropped", sha256="h", record_count=3),
            ],
        )
        new = _manifest(
            steps=[_step("a", 0, 5)],
            digests=[
                DataDigest(uri="same", sha256="h", record_count=3),
                DataDigest(uri="changed", sha256="h2", record_count=3),
                DataDigest(uri="count", sha256="h", record_count=4),
                DataDigest(uri="unhashed", sha256=None, record_count=3),
            ],
            status="failed",
        )
        text = "\n".join(_compare(old, new))
        assert "input changed: content changed" in text
        assert "input count: 3 record(s) then, 4 now" in text
        assert "input unhashed: not hashed" in text
        assert "input dropped: no longer read" in text
        assert "step a: 0->3 then, 0->5 now" in text
        assert "step gone: missing from the replay" in text
        assert "status: success then, failed now" in text
        assert "input same" not in text


class TestRedactedPaths:
    def test_finds_nested_and_listed_redactions(self) -> None:
        cfg = {
            "steps": [{"params": {"api_key": "[REDACTED]", "ok": 1}}],
            "k": "[REDACTED]",
        }
        assert sorted(_redacted_paths(cfg)) == ["k", "steps[0].params.api_key"]

    def test_a_clean_config_has_none(self) -> None:
        assert _redacted_paths({"a": [1, {"b": "x"}]}) == []


class TestDoctor:
    def test_extra_module_map_matches_pyproject(self) -> None:
        """Every installable extra the project declares (other than tooling
        and the meta-extra) has a probe, and no probe is for an extra that
        does not exist -- a new extra cannot be forgotten here."""
        data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(data["project"]["optional-dependencies"]) - {
            "dev",
            "docs",
            "all",
        }
        assert set(doctor.EXTRA_MODULES) == declared - {"notebooks"}

    def test_credentials_are_reported_by_name_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in doctor.CREDENTIAL_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(
            "HF_TOKEN", "hf_value_that_must_not_leak"
        )  # pragma: allowlist secret
        report = doctor.collect()
        assert report["credentials"]["HF_TOKEN"] is True
        assert report["credentials"]["OPENAI_API_KEY"] is False
        assert "hf_value_that_must_not_leak" not in json.dumps(report)
        assert "hf_value_that_must_not_leak" not in doctor.format_text(report)

    def test_missing_extras_produce_install_hints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor, "_has_module", lambda name: False)
        monkeypatch.setattr(doctor, "_dist_version", lambda name: None)
        report = doctor.collect()
        assert all(not e["installed"] for e in report["extras"].values())
        assert any("pip install 'openbtk[ehr]'" in h for h in report["hints"])
        text = doctor.format_text(report)
        assert "MISSING" in text and "missing" in text and "To fix" in text
        assert report["openbtk"].startswith("unknown")

    def test_the_spacy_model_hint_appears_only_when_text_is_ready_without_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            doctor, "_has_module", lambda name: name != doctor.SPACY_MODEL
        )
        report = doctor.collect()
        assert any("spacy download" in h for h in report["hints"])
        assert report["models"] == {f"spacy:{doctor.SPACY_MODEL}": False}

    def test_a_fully_provisioned_environment_has_no_hints(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(doctor, "_has_module", lambda name: True)
        report = doctor.collect()
        assert report["hints"] == []
        assert "To fix" not in doctor.format_text(report)

    def test_a_broken_parent_package_counts_as_missing(self) -> None:
        assert doctor._has_module("definitely_not_a_module.sub") is False

    def test_unmet_names_unknown_extras(self) -> None:
        report = doctor.collect()
        assert doctor.unmet(report, ["nope"]) == ["nope"]
        assert doctor.unmet(report, []) == []


class TestCommandBranches:
    def test_list_a_category_as_text_and_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["list", "guardrail"]) == 0
        assert "guardrail.general.phi_leakage" in capsys.readouterr().out
        assert main(["list", "guardrail", "--json"]) == 0
        rows = _json_from(capsys.readouterr().out)
        assert {"key", "class", "module", "sends_data_offsite", "summary"} <= set(
            rows[0]
        )

    def test_list_json_of_the_categories(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["list", "--json"]) == 0
        counts = _json_from(capsys.readouterr().out)
        assert counts["guardrail"] >= 6 and "loader" in counts

    def test_a_category_that_may_be_empty_still_lists_cleanly(self) -> None:
        # segmenter/feature_extractor have no built-in component; other tests
        # may register doubles, so assert only that listing succeeds.
        assert main(["list", "feature_extractor"]) == 0

    def test_doctor_json_in_process(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["doctor", "--json", "--require", "nope"]) == 1
        payload = _json_from(capsys.readouterr().out)
        assert payload["unmet"] == ["nope"]

    def test_validate_json_in_process(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "name: x\nsteps:\n  - id: a\n    type: loader.clinical_text.nope\n",
            encoding="utf-8",
        )
        assert main(["validate", str(cfg), "--json"]) == 1
        assert _json_from(capsys.readouterr().out)["valid"] is False

    def test_validate_warnings_only_still_passes(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from openbtk.core.config import ValidationIssue
        from openbtk.pipelines import Pipeline

        monkeypatch.setattr(
            Pipeline,
            "validate",
            lambda self: [ValidationIssue(severity="warning", message="heads up")],
        )
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "name: x\nsteps:\n  - id: a\n    type: loader.clinical_text.plain_text\n",
            encoding="utf-8",
        )
        assert main(["validate", str(cfg)]) == 0
        assert "Valid, with warnings" in capsys.readouterr().out

    def test_run_json_in_process(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        notes = tmp_path / "n"
        notes.mkdir()
        (notes / "a.txt").write_text("hello", encoding="utf-8")
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            f"name: x\nprovenance: {{manifest_dir: {(tmp_path / 'r').as_posix()}}}\n"
            "steps:\n  - id: load\n    type: loader.clinical_text.plain_text\n"
            f'    params: {{path: "{notes.as_posix()}"}}\n',
            encoding="utf-8",
        )
        assert main(["run", str(cfg), "--json"]) == 0
        assert _json_from(capsys.readouterr().out)["status"] == "success"

    def test_replay_json_and_unreadable_manifest_in_process(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "m.json"
        bad.write_text("{nope", encoding="utf-8")
        assert main(["replay", str(bad)]) == 2
        assert "not a readable RunManifest" in capsys.readouterr().err

        good = _manifest(
            config={
                "name": "p",
                "provenance": {"manifest_dir": str(tmp_path / "runs")},
                "steps": [
                    {
                        "id": "load",
                        "type": "loader.clinical_text.plain_text",
                        "params": {"path": str(tmp_path)},
                    }
                ],
            },
            steps=[_step("load", 0, 0)],
        )
        (tmp_path / "good.json").write_text(good.model_dump_json(), encoding="utf-8")
        capsys.readouterr()
        code = main(["replay", str(tmp_path / "good.json"), "--json"])
        payload = _json_from(capsys.readouterr().out)
        assert code == (0 if payload["matches"] else 1)
