"""The ``openbtk`` command line, driven the way a user drives it: real config
files, real notes, the real pipeline executor -- and a subprocess wherever what
matters is what reaches stdout versus stderr."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest

from openbtk.cli import main
from openbtk.cli.commands import _safe_filename
from openbtk.core.errors import ProcessingError

if TYPE_CHECKING:
    from pathlib import Path

_SSN = "123-45-6789"  # phi-fixture-ok: fictitious, obviously fake
_NOTE = f"Chief Complaint:\nPatient SSN {_SSN} reports chest pain.\nPlan:\nAdmit.\n"


def _run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "openbtk", *argv],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def _notes(tmp_path: Path) -> Path:
    d = tmp_path / "notes"
    d.mkdir(exist_ok=True)
    (d / "n1.txt").write_text(_NOTE, encoding="utf-8")
    return d


def _config(tmp_path: Path, notes: Path, *, extra_steps: str = "") -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        f"""
name: cli-test
provenance:
  manifest_dir: {(tmp_path / "runs").as_posix()}
steps:
  - id: load
    type: loader.clinical_text.plain_text
    params: {{path: "{notes.as_posix()}"}}
  - id: deid
    type: preprocessor.general.deidentify
    params: {{mode: redact}}
    after: [load]
{extra_steps}
""",
        encoding="utf-8",
    )
    return path


def _jsonl_config(tmp_path: Path, notes: Path) -> Path:
    path = tmp_path / "jsonl.yaml"
    path.write_text(
        f"""
name: replayable
provenance:
  manifest_dir: {(tmp_path / "runs").as_posix()}
steps:
  - id: load
    type: loader.clinical_text.jsonl
    params: {{path: "{notes.as_posix()}"}}
  - id: deid
    type: preprocessor.general.deidentify
    after: [load]
""",
        encoding="utf-8",
    )
    return path


class TestEntryPoint:
    def test_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--version"]) == 0
        assert capsys.readouterr().out.startswith("openbtk ")

    def test_no_command_is_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 2
        assert "usage" in capsys.readouterr().err.lower()

    def test_an_unknown_command_is_a_usage_error(self) -> None:
        assert main(["frobnicate"]) == 2

    def test_help_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--help"]) == 0
        assert "doctor" in capsys.readouterr().out

    def test_python_dash_m_works_and_stdout_is_only_the_answer(self) -> None:
        proc = _run("--version")
        assert proc.returncode == 0
        assert proc.stdout.startswith("openbtk ") and proc.stdout.count("\n") == 1

    def test_usage_errors_map_to_2_and_runtime_errors_to_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from openbtk.cli import commands
        from openbtk.core.errors import ConfigError

        def bad_config(_: Any) -> tuple[int, str]:
            raise ConfigError("bad thing")

        def failed_run(_: Any) -> tuple[int, str]:
            raise ProcessingError("it broke")

        monkeypatch.setattr(commands, "cmd_doctor", bad_config)
        assert main(["doctor"]) == 2
        monkeypatch.setattr(commands, "cmd_doctor", failed_run)
        assert main(["doctor"]) == 1
        assert "error: it broke" in capsys.readouterr().err


class TestList:
    def test_lists_every_category_with_counts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "loader" in out and "guardrail" in out

    def test_json_stdout_is_pure_json_with_logs_on_stderr(self) -> None:
        proc = _run("list", "loader", "--json")
        assert proc.returncode == 0
        rows = json.loads(proc.stdout)  # would raise on any log line
        keys = {r["key"] for r in rows}
        assert {"loader.clinical_text.plain_text", "loader.ehr.fhir"} <= keys
        assert "registry.register" in proc.stderr  # the noise went to stderr

    def test_offsite_providers_are_flagged(self) -> None:
        proc = _run("list", "llm")
        assert proc.returncode == 0
        assert "llm.general.openai" in proc.stdout
        assert "[offsite]" in proc.stdout
        line = next(x for x in proc.stdout.splitlines() if "huggingface_local" in x)
        assert "[offsite]" not in line

    def test_an_unknown_category_names_the_valid_ones(self) -> None:
        proc = _run("list", "nope")
        assert proc.returncode == 2
        assert "Available:" in proc.stderr and "loader" in proc.stderr


class TestValidate:
    def test_a_good_config_is_ok_and_nothing_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        notes = _notes(tmp_path)
        assert main(["validate", str(_config(tmp_path, notes))]) == 0
        assert "OK:" in capsys.readouterr().out
        assert not (tmp_path / "runs").exists()  # nothing was executed

    def test_an_unknown_component_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            "name: x\nsteps:\n  - id: a\n    type: loader.clinical_text.nope\n",
            encoding="utf-8",
        )
        assert main(["validate", str(cfg)]) == 1
        assert "not found in registry" in capsys.readouterr().out

    def test_an_unknown_parameter_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _config(
            tmp_path,
            _notes(tmp_path),
            extra_steps=(
                "  - id: chunk\n    type: chunker.clinical_text.section_aware\n"
                "    params: {max_tokns: 5}\n    after: [deid]\n"
            ),
        )
        assert main(["validate", str(cfg)]) == 1
        assert "'max_tokns'" in capsys.readouterr().out

    def test_an_offsite_provider_under_the_default_policy_is_an_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "offsite.yaml"
        cfg.write_text(
            "name: x\nsteps:\n  - id: llm\n    type: llm.general.openai\n",
            encoding="utf-8",
        )
        assert main(["validate", str(cfg)]) == 1
        assert "sends data offsite" in capsys.readouterr().out

    def test_a_non_linear_pipeline_is_caught_before_running(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _config(
            tmp_path,
            _notes(tmp_path),
            extra_steps=(
                "  - id: chunk\n    type: chunker.clinical_text.section_aware\n"
                "    after: [load]\n"  # load now has two dependents
            ),
        )
        assert main(["validate", str(cfg)]) == 1
        assert "more than one dependent" in capsys.readouterr().out

    def test_json_output(self, tmp_path: Path) -> None:
        proc = _run("validate", str(_config(tmp_path, _notes(tmp_path))), "--json")
        payload = json.loads(proc.stdout)
        assert proc.returncode == 0 and payload == {"valid": True, "issues": []}

    def test_json_output_lists_issues(self, tmp_path: Path) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            "name: x\nsteps:\n  - id: a\n    type: loader.clinical_text.nope\n",
            encoding="utf-8",
        )
        proc = _run("validate", str(cfg), "--json")
        payload = json.loads(proc.stdout)
        assert proc.returncode == 1 and payload["valid"] is False
        assert payload["issues"][0]["severity"] == "error"

    def test_a_missing_config_file_is_a_usage_error(self, tmp_path: Path) -> None:
        proc = _run("validate", str(tmp_path / "absent.yaml"))
        assert proc.returncode == 2 and "error:" in proc.stderr


class TestRun:
    def test_runs_and_writes_the_manifest_to_the_configured_directory(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _config(tmp_path, _notes(tmp_path))
        assert main(["run", str(cfg)]) == 0
        out = capsys.readouterr().out
        assert ": success" in out and "load: 0 in -> 1 out" in out
        (manifest_file,) = (tmp_path / "runs").glob("*.json")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest["status"] == "success"
        assert [s["step_id"] for s in manifest["steps"]] == ["load", "deid"]
        # The manifest carries counts and ids, never note content.
        assert _SSN not in manifest_file.read_text(encoding="utf-8")

    def test_manifest_flag_chooses_the_path(self, tmp_path: Path) -> None:
        cfg = _config(tmp_path, _notes(tmp_path))
        target = tmp_path / "out" / "m.json"
        assert main(["run", str(cfg), "--manifest", str(target)]) == 0
        assert target.is_file()

    def test_a_pipeline_that_fails_exits_1_and_still_writes_a_manifest(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = _config(tmp_path, tmp_path / "does-not-exist")
        assert main(["run", str(cfg)]) == 1
        assert "failed" in capsys.readouterr().out
        (manifest_file,) = (tmp_path / "runs").glob("*.json")
        assert (
            json.loads(manifest_file.read_text(encoding="utf-8"))["status"] == "failed"
        )

    def test_an_invalid_config_is_refused_before_anything_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = tmp_path / "bad.yaml"
        cfg.write_text(
            f"name: x\nprovenance: {{manifest_dir: {(tmp_path / 'runs').as_posix()}}}\n"
            "steps:\n  - id: a\n    type: loader.clinical_text.nope\n",
            encoding="utf-8",
        )
        assert main(["run", str(cfg)]) == 2
        assert "refusing to run" in capsys.readouterr().out
        assert not (tmp_path / "runs").exists()

    def test_json_flag_prints_the_manifest(self, tmp_path: Path) -> None:
        proc = _run("run", str(_config(tmp_path, _notes(tmp_path))), "--json")
        assert proc.returncode == 0
        assert json.loads(proc.stdout)["status"] == "success"


class TestReplay:
    def _record(self, tmp_path: Path) -> tuple[Path, Path]:
        notes = tmp_path / "notes.jsonl"
        notes.write_text(
            json.dumps({"record_id": "a", "text": _NOTE}) + "\n", encoding="utf-8"
        )
        cfg = _jsonl_config(tmp_path, notes)
        assert main(["run", str(cfg), "--manifest", str(tmp_path / "m1.json")]) == 0
        return notes, tmp_path / "m1.json"

    def test_an_unchanged_replay_matches_the_recorded_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, manifest = self._record(tmp_path)
        capsys.readouterr()
        code = main(
            ["replay", str(manifest), "--new-manifest", str(tmp_path / "m2.json")]
        )
        assert code == 0
        assert "replay matches the recorded run" in capsys.readouterr().out
        assert (tmp_path / "m2.json").is_file()

    def test_changed_input_content_is_reported_as_divergence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        notes, manifest = self._record(tmp_path)
        notes.write_text(
            json.dumps({"record_id": "a", "text": "Different note."}) + "\n",
            encoding="utf-8",
        )
        capsys.readouterr()
        code = main(
            ["replay", str(manifest), "--new-manifest", str(tmp_path / "m2.json")]
        )
        assert code == 1
        out = capsys.readouterr().out
        assert "DIVERGED" in out and "content changed" in out

    def test_a_changed_record_count_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        notes, manifest = self._record(tmp_path)
        with notes.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"record_id": "b", "text": "More."}) + "\n")
        capsys.readouterr()
        assert (
            main(["replay", str(manifest), "--new-manifest", str(tmp_path / "m2.json")])
            == 1
        )
        assert "content changed" in capsys.readouterr().out

    def test_json_output(self, tmp_path: Path) -> None:
        _, manifest = self._record(tmp_path)
        proc = _run(
            "replay",
            str(manifest),
            "--new-manifest",
            str(tmp_path / "m2.json"),
            "--json",
        )
        payload = json.loads(proc.stdout)
        assert proc.returncode == 0
        assert payload["matches"] is True and payload["differences"] == []

    def test_a_manifest_with_redacted_secrets_cannot_be_replayed_on_its_own(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, manifest = self._record(tmp_path)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["config"]["steps"][0]["params"]["api_key"] = "[REDACTED]"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        capsys.readouterr()
        assert main(["replay", str(manifest)]) == 2
        out = capsys.readouterr().out
        assert "redacted" in out and "steps[0].params.api_key" in out

    def test_an_unreadable_manifest_is_a_usage_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "m.json"
        bad.write_text("{not a manifest", encoding="utf-8")
        proc = _run("replay", str(bad))
        assert proc.returncode == 2 and "not a readable RunManifest" in proc.stderr

    def test_a_manifest_whose_config_no_longer_validates_is_refused(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _, manifest = self._record(tmp_path)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["config"]["steps"][0]["type"] = "loader.clinical_text.gone"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        capsys.readouterr()
        assert main(["replay", str(manifest)]) == 2
        assert "not found in registry" in capsys.readouterr().out


class TestDeid:
    def test_a_directory_of_notes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        assert main(["deid", str(_notes(tmp_path)), "--out", str(out)]) == 0
        assert "de-identified 1 document(s)" in capsys.readouterr().out
        text = (out / "n1.txt").read_text(encoding="utf-8")
        assert _SSN not in text and "[REDACTED]" in text
        summary = json.loads((out / "deid_summary.json").read_text(encoding="utf-8"))
        assert summary["documents"] == 1 and summary["entities_by_category"]["ssn"] == 1
        assert _SSN not in json.dumps(summary)

    def test_a_jsonl_file(self, tmp_path: Path) -> None:
        src = tmp_path / "notes.jsonl"
        src.write_text(
            "\n".join(
                json.dumps({"record_id": f"r{i}", "text": _NOTE}) for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"
        assert main(["deid", str(src), "--out", str(out), "--mode", "tag"]) == 0
        rows = [
            json.loads(line)
            for line in (out / "notes.deid.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [r["record_id"] for r in rows] == ["r0", "r1", "r2"]
        assert all(_SSN not in r["text"] for r in rows)
        assert all(r["deid_status"] == "deidentified" for r in rows)
        summary = json.loads((out / "deid_summary.json").read_text(encoding="utf-8"))
        assert summary["documents"] == 3 and len(summary["input_sha256"]) == 64

    def test_the_names_warning_is_on_stderr(self, tmp_path: Path) -> None:
        proc = _run("deid", str(_notes(tmp_path)), "--out", str(tmp_path / "o"))
        assert proc.returncode == 0
        assert "names and street addresses are NOT detected" in proc.stderr
        assert "NOT detected" not in proc.stdout

    def test_a_non_empty_output_directory_is_refused_without_force(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "keep.txt").write_text("existing", encoding="utf-8")
        assert main(["deid", str(_notes(tmp_path)), "--out", str(out)]) == 2
        assert "not empty" in capsys.readouterr().out
        assert (out / "keep.txt").read_text(encoding="utf-8") == "existing"
        assert main(["deid", str(_notes(tmp_path)), "--out", str(out), "--force"]) == 0

    def test_a_bad_input_is_a_usage_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        bad = tmp_path / "x.csv"
        bad.write_text("a,b", encoding="utf-8")
        assert main(["deid", str(bad), "--out", str(tmp_path / "o")]) == 2
        assert "not a directory of .txt notes" in capsys.readouterr().out

    def test_record_ids_become_safe_unique_filenames(self) -> None:
        taken: set[str] = set()
        names = [
            _safe_filename(i, taken)
            for i in ["../escape", "a/b", "a/b", "..", "", "ok.1"]
        ]
        assert names == ["_escape", "a_b", "a_b-2", "record", "record-2", "ok.1"]
        assert all(
            "/" not in n and "\\" not in n and not n.startswith(".") for n in names
        )

    def test_the_outputs_are_the_only_place_text_appears(self, tmp_path: Path) -> None:
        proc = _run("deid", str(_notes(tmp_path)), "--out", str(tmp_path / "o"))
        assert _SSN not in proc.stdout + proc.stderr

    def test_an_unknown_mode_is_rejected_by_the_parser(self, tmp_path: Path) -> None:
        assert (
            main(
                [
                    "deid",
                    str(_notes(tmp_path)),
                    "--out",
                    str(tmp_path / "o"),
                    "--mode",
                    "x",
                ]
            )
            == 2
        )


class TestDoctor:
    def test_text_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["doctor"]) == 0
        out = capsys.readouterr().out
        assert "Core dependencies" in out and "pydantic" in out
        assert "Optional extras" in out and "Credentials" in out

    def test_json_report_and_credentials_are_never_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "OPENAI_API_KEY", "sk-do-not-print-me"
        )  # pragma: allowlist secret
        env = dict(os.environ)
        proc = subprocess.run(
            [sys.executable, "-m", "openbtk", "doctor", "--json"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        report = json.loads(proc.stdout)
        assert report["credentials"]["OPENAI_API_KEY"] is True
        assert "sk-do-not-print-me" not in proc.stdout + proc.stderr

    def test_require_fails_for_an_unknown_or_missing_extra(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["doctor", "--require", "no-such-extra"]) == 1
        assert "required but not ready: no-such-extra" in capsys.readouterr().out

    def test_require_json_lists_unmet(self) -> None:
        proc = _run("doctor", "--json", "--require", "no-such-extra")
        payload = json.loads(proc.stdout)
        assert proc.returncode == 1 and payload["unmet"] == ["no-such-extra"]
        assert payload["required"] == ["no-such-extra"]

    def test_require_passes_when_the_extra_is_present(self) -> None:
        from openbtk.cli import doctor

        report = doctor.collect()
        ready = [n for n, e in report["extras"].items() if e["installed"]]
        if not ready:
            pytest.skip("no optional extra is installed in this environment")
        assert main(["doctor", "--require", ready[0]]) == 0
