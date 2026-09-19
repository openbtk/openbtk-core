"""End-to-end tests for ``python -m openbtk.eval.deid_benchmark``: the real
``DeidEngine`` (rule recognizer), the real n2c2 adapter and the real scorer,
over hand-built files in the i2b2 2014 shape. No mocking of any of them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from openbtk.eval.deid_benchmark import main

if TYPE_CHECKING:
    from pathlib import Path

# The rule recognizer finds the date and the phone number, not the name --
# a known M2 limit (NER is opt-in), so name recall is 0 by design here.
_DOC_A = "Seen on 03/14/2024. Call (555) 010-2345 for results. Dr Jane Roe agrees."
_DOC_B = "Refill on 04/01/2024."


def _tag(element: str, typ: str, text: str, doc: str) -> str:
    start = doc.index(text)
    return (
        f'<{element} id="P" start="{start}" end="{start + len(text)}" '
        f'text="{text}" TYPE="{typ}" comment=""/>'
    )


def _xml(text: str, tags: str) -> str:
    return f"<deIdi2b2><TEXT><![CDATA[{text}]]></TEXT><TAGS>{tags}</TAGS></deIdi2b2>"


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "n2c2"
    d.mkdir()
    a_tags = (
        _tag("DATE", "DATE", "03/14/2024", _DOC_A)
        + _tag("CONTACT", "PHONE", "(555) 010-2345", _DOC_A)
        + _tag("NAME", "DOCTOR", "Jane Roe", _DOC_A)
        + _tag("AGE", "AGE", "Dr", _DOC_A)  # no Safe Harbor category -> unscored
    )
    (d / "a.xml").write_text(_xml(_DOC_A, a_tags), encoding="utf-8")
    (d / "b.xml").write_text(_xml(_DOC_B, ""), encoding="utf-8")  # date is a FP
    return d


class TestSuccessfulRun:
    def test_scores_match_hand_worked_counts(
        self, dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_json = tmp_path / "r.json"
        out_md = tmp_path / "r.md"
        code = main(
            [
                "--dataset",
                "n2c2",
                "--path",
                str(dataset_dir),
                "--json",
                str(out_json),
                "--markdown",
                str(out_md),
            ]
        )
        assert code == 0
        report = json.loads(out_json.read_text(encoding="utf-8"))
        # doc a: date TP, phone TP, name FN.  doc b: date FP (no gold date).
        assert report["n_documents"] == 2
        assert report["unscored_gold_spans"] == 1
        assert report["per_category"]["date"]["true_positives"] == 1
        assert report["per_category"]["date"]["false_positives"] == 1
        assert report["per_category"]["phone_number"]["true_positives"] == 1
        assert report["per_category"]["name"]["false_negatives"] == 1
        assert report["overall"]["true_positives"] == 2
        assert report["overall"]["false_positives"] == 1
        assert report["overall"]["false_negatives"] == 1
        assert report["binary"]["recall"] == pytest.approx(2 / 3)
        assert report["recognizers"] == ["rule"]
        assert "### " in out_md.read_text(encoding="utf-8")
        assert "overall (category-aware)" in capsys.readouterr().out

    def test_without_output_flags_the_report_goes_to_stdout_only(
        self, dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--dataset", "n2c2", "--path", str(dataset_dir)]) == 0
        assert "| date |" in capsys.readouterr().out
        assert not list(tmp_path.glob("*.json"))

    def test_the_report_files_never_contain_document_text(
        self, dataset_dir: Path, tmp_path: Path
    ) -> None:
        out_json = tmp_path / "r.json"
        out_md = tmp_path / "r.md"
        main(
            [
                "--dataset", "n2c2", "--path", str(dataset_dir),
                "--json", str(out_json), "--markdown", str(out_md),
            ]
        )  # fmt: skip
        combined = out_json.read_text(encoding="utf-8") + out_md.read_text(
            encoding="utf-8"
        )
        for fragment in ("Jane Roe", "555", "03/14/2024", "Refill"):
            assert fragment not in combined

    def test_limit_scores_only_the_first_documents(
        self, dataset_dir: Path, tmp_path: Path
    ) -> None:
        out_json = tmp_path / "r.json"
        main(
            [
                "--dataset", "n2c2", "--path", str(dataset_dir),
                "--limit", "1", "--json", str(out_json),
            ]
        )  # fmt: skip
        assert json.loads(out_json.read_text(encoding="utf-8"))["n_documents"] == 1


class TestRefusals:
    def test_missing_dataset_directory_exits_2_naming_the_registration_page(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", "n2c2", "--path", str(tmp_path / "absent")])
        assert code == 2
        assert "Register at" in capsys.readouterr().err

    def test_empty_directory_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", "n2c2", "--path", str(tmp_path)])
        assert code == 2
        assert "no *.xml" in capsys.readouterr().err

    def test_a_malformed_document_exits_2(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        (tmp_path / "bad.xml").write_text("<oops", encoding="utf-8")
        assert main(["--dataset", "n2c2", "--path", str(tmp_path)]) == 2
        assert "not well-formed" in capsys.readouterr().err

    def test_a_bad_limit_exits_2(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["--dataset", "n2c2", "--path", str(dataset_dir), "--limit", "0"])
        assert code == 2
        assert "--limit" in capsys.readouterr().err

    def test_an_unknown_recognizer_exits_2(
        self, dataset_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(
            ["--dataset", "n2c2", "--path", str(dataset_dir), "--recognizers", "nope"]
        )
        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestAsAModule:
    def test_python_m_invocation_runs_and_exits_2_without_data(
        self, tmp_path: Path
    ) -> None:
        """The advertised ``python -m`` entry point actually exists and runs
        -- checked by executing it, not by reading the ``__main__`` guard."""
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "openbtk.eval.deid_benchmark",
                "--dataset",
                "n2c2",
                "--path",
                str(tmp_path / "absent"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert "Register at" in proc.stderr
