"""The tutorial notebooks are executed, not just stored.

Two layers:

* **Structure** (always runs, needs only the standard library): eight notebooks
  exist; each was executed before it was committed (its outputs are stored and
  none is an error); none leaks a machine-specific path; each says its data is
  fictitious; relative links between notebooks resolve.
* **Execution** (needs ``pip install "openbtk[notebooks]"`` plus whatever extra a
  notebook uses): every notebook is run top to bottom in a fresh kernel, in an
  empty working directory, and must finish without an error. This is what stops
  a tutorial from quietly rotting when the API moves.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOKS = sorted((_ROOT / "notebooks").glob("*.ipynb"))

# Extras a notebook needs beyond the notebook tooling itself.
_REQUIRES: dict[str, tuple[str, ...]] = {
    "03_ehr_to_text.ipynb": ("fhir.resources", "pyarrow"),
    "05_rag_with_provenance.ipynb": ("faiss",),
    "08_langchain_and_langgraph.ipynb": ("langchain_core", "langgraph"),
}


def _load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _source(cell: dict[str, Any]) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else str(src)


def test_there_are_eight_numbered_tutorials() -> None:
    names = [p.name for p in _NOTEBOOKS]
    assert len(names) == 8, names
    assert [n[:2] for n in names] == [f"{i:02d}" for i in range(1, 9)]


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
class TestStructure:
    def test_it_opens_with_a_title_and_says_its_data_is_fictitious(
        self, path: Path
    ) -> None:
        cells = _load(path)["cells"]
        assert cells[0]["cell_type"] == "markdown"
        assert _source(cells[0]).startswith("# ")
        assert "fictitious" in _source(cells[0])

    def test_it_was_executed_and_no_stored_output_is_an_error(self, path: Path) -> None:
        code = [c for c in _load(path)["cells"] if c["cell_type"] == "code"]
        assert len(code) >= 4
        assert all(c["execution_count"] is not None for c in code), "not executed"
        for c in code:
            assert not [o for o in c["outputs"] if o["output_type"] == "error"]

    def test_it_quiets_library_logging_before_importing_anything(
        self, path: Path
    ) -> None:
        first_code = next(c for c in _load(path)["cells"] if c["cell_type"] == "code")
        assert "OPENBTK_LOG_LEVEL" in _source(first_code)

    def test_stored_outputs_do_not_leak_a_machine_path(self, path: Path) -> None:
        text = json.dumps([c.get("outputs", []) for c in _load(path)["cells"]])
        assert not re.search(r"[A-Za-z]:\\\\Users|/home/|/Users/", text)
        assert "AppData" not in text

    def test_relative_links_to_other_notebooks_resolve(self, path: Path) -> None:
        for cell in _load(path)["cells"]:
            if cell["cell_type"] != "markdown":
                continue
            for target in re.findall(r"\]\(([\w./-]+\.ipynb)\)", _source(cell)):
                assert (path.parent / target).is_file(), f"{path.name} -> {target}"

    def test_it_leaves_no_files_next_to_itself_by_working_in_a_scratch_folder(
        self, path: Path
    ) -> None:
        """A tutorial that writes files must chdir into a temp folder first, or
        running it in Jupyter litters the folder it lives in."""
        code = "\n".join(
            _source(c) for c in _load(path)["cells"] if c["cell_type"] == "code"
        )
        if re.search(r"write_text|mkdir\(", code):
            assert "mkdtemp" in code or "TemporaryDirectory" in code, path.name


@pytest.fixture
def _selector_event_loop() -> Iterator[None]:
    """pyzmq (under Jupyter's kernel client) needs a selector event loop; on
    Windows asyncio defaults to the Proactor loop, which makes pyzmq warn -- and
    this project's pytest config turns warnings into errors."""
    if sys.platform != "win32":
        yield
        return
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        yield
    finally:
        asyncio.set_event_loop_policy(previous)


@pytest.mark.usefixtures("_selector_event_loop")
@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_top_to_bottom(
    path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    for module in _REQUIRES.get(path.name, ()):
        pytest.importorskip(module)

    notebook = nbformat.read(path, as_version=4)
    # A fresh kernel inherits this environment; make sure it does not inherit a
    # log-level override that would hide what the notebook itself sets.
    monkeypatch.delenv("OPENBTK_LOG_LEVEL", raising=False)
    client = nbclient.NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()  # raises CellExecutionError, naming the failing cell, on any error
    assert list(tmp_path.iterdir()) == [], "the notebook wrote into its start folder"
