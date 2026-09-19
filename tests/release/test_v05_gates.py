"""The v0.5 release gates (docs/02_PRD.md section 7, "v0.5 -- Credibility"),
checked mechanically.

The PRD's rule is "a gate is not met until it is demonstrated in CI", so each
gate below is a test that fails if the evidence goes away. Gates whose
evidence is a *behaviour* (the Synthea-shaped EHR round trip, the RAG
pipeline, the guardrails) are proven by their own test suites; what this file
adds is the audit that those suites, registrations, workflows and docs still
exist and still meet the numeric thresholds -- so a gate cannot be quietly
lost in a refactor.

One gate cannot be proven from inside the repository, and this file says so
rather than pretending (a second, the live site, is checked by request and
recorded in the CHANGELOG -- Pages is a repository setting, not code):

* **Published de-id benchmark on i2b2/n2c2** -- the harness exists (tested),
  but the corpus is Data-Use-Agreement-restricted and has never been run. The
  gate test is a *strict* xfail: it flips to a failure the moment a real
  result is published, forcing whoever publishes it to remove the xfail and
  close the gate deliberately.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS = _ROOT / ".github" / "workflows"


def _registry_keys(category: str, *modules: str) -> list[str]:
    # Importing a component package registers its classes; that logs
    # registry events to stdout, which is noise here.
    with contextlib.redirect_stdout(io.StringIO()):
        for module in modules:
            importlib.import_module(module)
        from openbtk.core.registry import get_registry

        return get_registry(category).list_keys()


class TestDeidBenchmark:
    def test_the_harness_and_dataset_adapter_exist(self) -> None:
        keys = _registry_keys("dataset", "openbtk.data.clinical_text")
        assert "dataset.clinical_text.n2c2_deid" in keys
        assert importlib.util.find_spec("openbtk.eval.deid_benchmark") is not None

    def test_a_reproducible_synthetic_benchmark_is_published(self) -> None:
        page = (_ROOT / "mkdocs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "benchmark:rule:start" in page
        assert "python tests/accuracy/report.py" in page

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "OPEN GATE: no i2b2/n2c2 result is published -- the corpus is "
            "DUA-restricted and has not been run. Remove this xfail when a "
            "real result is added to mkdocs/benchmarks.md."
        ),
    )
    def test_an_i2b2_n2c2_result_is_published(self) -> None:
        page = (_ROOT / "mkdocs" / "benchmarks.md").read_text(encoding="utf-8")
        assert "**Not run.**" not in page


class TestEHR:
    def test_fhir_and_omop_loaders_are_registered(self) -> None:
        keys = _registry_keys("loader", "openbtk.data.ehr")
        assert "loader.ehr.fhir" in keys
        assert "loader.ehr.omop" in keys

    def test_the_synthea_shaped_round_trip_test_exists(self) -> None:
        source = (_ROOT / "tests" / "integration" / "test_ehr_pipeline.py").read_text(
            encoding="utf-8"
        )
        assert re.search(r"def test_\w+", source)
        assert "Synthea" in source or "synthea" in source

    def test_ci_runs_the_ehr_tests_with_the_extra_installed(self) -> None:
        ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        assert re.search(r"test-ehr:", ci)
        assert re.search(r"\.\[dev,ehr\]", ci)


class TestProviders:
    """Counted as distinct named options a user can construct: a registered
    provider class, or a preset (configuration over a registered class)."""

    def test_at_least_five_llm_options(self) -> None:
        classes = _registry_keys("llm", "openbtk.llms")
        with contextlib.redirect_stdout(io.StringIO()):
            from openbtk.llms.presets import list_llm_presets

            presets = list_llm_presets()
        assert len(classes) + len(presets) >= 5

    def test_at_least_five_embedding_options(self) -> None:
        classes = _registry_keys("embedding", "openbtk.embeddings")
        with contextlib.redirect_stdout(io.StringIO()):
            from openbtk.embeddings.presets import list_embedding_presets

            presets = list_embedding_presets()
        assert len(classes) + len(presets) >= 5

    def test_offsite_providers_declare_it(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            import openbtk.embeddings
            import openbtk.llms  # noqa: F401
            from openbtk.core.registry import get_registry

            classes = [
                get_registry(c).get(k)
                for c in ("llm", "embedding")
                for k in get_registry(c).list_keys()
            ]
        assert all(hasattr(cls, "sends_data_offsite") for cls in classes)


class TestRetrievalAndReranking:
    def test_three_vector_stores_and_a_concept_reranker(self) -> None:
        stores = _registry_keys("vectorstore", "openbtk.retrieval")
        rerankers = _registry_keys("reranker", "openbtk.retrieval")
        assert len(stores) >= 3
        assert "reranker.general.concept_overlap" in rerankers

    def test_a_rag_pipeline_exists(self) -> None:
        assert importlib.util.find_spec("openbtk.pipelines.rag") is not None


class TestGuardrailsAndTerminology:
    def test_the_guardrail_suite_is_registered(self) -> None:
        keys = _registry_keys("guardrail", "openbtk.guardrails")
        for required in (
            "guardrail.general.phi_leakage",
            "guardrail.general.terminology",
            "guardrail.general.groundedness",
        ):
            assert required in keys

    def test_the_terminology_service_has_all_three_backends(self) -> None:
        keys = _registry_keys("terminology", "openbtk.terminology")
        assert {
            "terminology.general.bundled_minimal",
            "terminology.general.local",
            "terminology.general.umls",
        } <= set(keys)


class TestLangChainAdapter:
    def test_the_adapter_and_its_extra_exist(self) -> None:
        assert importlib.util.find_spec("openbtk.integrations.langchain") is not None
        pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert re.search(r"^langchain = \[", pyproject, re.MULTILINE)

    def test_ci_proves_both_halves_of_adr_0001(self) -> None:
        ci = (_WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
        assert "test-langchain:" in ci
        assert "test-no-langchain:" in ci


class TestMimicScaleStreaming:
    """NFR-01: stream 10M notes in under 4 GB RSS. The benchmark itself is
    nightly-only (it takes tens of minutes); these checks make sure it still
    asks for the literal target and that a scheduled workflow really runs it."""

    def _memory_benchmark(self) -> str:
        return (_ROOT / "tests" / "benchmark" / "test_memory.py").read_text(
            encoding="utf-8"
        )

    def test_the_benchmark_asks_for_the_literal_nfr_01_target(self) -> None:
        source = self._memory_benchmark()
        count = re.search(r"^_DEFAULT_NOTE_COUNT = ([\d_]+)", source, re.MULTILINE)
        assert count is not None
        assert int(count.group(1).replace("_", "")) >= 10_000_000
        assert re.search(r"^_RSS_TARGET_BYTES = 4 \* 1024\*\*3", source, re.MULTILINE)

    def test_a_scheduled_workflow_runs_it(self) -> None:
        workflow = (_WORKFLOWS / "benchmark.yml").read_text(encoding="utf-8")
        assert "schedule:" in workflow
        assert "OPENBTK_RUN_BENCHMARKS" in workflow
        assert "tests/benchmark" in workflow


class TestDocsSite:
    def test_the_site_builds_and_deploys_from_a_workflow(self) -> None:
        workflow = (_WORKFLOWS / "docs.yml").read_text(encoding="utf-8")
        assert "mkdocs build" in workflow
        assert "mike deploy" in workflow

    def test_site_url_is_the_served_address_not_a_redirect(self) -> None:
        """The site is served from the organisation's custom domain;
        openbtk.github.io/openbtk-core only redirects there, and a canonical
        link to a redirect is a broken habit."""
        config = (_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        assert "site_url: https://openbtk.org/openbtk-core/" in config

    def test_the_nav_covers_the_pages_a_v0_5_reader_needs(self) -> None:
        nav = (_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        for page in ("index.md", "quickstart.md", "benchmarks.md", "langchain.md"):
            assert page in nav
        for page in ("index.md", "quickstart.md", "benchmarks.md", "langchain.md"):
            assert (_ROOT / "mkdocs" / page).is_file()

    def test_no_docs_page_still_claims_the_project_is_pre_m4(self) -> None:
        """Guards the staleness this milestone fixed: pages that said only
        M1-M3 existed and that PyPI held just a placeholder."""
        for path in [_ROOT / "README.md", *(_ROOT / "mkdocs").glob("*.md")]:
            text = path.read_text(encoding="utf-8")
            assert "0.0.1` on PyPI is still" not in text, path.name
            assert "not yet — `0.0.1`" not in text.lower(), path.name
            assert "Milestones **M1 (core framework)**" not in text, path.name
