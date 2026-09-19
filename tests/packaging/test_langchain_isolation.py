"""ADR-0001, enforced mechanically: nothing outside
``openbtk.integrations.langchain`` may need LangChain or LangGraph.

Each test runs in a fresh interpreter with an import blocker installed that
makes ``langchain*`` / ``langgraph*`` unimportable -- so the result is the same
whether or not those packages happen to be installed where the suite runs, and
the ``test-no-langchain`` CI job (which really does not install them) proves
the same thing from the other side.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

_BLOCKER = textwrap.dedent(
    """
    import importlib.abc, sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"langchain", "langchain_core", "langgraph"}:
                raise ImportError(f"blocked for the ADR-0001 test: {name}")
            return None

    sys.meta_path.insert(0, _Block())
    """
)


def _run(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_module_outside_the_adapter_imports_without_langchain() -> None:
    proc = _run(
        """
        import importlib, pkgutil, sys
        import openbtk

        ADAPTER = "openbtk.integrations.langchain"
        LC = {"langchain", "langchain_core", "langgraph"}

        def names(package):
            # Not pkgutil.walk_packages: it imports every package to recurse,
            # which would import the adapter itself.
            for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                if info.name.startswith(ADAPTER):
                    continue
                yield info.name
                if info.ispkg:
                    yield from names(importlib.import_module(info.name))

        failed = {}
        imported = 0
        for name in names(openbtk):
            try:
                importlib.import_module(name)
                imported += 1
            except Exception as e:  # report every failure, not just the first
                failed[name] = f"{type(e).__name__}: {e}"

        leaked = sorted(m for m in sys.modules if m.split(".")[0] in LC)
        assert imported > 30, f"walked suspiciously few modules: {imported}"
        assert not failed, failed
        assert not leaked, leaked
        print("ok", imported)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().splitlines()[-1].startswith("ok")


def test_the_adapter_fails_with_an_actionable_message_when_langchain_is_absent() -> (
    None
):
    proc = _run(
        """
        from openbtk.core.errors import MissingDependencyError
        try:
            import openbtk.integrations.langchain
        except MissingDependencyError as e:
            assert "openbtk[langchain]" in str(e), str(e)
            print("ok")
        else:
            raise SystemExit("adapter imported although langchain_core is blocked")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "ok"
