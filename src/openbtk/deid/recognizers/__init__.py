"""Pluggable PHI recognizers: rule-based, NER, and an optional LLM verifier.

Each returns spans with confidence; SpanMerger combines them.

``RuleRecognizer`` is imported here (not left to whoever happens to import
``openbtk.deid.recognizers.rule`` directly) so that ``recognizer.general.rule``
is registered as soon as this package is, with no separate import step --
safe because ``rule.py`` has zero heavy dependencies (stdlib ``re`` only).
This is deliberately NOT the pattern for a future NER/LLM-verifier
recognizer: those carry real optional dependencies and must stay
unimported until actually requested (CLAUDE.md rule 5), so they will need
their own lazy registration path rather than an eager import here.
"""

from __future__ import annotations

from openbtk.deid.recognizers.rule import RuleRecognizer

__all__ = ["RuleRecognizer"]
