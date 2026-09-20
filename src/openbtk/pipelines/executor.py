"""The streaming DAG executor (docs/03_ARCHITECTURE.md section 7): topologically
orders steps, streams records through them, collects provenance, and runs
guardrails at declared points. Deliberately small -- LangChain/LangGraph handle
anything more complex than a data pipeline (ADR-0001).

**Scope, disclosed rather than silently assumed:**

  * The steps form a DAG (FR-L-04), with two rules. A step with **several dependents**
    (fan-out) feeds each of them every record, through ``itertools.tee``; a step with
    **several predecessors** (fan-in) receives their streams *interleaved*, one record
    at a time in turn. Interleaving is a merge, not a join: the step sees each record
    on its own and a step that cannot handle a record's type fails, naming itself. It
    does not pair records across streams (see ``openbtk.pipelines.join`` for joining
    notes to structured data). Memory stays bounded because the run pulls from every
    leaf in turn, so the buffer between branches holds only the lag between them (a
    chunker that emits many records per input widens it by that many).
  * A diamond (``a -> b``, ``a -> c``, ``b, c -> d``) gives ``d`` every record of ``a``
    twice, once by each path; that is the topology asked for, not a bug.
  * A step's outputs are still discarded at the leaves: the run's product is its
    manifest, and the steps' own side effects.
  * Only ``loader``, ``preprocessor``, ``chunker`` and ``segmenter`` steps
    are executable. No real ``embedding``/``vectorstore`` component exists
    in this repository yet to validate an execution path against --
    building speculative dispatch for them now would repeat exactly the
    "5,227 lines that never executed" failure CLAUDE.md exists to prevent.
    A step of any other category raises ``ConfigError`` naming the
    category, not a silent no-op.
  * ``GuardrailOutcome`` aggregates per (guardrail, attachment point) --
    see ``openbtk.core.provenance``'s module docstring for why.
  * A step's ``StepProvenance.status`` is ``"failed"`` only for the step
    whose OWN transformation call raised -- every other wired step that
    processed everything it was handed without error keeps ``"success"``,
    even if the run overall failed elsewhere. This pinpoints where to
    look; it does not mean the whole run succeeded (check
    ``RunManifest.status`` for that).
  * This file runs a little over the doc's own "< 400 lines" target
    (``Step``/``Pipeline``, the builder API, already live in
    ``openbtk.pipelines.pipeline`` to keep this file to just the four
    things it actually claims to do) -- per-step failure attribution and
    PHI-safe secret redaction of the embedded config snapshot are real
    correctness requirements (ADR-0005, docs/06_SECURITY_COMPLIANCE.md
    section 3.7), not scope creep, and cost real lines. Disclosed rather
    than trimmed away to hit a round number.
"""

from __future__ import annotations

import hashlib
import itertools
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from openbtk.core.base import BaseChunker, BasePreprocessor, BaseSegmenter
from openbtk.core.errors import (
    ConfigError,
    GuardrailViolation,
    OpenBTKError,
    ProcessingError,
)
from openbtk.core.logging import get_logger
from openbtk.core.provenance import (
    DataDigest,
    GuardrailOutcome,
    RunManifest,
    StepProvenance,
)
from openbtk.core.registry import get_registry
from openbtk.core.schemas import GuardrailSeverity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from openbtk.core.base import Component
    from openbtk.core.config import GuardrailConfig, PipelineConfig, StepConfig

log = get_logger(__name__)

_HASHABLE_SIZE_CAP = 100 * 1024 * 1024
"""A single input file larger than this is not hashed for its DataDigest --
see DataDigest's own docstring. Reading 100MB to hash it is a bounded,
one-off cost; reading an arbitrarily large corpus would not be."""

_SECRET_KEY_RE = re.compile(r"(key|token|secret|password|credential)", re.IGNORECASE)
"""docs/06_SECURITY_COMPLIANCE.md section 3.7's exact pattern: an
interpolated ``${OPENAI_API_KEY}``-shaped value must never be written back
into a serialised manifest."""

_TOKEN_COUNT_KEY_RE = re.compile(
    r"(?:^|_)(?:max|min|overlap|num|count|total|prompt|completion)_tokens$",
    re.IGNORECASE,
)
"""Keys that contain "token" but are LENGTHS, not credentials -- ``max_tokens``
is a chunker's size limit and every LLM's output cap. Redacting them (as the
pattern above alone would) erased a real setting from the audit record and made
any pipeline with a chunker impossible to replay. The exemption is deliberately
narrow: ``access_token``, ``hf_token`` and the like are still redacted."""


def _is_secret_key(key: str) -> bool:
    return bool(_SECRET_KEY_RE.search(key)) and not _TOKEN_COUNT_KEY_RE.search(key)


def _redact_secrets(value: Any) -> Any:
    """Recursively replace any dict value whose key looks like a credential."""
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if _is_secret_key(k) else _redact_secrets(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


class _Counts:
    """Mutable per-step tally, updated as items are actually pulled through
    a lazily-composed generator chain -- real counts of what happened, not
    a static prediction, and never materialised beyond two integers."""

    __slots__ = ("input_count", "output_count")

    def __init__(self) -> None:
        self.input_count = 0
        self.output_count = 0


class _StepRecord:
    __slots__ = ("component", "counts", "digest_sha", "digest_uri", "step_id")

    def __init__(self, step_id: str, component: Component, counts: _Counts) -> None:
        self.step_id = step_id
        self.component = component
        self.counts = counts
        self.digest_uri: str | None = None
        self.digest_sha: str | None = None


class _GuardrailTally:
    __slots__ = ("blocked", "checked", "samples", "warned")

    def __init__(self) -> None:
        self.checked = 0
        self.blocked = 0
        self.warned = 0
        self.samples: list[str] = []

    def record(self, passed: bool, severity: GuardrailSeverity, message: str) -> None:
        self.checked += 1
        if passed:
            return
        if severity is GuardrailSeverity.BLOCK:
            self.blocked += 1
        elif severity is GuardrailSeverity.WARNING:
            self.warned += 1
        if len(self.samples) < 5:
            self.samples.append(message)


def _validate_shape(steps: list[StepConfig]) -> None:
    """The shape rules that hold whatever the topology: a step names each predecessor
    at most once, and a loader is a root. (Cycles and unknown ``after`` ids are
    ``PipelineConfig.validate_registry``'s.)"""
    for step in steps:
        if len(set(step.after)) != len(step.after):
            raise ConfigError(
                f"Step {step.id!r} lists the same predecessor more than once.",
                context={"step_id": step.id},
            )
        if step.type.split(".", 1)[0] == "loader" and step.after:
            raise ConfigError(
                f"Step {step.id!r}: a loader step must be a root step (no 'after').",
                context={"step_id": step.id},
            )


def _interleave(streams: list[Iterator[Any]]) -> Iterator[Any]:
    """Merge streams fairly: one item from each live stream in turn, until all end. An
    exception from any stream propagates as it is."""
    live = [iter(stream) for stream in streams]
    while live:
        for iterator in list(live):
            try:
                item = next(iterator)
            except StopIteration:
                live.remove(iterator)
                continue
            yield item


def _drain(leaves: list[Iterator[Any]]) -> None:
    """Pull every leaf to the end, one item from each in turn. Alternating, rather than
    finishing one leaf before the next, is what keeps a fan-out's buffer small: the
    branches advance together, so no branch waits far ahead of another."""
    live = list(leaves)
    while live:
        for iterator in list(live):
            try:
                next(iterator)
            except StopIteration:
                live.remove(iterator)


def _topological_order(steps: list[StepConfig]) -> list[str]:
    """DFS postorder topological sort. Callers must have already rejected
    cycles (``PipelineConfig.validate_registry()``) -- this does not detect
    them a second time."""
    graph = {s.id: s.after for s in steps}
    order: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visited:
            return
        visited.add(node)
        for dep in graph.get(node, ()):
            visit(dep)
        order.append(node)

    for step_id in graph:
        visit(step_id)
    return order


def _digest_source(source: Any) -> tuple[str, str | None] | None:
    """``(uri, sha256)`` for a loader step's source, when it is a single,
    existing, bounded-size local file. See ``DataDigest``'s own docstring
    for every case this returns ``None`` for instead of guessing."""
    if not isinstance(source, str):
        return None
    path = Path(source)
    if not path.is_file():
        return str(source), None
    try:
        if path.stat().st_size > _HASHABLE_SIZE_CAP:
            return str(source), None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return str(source), None
    return str(source), digest


def _run_source(
    step_id: str,
    produce: Callable[[], Iterator[Any]],
    counts: _Counts,
    failed: list[str | None],
) -> Iterator[Any]:
    """Wrap a step's own, self-contained iterator (no upstream to pull
    from) so any exception it raises is attributed to this step."""
    try:
        for item in produce():
            counts.output_count += 1
            yield item
    except OpenBTKError:
        failed[0] = step_id
        raise
    except Exception as e:
        failed[0] = step_id
        raise ProcessingError(
            f"Step {step_id!r} failed: {e}", context={"step_id": step_id}
        ) from e


def _run_per_record(
    step_id: str,
    upstream: Iterator[Any],
    transform: Callable[[Any], Iterator[Any]],
    counts: _Counts,
    failed: list[str | None],
) -> Iterator[Any]:
    """Pull one item at a time from ``upstream`` (untouched -- an upstream
    failure propagates as-is and is never re-attributed here) and feed each
    to ``transform``, which may yield zero, one, or many outputs. Only
    ``transform``'s own execution is attributed to ``step_id`` on failure."""
    it = iter(upstream)
    while True:
        try:
            item = next(it)
        except StopIteration:
            return
        counts.input_count += 1
        try:
            for output in transform(item):
                counts.output_count += 1
                yield output
        except OpenBTKError:
            failed[0] = step_id
            raise
        except Exception as e:
            failed[0] = step_id
            raise ProcessingError(
                f"Step {step_id!r} failed: {e}", context={"step_id": step_id}
            ) from e


class _Executor:
    def __init__(self, config: PipelineConfig) -> None:
        self._config = config

    def run(self) -> RunManifest:
        run_id = uuid.uuid4().hex
        started = datetime.now(UTC)
        step_records: list[_StepRecord] = []
        guardrail_tallies: dict[tuple[str, str], _GuardrailTally] = {}
        failed: list[str | None] = [None]
        status: Literal["success", "failed"] = "success"
        error_message: str | None = None

        try:
            issues = [
                i for i in self._config.validate_registry() if i.severity == "error"
            ]
            if issues:
                raise ConfigError(
                    "Pipeline config failed registry validation: "
                    + "; ".join(i.message for i in issues),
                    context={"pipeline": self._config.name},
                )
            _validate_shape(self._config.steps)
            order = _topological_order(self._config.steps)
            by_id = {s.id: s for s in self._config.steps}
            guardrails_by_point = self._index_guardrails()

            dependents: dict[str, int] = {}
            for step in self._config.steps:
                for predecessor_id in step.after:
                    dependents[predecessor_id] = dependents.get(predecessor_id, 0) + 1

            # Each step's output, as one iterator per dependent (a fan-out step gets a
            # tee of its stream, one branch for each step that reads it).
            outputs: dict[str, list[Iterator[Any]]] = {}
            for step_id in order:
                step = by_id[step_id]
                # Looked up by the step's OWN declared `after`, never "whatever
                # the previous loop iteration produced" -- topological order is
                # not the same thing as declaration order once a config has
                # more than one root (e.g. two independent loaders), and
                # conflating the two was a real bug caught by a test with two
                # root steps sharing this executor's own step-wiring loop.
                upstreams = [outputs[dep].pop() for dep in step.after]
                predecessor = (
                    None
                    if not upstreams
                    else upstreams[0]
                    if len(upstreams) == 1
                    else _interleave(upstreams)
                )
                stream, record = self._wire_step(step, predecessor, failed)
                step_records.append(record)
                point = f"after:{step_id}"
                if point in guardrails_by_point:
                    stream = self._apply_guardrails(
                        point, guardrails_by_point[point], stream, guardrail_tallies
                    )
                branches = dependents.get(step_id, 0)
                outputs[step_id] = (
                    list(itertools.tee(stream, branches)) if branches > 1 else [stream]
                )
            _drain(
                [
                    outputs[step_id][0]
                    for step_id in order
                    if not dependents.get(step_id)
                ]
            )
        except GuardrailViolation as e:
            status, error_message = "failed", str(e)
        except OpenBTKError as e:
            status, error_message = "failed", str(e)
        except Exception as e:  # pragma: no cover -- defensive backstop, see below
            # ADR-0005's "no manifest-off switch" must hold even for a bug this
            # executor didn't anticipate, not just the documented error types above.
            status, error_message = "failed", f"Unexpected error: {e}"

        ended = datetime.now(UTC)
        steps = [
            StepProvenance(
                step_id=r.step_id,
                component=r.component.provenance(),
                records_in=r.counts.input_count,
                records_out=r.counts.output_count,
                status="failed" if r.step_id == failed[0] else "success",
                error=error_message if r.step_id == failed[0] else None,
            )
            for r in step_records
        ]
        digests = [
            DataDigest(
                uri=r.digest_uri,
                sha256=r.digest_sha,
                record_count=r.counts.output_count,
            )
            for r in step_records
            if r.digest_uri is not None
        ]
        outcomes = [
            GuardrailOutcome(
                guardrail_key=key,
                at=at,
                checked_count=tally.checked,
                blocked_count=tally.blocked,
                warned_count=tally.warned,
                sample_messages=tally.samples,
            )
            for (key, at), tally in guardrail_tallies.items()
        ]
        return RunManifest(
            run_id=run_id,
            status=status,
            config=_redact_secrets(self._config.model_dump(mode="json")),
            started_at=started,
            ended_at=ended,
            steps=steps,
            input_digests=digests,
            guardrail_outcomes=outcomes,
            error=error_message,
        )

    def _index_guardrails(self) -> dict[str, list[GuardrailConfig]]:
        by_point: dict[str, list[GuardrailConfig]] = {}
        step_ids = {s.id for s in self._config.steps}
        for guardrail in self._config.guardrails:
            for point in guardrail.at:
                if (
                    not point.startswith("after:")
                    or point.removeprefix("after:") not in step_ids
                ):
                    raise ConfigError(
                        f"Guardrail {guardrail.type!r} attachment point {point!r} "
                        "must be 'after:<step_id>' naming a real step.",
                        context={"guardrail": guardrail.type, "at": point},
                    )
                by_point.setdefault(point, []).append(guardrail)
        return by_point

    def _wire_step(
        self,
        step: StepConfig,
        predecessor: Iterator[Any] | None,
        failed: list[str | None],
    ) -> tuple[Iterator[Any], _StepRecord]:
        category = step.type.split(".", 1)[0]
        registry = get_registry(category)
        counts = _Counts()

        if category == "loader":
            # docs/03_ARCHITECTURE.md section 7.3's own worked example flattens
            # a loader's ".load(source)" call argument ("path") and its
            # constructor kwargs ("note_type") into one params dict -- a real
            # loader's __init__ does not accept "path"/"source" itself
            # (PlainTextLoader(note_type=...), separately, .load(directory)),
            # so those two keys must be split out before construction, not
            # forwarded to it.
            ctor_kwargs = dict(step.params)
            source = ctor_kwargs.pop("source", None)
            if source is None:
                source = ctor_kwargs.pop("path", None)
            component = registry.create(
                step.type, policy=self._config.policy, **ctor_kwargs
            )
            record = _StepRecord(step.id, component, counts)
            if predecessor is not None:
                raise ConfigError(
                    f"Step {step.id!r}: a loader step must be a root step "
                    "(no 'after').",
                    context={"step_id": step.id},
                )
            digest = _digest_source(source)
            if digest is not None:
                record.digest_uri, record.digest_sha = digest
            stream = _run_source(
                step.id, lambda: component.load(source), counts, failed
            )
            return stream, record

        component = registry.create(
            step.type, policy=self._config.policy, **step.params
        )
        record = _StepRecord(step.id, component, counts)

        if isinstance(component, BasePreprocessor):
            if predecessor is None:
                raise ConfigError(
                    f"Step {step.id!r}: a preprocessor step needs a predecessor.",
                    context={"step_id": step.id},
                )
            stream = _run_per_record(
                step.id,
                predecessor,
                lambda r: iter((component.process(r),)),
                counts,
                failed,
            )
        elif isinstance(component, (BaseChunker, BaseSegmenter)):
            if predecessor is None:
                raise ConfigError(
                    f"Step {step.id!r}: a chunker/segmenter step needs a predecessor.",
                    context={"step_id": step.id},
                )
            method = (
                component.chunk
                if isinstance(component, BaseChunker)
                else component.segment
            )
            stream = _run_per_record(step.id, predecessor, method, counts, failed)
        else:
            raise ConfigError(
                f"Step {step.id!r}: type {step.type!r} (category {category!r}) "
                "is not yet executable by this pipeline -- only loader, "
                "preprocessor, chunker and segmenter steps are (see this "
                "module's docstring).",
                context={"step_id": step.id, "category": category},
            )
        return stream, record

    def _apply_guardrails(
        self,
        point: str,
        configs: list[GuardrailConfig],
        stream: Iterator[Any],
        guardrail_tallies: dict[tuple[str, str], _GuardrailTally],
    ) -> Iterator[Any]:
        registry = get_registry("guardrail")
        instances = [
            (cfg, registry.create(cfg.type, policy=self._config.policy))
            for cfg in configs
        ]
        for _cfg, guardrail in instances:
            guardrail_tallies.setdefault(
                (guardrail.registry_key, point), _GuardrailTally()
            )
        for item in stream:
            for cfg, guardrail in instances:
                result = guardrail.check(item)
                guardrail_tallies[(guardrail.registry_key, point)].record(
                    result.passed, result.severity, result.message
                )
                blocked = (
                    not result.passed and result.severity is GuardrailSeverity.BLOCK
                )
                if blocked and cfg.on_violation == "block":
                    raise GuardrailViolation(
                        f"Guardrail {guardrail.registry_key!r} blocked at "
                        f"{point}: {result.message}",
                        context={"guardrail": guardrail.registry_key, "at": point},
                    )
            yield item
