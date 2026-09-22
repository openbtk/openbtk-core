"""The ``openbtk`` subcommands.

Each handler takes the parsed arguments and returns ``(exit_code, stdout_text)``
instead of printing: ``main`` owns the streams, so component log lines (which
the library writes to stdout) can be steered to stderr while the command's own
output -- the only thing a script should parse -- is written to stdout alone.

Exit codes: ``0`` success; ``1`` the thing ran and failed, or a check found a
problem (a pipeline failed, validation found errors, a replay diverged);
``2`` the invocation itself was wrong (bad path, bad config, unknown name).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openbtk.cli import doctor
from openbtk.cli._components import load_components
from openbtk.core.config import PipelineConfig
from openbtk.core.errors import ConfigError
from openbtk.core.provenance import RunManifest
from openbtk.core.registry import get_registry, list_categories
from openbtk.pipelines import Pipeline

if TYPE_CHECKING:
    import argparse

    from openbtk.core.config import ValidationIssue

Result = tuple[int, str]

_REDACTED = "[REDACTED]"


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------- list


def cmd_list(args: argparse.Namespace) -> Result:
    """``openbtk list [category]`` -- registry introspection (FR-N-03)."""
    load_components()
    if args.category is None:
        counts = {c: len(get_registry(c).list_keys()) for c in list_categories()}
        if args.json:
            return 0, _json(counts)
        width = max(map(len, counts))
        lines = [f"{c.ljust(width)}  {n}" for c, n in counts.items()]
        return 0, "\n".join(lines) + "\n\nopenbtk list <category> shows the keys.\n"

    registry = get_registry(args.category)  # RegistryError names the valid ones
    rows: list[dict[str, Any]] = []
    for key in registry.list_keys():
        info = registry.describe(key)
        cls = registry.get(key)
        summary = (info.docstring or "").strip().splitlines()[:1]
        rows.append(
            {
                "key": key,
                "class": info.class_name,
                "module": info.module,
                "sends_data_offsite": getattr(cls, "sends_data_offsite", None),
                "summary": summary[0] if summary else "",
            }
        )
    if args.json:
        return 0, _json(rows)
    width = max((len(r["key"]) for r in rows), default=0)
    lines = []
    for r in rows:
        flag = "  [offsite]" if r["sends_data_offsite"] else ""
        lines.append(f"{r['key'].ljust(width)}  {r['class']}{flag}")
    return 0, "\n".join(lines) + "\n"


# ----------------------------------------------------------------- validate


def _format_issue(issue: ValidationIssue) -> str:
    return f"{issue.severity}: {issue.message}"


def cmd_validate(args: argparse.Namespace) -> Result:
    """``openbtk validate CONFIG`` -- check a config without running it (FR-N-02).

    Nothing is instantiated and no data is read.
    """
    load_components()
    pipeline = Pipeline.from_yaml(args.config)
    issues = pipeline.validate()
    errors = [i for i in issues if i.severity == "error"]
    if args.json:
        payload = {
            "valid": not errors,
            "issues": [i.model_dump(mode="json") for i in issues],
        }
        return (1 if errors else 0), _json(payload)
    if not issues:
        steps = len(pipeline.to_config().steps)
        return 0, f"OK: {args.config} ({steps} step(s)); nothing was run.\n"
    body = "\n".join(_format_issue(i) for i in issues) + "\n"
    if errors:
        return 1, body + f"{len(errors)} error(s); the pipeline will not run.\n"
    return 0, body + "Valid, with warnings.\n"


# ---------------------------------------------------------------------- run


def _summary(manifest: RunManifest, manifest_path: Path | None) -> str:
    lines = [f"run {manifest.run_id}: {manifest.status}"]
    for step in manifest.steps:
        lines.append(
            f"  {step.step_id}: {step.records_in} in -> {step.records_out} out"
        )
    if manifest.error:
        lines.append(f"error: {manifest.error}")
    if manifest_path is not None:
        lines.append(f"manifest: {manifest_path}")
    return "\n".join(lines) + "\n"


def _write_manifest(manifest: RunManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _run_config(
    config: PipelineConfig,
    manifest_arg: str | None,
    *,
    checkpoint_path: str | None = None,
    checkpoint_interval: int = 1000,
) -> tuple[RunManifest, Path]:
    pipeline = Pipeline.from_config(config)
    manifest = pipeline.run(
        checkpoint_path=checkpoint_path, checkpoint_interval=checkpoint_interval
    )
    path = (
        Path(manifest_arg)
        if manifest_arg
        else Path(config.provenance.manifest_dir) / f"{manifest.run_id}.json"
    )
    return manifest, _write_manifest(manifest, path)


def cmd_run(args: argparse.Namespace) -> Result:
    """``openbtk run CONFIG`` -- validate, execute, and write the manifest (FR-N-01).

    A config with validation errors is refused before anything runs (exit 2).
    The manifest goes to ``--manifest``, else
    ``<provenance.manifest_dir>/<run_id>.json``.

    ``--checkpoint PATH`` resumes from that file if it already holds one for this
    pipeline, and periodically records how far each root loader has read (FR-L-05);
    a successful run deletes it, a failed one leaves it for the next attempt. See
    ``openbtk.pipelines.checkpoint`` for exactly what this guarantees.
    """
    load_components()
    config = PipelineConfig.from_yaml(args.config)
    errors = [
        i for i in Pipeline.from_config(config).validate() if i.severity == "error"
    ]
    if errors:
        body = "\n".join(_format_issue(i) for i in errors)
        return (
            2,
            f"{body}\nrefusing to run an invalid config (see `openbtk validate`).\n",
        )
    manifest, path = _run_config(
        config,
        args.manifest,
        checkpoint_path=args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
    )
    if args.json:
        return (0 if manifest.status == "success" else 1), manifest.model_dump_json(
            indent=2
        ) + "\n"
    return (0 if manifest.status == "success" else 1), _summary(manifest, path)


# ------------------------------------------------------------------- replay


def _redacted_paths(value: Any, prefix: str = "") -> list[str]:
    if value == _REDACTED:
        return [prefix or "<root>"]
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            found += _redacted_paths(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found += _redacted_paths(v, f"{prefix}[{i}]")
    return found


def _compare(old: RunManifest, new: RunManifest) -> tuple[list[str], list[str]]:
    """Compare a recorded run with its replay.

    Returns ``(differences, unverified)``. ``differences`` are real divergences:
    input content or record count, per-step counts, final status. ``unverified``
    names inputs whose content could not be compared because one side has no
    SHA-256 (a directory, or a file over the hashing size cap) -- for those only
    the record count is compared, and the report says so instead of pretending
    the content matched or calling it a divergence.
    """
    diffs: list[str] = []
    unverified: list[str] = []
    new_digests = {d.uri: d for d in new.input_digests}
    for d in old.input_digests:
        other = new_digests.get(d.uri)
        if other is None:
            diffs.append(f"input {d.uri}: no longer read")
        elif (
            d.sha256 is not None
            and other.sha256 is not None
            and (d.sha256 != other.sha256)
        ):
            diffs.append(f"input {d.uri}: content changed since the recorded run")
        elif d.record_count != other.record_count:
            diffs.append(
                f"input {d.uri}: {d.record_count} record(s) then, "
                f"{other.record_count} now"
            )
        elif d.sha256 is None or other.sha256 is None:
            unverified.append(d.uri)
    new_steps = {s.step_id: s for s in new.steps}
    for s in old.steps:
        other_step = new_steps.get(s.step_id)
        if other_step is None:
            diffs.append(f"step {s.step_id}: missing from the replay")
        elif (s.records_in, s.records_out) != (
            other_step.records_in,
            other_step.records_out,
        ):
            diffs.append(
                f"step {s.step_id}: {s.records_in}->{s.records_out} then, "
                f"{other_step.records_in}->{other_step.records_out} now"
            )
    if old.status != new.status:
        diffs.append(f"status: {old.status} then, {new.status} now")
    return diffs, unverified


def cmd_replay(args: argparse.Namespace) -> Result:
    """``openbtk replay MANIFEST`` -- re-execute a recorded run and compare (FR-P-04).

    The manifest carries the config that produced it, so the pipeline is
    rebuilt from that. A manifest never contains secrets (they are redacted
    when it is written), so a config that needed one cannot be replayed from
    the manifest alone; the command says so rather than running with a
    placeholder. Exit 0 means the replay matched the recorded run on input
    content, per-step counts and status; exit 1 means it diverged.
    """
    load_components()
    path = Path(args.manifest)
    try:
        old = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(
            f"{path.name} is not a readable RunManifest.", context={"path": str(path)}
        ) from e
    redacted = _redacted_paths(old.config)
    if redacted:
        return 2, (
            "This manifest had secrets redacted from its config, so it cannot be "
            f"replayed on its own: {', '.join(redacted)}.\n"
            "Re-run from the original config (openbtk run), supplying them via "
            "${ENV_VAR}.\n"
        )
    config = PipelineConfig.model_validate(old.config)
    errors = [
        i for i in Pipeline.from_config(config).validate() if i.severity == "error"
    ]
    if errors:
        return 2, "\n".join(_format_issue(i) for i in errors) + "\n"
    new, new_path = _run_config(config, args.new_manifest)
    diffs, unverified = _compare(old, new)
    if args.json:
        return (1 if diffs else 0), _json(
            {
                "matches": not diffs,
                "differences": diffs,
                "unverified_inputs": unverified,
                "manifest": str(new_path),
            }
        )
    notes = "".join(
        f"note: input {uri} has no content hash (a directory or a very large "
        "file), so only its record count was compared.\n"
        for uri in unverified
    )
    if diffs:
        listing = "\n".join(f"  - {d}" for d in diffs)
        return 1, (
            _summary(new, new_path)
            + f"replay DIVERGED from the recorded run:\n{listing}\n"
            + notes
        )
    return 0, _summary(new, new_path) + "replay matches the recorded run.\n" + notes


# --------------------------------------------------------------------- deid

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _safe_filename(record_id: str, taken: set[str]) -> str:
    stem = _SAFE_NAME.sub("_", record_id).strip(".") or "record"
    name, n = stem, 1
    while name in taken:
        n += 1
        name = f"{stem}-{n}"
    taken.add(name)
    return name


def cmd_deid(args: argparse.Namespace) -> Result:
    """``openbtk deid INPUT --out DIR`` -- de-identify a corpus, no config (FR-N-04).

    ``INPUT`` is a directory of ``*.txt`` notes or one ``.jsonl`` file. Notes are
    streamed one at a time. Output is the de-identified text plus
    ``deid_summary.json`` (counts by category; never a detected value).
    """
    from openbtk.data.clinical_text.loaders import JSONLLoader, PlainTextLoader
    from openbtk.data.clinical_text.preprocessing import DeidPreprocessor
    from openbtk.deid.schemas import DeidMode

    src = Path(args.input)
    out = Path(args.out)
    if src.is_dir():
        records = PlainTextLoader().load(str(src))
        as_jsonl = False
    elif src.is_file() and src.suffix == ".jsonl":
        records = JSONLLoader().load(str(src))
        as_jsonl = True
    else:
        return 2, f"error: {src} is not a directory of .txt notes or a .jsonl file.\n"
    if out.exists() and any(out.iterdir()) and not args.force:
        return 2, f"error: {out} is not empty (use --force to write into it).\n"
    out.mkdir(parents=True, exist_ok=True)

    recognizers = [r for r in args.recognizers.split(",") if r]
    preprocessor = DeidPreprocessor(mode=DeidMode(args.mode), recognizers=recognizers)
    categories: Counter[str] = Counter()
    risk: Counter[str] = Counter()
    taken: set[str] = set()
    n = 0
    jsonl_file = (
        (out / f"{src.stem}.deid.jsonl").open("w", encoding="utf-8")
        if as_jsonl
        else None
    )
    try:
        for record in records:
            done = preprocessor.process(record)
            n += 1
            report = done.metadata.get("deid_report")
            if isinstance(report, dict):
                counts = report.get("entity_counts")
                if isinstance(counts, dict):
                    for category, count in counts.items():
                        if isinstance(count, int):
                            categories[str(category)] += count
                rr = report.get("residual_risk")
                level = rr.get("level") if isinstance(rr, dict) else None
                if isinstance(level, str):
                    risk[level] += 1
            if jsonl_file is not None:
                jsonl_file.write(
                    json.dumps(
                        {
                            "record_id": done.record_id,
                            "source": done.source,
                            "text": done.text,
                            "deid_status": done.deid_status.value,
                        }
                    )
                    + "\n"
                )
            else:
                name = _safe_filename(done.record_id, taken)
                (out / f"{name}.txt").write_text(done.text, encoding="utf-8")
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    summary = {
        "documents": n,
        "mode": args.mode,
        "recognizers": recognizers,
        "entities_by_category": dict(sorted(categories.items())),
        "residual_risk": dict(sorted(risk.items())),
        "input_sha256": (
            hashlib.sha256(src.read_bytes()).hexdigest() if as_jsonl else None
        ),
    }
    (out / "deid_summary.json").write_text(_json(summary), encoding="utf-8")
    if "ner" not in recognizers:
        sys.stderr.write(
            "warning: without the NER recognizer, names and street addresses are "
            "NOT detected. Use --recognizers rule,ner (needs openbtk[text] and a "
            "spaCy model; see `openbtk doctor`).\n"
        )
    total = sum(categories.values())
    return 0, (
        f"de-identified {n} document(s), {total} span(s) ({args.mode}); wrote {out}\n"
    )


# ------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> Result:
    """``openbtk doctor`` -- what is installed and what is missing (FR-N-06)."""
    report = doctor.collect()
    required = [r for r in (args.require or "").split(",") if r]
    missing = doctor.unmet(report, required)
    if args.json:
        payload = {**report, "required": required, "unmet": missing}
        return (1 if missing else 0), _json(payload)
    text = doctor.format_text(report)
    if missing:
        return 1, text + f"\nrequired but not ready: {', '.join(missing)}\n"
    return 0, text
