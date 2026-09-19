# Stability policy

From **1.0**, OpenBTK follows [Semantic Versioning](https://semver.org/) and this
policy. Before 1.0, any minor release may break things; every break is recorded in
the [CHANGELOG](https://github.com/openbtk/openbtk-core/blob/main/CHANGELOG.md).

The policy is enforced, not aspirational: `tests/api/` compares the public API,
the registry keys, the command line and the manifest schema versions against
checked-in snapshots, and OpenBTK's own test suite turns its deprecation warning
into an error. A change that breaks the promise fails CI.

## What is public

**Public API is what the [API reference](api/core.md) documents** — and only that.
Concretely, under this policy:

| Public (stable) | Not public (may change any time) |
|---|---|
| Names on the API reference pages: signatures, pydantic fields, enum members, public methods | Anything with a leading underscore |
| **Registry keys** (`loader.clinical_text.plain_text`, ...) | Modules and names not on an API page, even if importable |
| The `openbtk` command: sub-commands, flags, exit codes, and `--json` output | The wording of human-readable output and of error messages |
| Pipeline config (YAML) fields and the manifest schema (`RunManifest`, `EvalManifest`) | Log line text and field names (`OPENBTK_LOG_LEVEL` and `set_log_level` are stable) |
| The safety properties: PHI is never written to a log, error, manifest or report; reports hold identifiers and counts only | The exact detections, scores or wording a component produces (see below) |

**Detection and score outputs are not frozen.** A better recognizer, a fixed
heuristic or a corrected metric changes what a component returns. That is the point
of the project. Such changes are listed in the CHANGELOG, and a change that *makes a
result less strict* (finds less, blocks less) is called out under "Behaviour
change" so you can re-validate. If you depend on exact output, pin the version.

## What counts as a breaking change

Breaking, therefore **major-version only**:

- removing or renaming a public name, registry key, config field, manifest field,
  CLI sub-command, flag or `--json` key;
- a signature change that makes a valid call invalid (removing a parameter, adding
  a required one, narrowing an accepted type);
- making an optional pydantic field required, or removing one;
- a non-additive manifest change (a manifest's `manifest_version` is semver of its
  own, additive-only within a major);
- changing an exit code's meaning.

Not breaking, allowed in a **minor**: adding an optional parameter, a field with a
default, a registry key, a component, a CLI flag or an error subclass; widening an
accepted type; and fixes (below).

**Bug fixes** may change behaviour in a patch or minor release when the old
behaviour contradicted the documentation. They are always in the CHANGELOG's
"Fixed" section. **Security fixes** may change behaviour in a patch release.

## Deprecation

A public name is **never removed without first being deprecated**:

1. In a minor release it is marked deprecated — it keeps working and emits an
   `OpenBTKDeprecationWarning` naming the release that will remove it and what to
   use instead. The CHANGELOG lists it under "Deprecated".
2. It stays for **at least one full minor release**, and is removed only in the next
   **major**.
3. **Registry keys, config fields and manifest fields are never removed within a
   major version.** A renamed key keeps its old spelling as an alias
   (`Registry.register_alias(old, new, since=..., removal=...)`), which warns on
   use; your existing YAML keeps validating.

The warning is a `DeprecationWarning` subclass, so Python hides it by default. To see
it, or to fail on it in your own CI:

```bash
python -W default::openbtk.core.deprecation.OpenBTKDeprecationWarning your_script.py
python -W error::openbtk.core.deprecation.OpenBTKDeprecationWarning -m pytest
```

For maintainers, the mechanism is `openbtk.core.deprecation` (`deprecated`,
`warn_deprecated`) and `register_alias`; see the [core API](api/core.md#deprecation).

## Supported environments

Python **3.11 and later**. A Python version is dropped only in a minor release,
announced in the release before, and never before it is end-of-life upstream.
The six core dependencies keep their current lower bounds within a major;
optional-extra ranges may widen freely and narrow only in a minor with a note.

`openbtk.integrations.langchain` is stable *as an OpenBTK API*, but it adapts to
`langchain-core`, whose own changes are outside our control ([ADR-0001](https://github.com/openbtk/openbtk-core/blob/main/docs/adr/0001-langchain-interop-not-fork.md)):
a LangChain release may require an adapter update in a minor.

## Reading manifests across versions

A manifest records the schema version it was written with. Read a manifest with the
**same or a newer** OpenBTK: readers reject fields they do not know (an older library
cannot read a newer minor's additions). Newer readers read every older manifest of the
same major.

## For contributors

`python tests/api/surface.py --update` regenerates the snapshots after a change.
Adding something is fine, but it must be a deliberate, reviewed snapshot diff. If the
diff shows a removed or changed public name, stop: that needs a deprecation cycle
first. Say why in the commit message and the CHANGELOG.
