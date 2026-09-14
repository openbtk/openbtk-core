"""PHI-safe structured logging.

OpenBTK is a library, not an application -- a host application may already
call ``structlog.configure()`` for its own purposes (centralised JSON
logging, a log aggregator, whatever). If OpenBTK also called
``structlog.configure()`` at import time, it would silently clobber that
configuration, which is exactly the kind of surprising global side effect a
library should not have.

So instead of touching global structlog state at all, :func:`get_logger`
wraps each logger individually via ``structlog.wrap_logger()`` with its own,
fixed processor chain. This guarantees PHI-safe redaction on every log call
OpenBTK's own code makes, regardless of what a host application does with
structlog -- verified directly: an OpenBTK logger built this way keeps
redacting even after a host application calls ``structlog.configure()`` with
a completely different chain, and the host's own loggers are correspondingly
unaffected by anything here.

Use this everywhere in OpenBTK's own code instead of raw ``structlog``:

    from openbtk.core.logging import get_logger
    log = get_logger(__name__)
    log.info("loader.start", modality="clinical_text", source=str(path))

Never do this:

    log.debug("processing", text=record.raw_text)   # PHI in a log line

The redaction processor is a backstop, not permission to be careless
(docs/06_SECURITY_COMPLIANCE.md section 3.1) -- it exists for the field
nobody thought to scrub, not as a substitute for not logging PHI in the
first place.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import MutableMapping

# ---------------------------------------------------------------------------
# Redaction policy (docs/06_SECURITY_COMPLIANCE.md section 3.1)
# ---------------------------------------------------------------------------

_DENY_LIST_KEYS = frozenset(
    {
        "raw_text",
        "text",
        "note",
        "content",
        "patient_name",
        "name",
        "mrn",
        "ssn",
        "dob",
        "address",
        "phone",
        "email",
    }
)
_HASH_KEYS = frozenset({"record_id", "patient_id"})
_TRUNCATE_AT = 200
_TRUNCATED_SUFFIX = "...[truncated]"

# Process-wide switch. Defaults to False (redaction ON), matching "installed
# by default" -- there is no way to end up with PHI-unsafe logging except by
# calling configure_logging(allow_phi=True) explicitly. Consulted freshly on
# every log call (not baked into a processor chain at get_logger() time), so
# the order in which get_logger() and configure_logging() are called does not
# matter -- verified this matters: a module-level `log = get_logger(__name__)`
# created before configure_logging() runs must still respect a later change.
_allow_phi: bool = False


def _hash_identifier(value: Any) -> str:
    """Hash an identifier for safe correlation in logs.

    A backstop, not the primary protection: the actual contract
    (CLAUDE.md rule 3) is that callers pass an already-hashed or
    pseudonymous record_id/patient_id. This is a plain SHA-256 prefix, not a
    keyed HMAC -- deliberately simpler than deid's ConsistencyStore, which
    uses a real HMAC key because it must be resistant to targeted
    brute-forcing of a small identifier space. A logging backstop's job is to
    catch a caller who forgot to hash at all; it does not need to defend
    against a determined attacker with log access, since anyone with log
    access in that scenario has a much larger problem than this hash.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _redact_processor(
    logger: Any,  # noqa: ARG001 -- structlog processor signature, unused by design
    method_name: str,  # noqa: ARG001 -- ditto
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """The PHI redaction processor. See this module's docstring.

    Order of operations: drop deny-listed keys entirely, hash the two
    identifier keys, then truncate any remaining long string -- the backstop
    for a field nobody thought to name. Skipped entirely when
    ``configure_logging(allow_phi=True)`` has been called.
    """
    if _allow_phi:
        return event_dict

    for key in _DENY_LIST_KEYS:
        event_dict.pop(key, None)

    for key in _HASH_KEYS:
        if key in event_dict:
            event_dict[key] = _hash_identifier(event_dict[key])

    for key, value in list(event_dict.items()):
        if isinstance(value, str) and len(value) > _TRUNCATE_AT:
            event_dict[key] = value[:_TRUNCATE_AT] + _TRUNCATED_SUFFIX

    return event_dict


def get_logger(name: str) -> Any:
    """Return a PHI-safe structured logger for ``name``.

    Use ``__name__`` as ``name``, matching every other logging convention in
    this codebase. The returned logger's processor chain is fixed at
    creation and independent of any global ``structlog.configure()`` call --
    by a host application or by OpenBTK itself, since OpenBTK never makes
    one.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("probe.event", safe_field=1)  # doctest: +SKIP
    """
    return structlog.wrap_logger(
        structlog.PrintLogger(),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        logger_name=name,
    )


def configure_logging(*, allow_phi: bool = False) -> None:
    """Enable or disable PHI redaction process-wide.

    Args:
        allow_phi: If True, disables the redaction processor for every
            logger obtained via :func:`get_logger`, past and future --
            checked fresh on each log call, not baked in at logger-creation
            time. Logs a warning immediately, because making the unsafe
            path loud is the point (docs/06_SECURITY_COMPLIANCE.md section
            3.1): nobody should be able to end up here by accident.

    Example:
        >>> configure_logging(allow_phi=True)  # doctest: +SKIP
        >>> configure_logging(allow_phi=False)  # restore the safe default
    """
    global _allow_phi
    _allow_phi = allow_phi
    if allow_phi:
        get_logger(__name__).warning(
            "logging.phi_redaction_disabled",
            message=(
                "PHI redaction has been explicitly disabled via "
                "configure_logging(allow_phi=True). Log output may now "
                "contain PHI."
            ),
        )
