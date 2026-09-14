"""Transform: turn detected spans into de-identified text (docs/04_API_DESIGN.md
section 5, ``DeidMode``).

Operates on the original document text plus a **non-overlapping**,
sorted-by-start list of ``Detection`` spans (``SpanMerger``'s job to
guarantee that; this module defends against a violation but does not fix
one -- see ``_apply_over_text``). Slicing ``text[span.start:span.end]`` to
find out what's actually being replaced is legitimate and necessary here,
in a way it is deliberately NOT in ``Detection`` itself: a report must
never carry the original value, but the transform step's whole job is to
consume it exactly once and turn it into something safe.

Five modes:

  * ``REDACT`` -- uniform ``[REDACTED]``. No category information leaked.
  * ``TAG`` -- ``[CATEGORY]`` (e.g. ``[NAME]``). Useful for downstream NLP
    that needs entity boundaries and types without real content.
  * ``HASH`` -- ``[CATEGORY_HASH_xxxxxxxx]``, an HMAC-SHA256 of the original
    value, keyed (never a plain hash -- the same rainbow-table weakness
    ``ConsistencyStore``'s docstring explains applies here too). Stable
    across calls with the same key with no stored state at all: the hash
    is a pure function of ``(key, category, original)``.
  * ``SURROGATE`` -- a human-readable, sequential placeholder
    (``[NAME_1]``, ``[NAME_2]``, ...), stable per
    ``(patient_id, category, original)`` via ``ConsistencyStore``.
  * ``DATE_SHIFT`` -- shifts a parsed date by a per-patient, deterministic
    number of days (derived the same way ``HASH`` derives its token: an
    HMAC of ``(key, patient_id)``, no stored state). Every date belonging
    to the same patient shifts by the same amount, so the *interval*
    between any two of a patient's dates is preserved exactly --
    docs/07_TEST_CHARTER.md section 3.4's named property.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from openbtk.core.errors import DeidError
from openbtk.deid.consistency import ConsistencyStore
from openbtk.deid.schemas import DeidMode, PHICategory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openbtk.deid.schemas import Detection

_MAX_SHIFT_DAYS = 365
"""Half-width of the per-patient date shift range: shifts land in
[-365, 365] days. No significance beyond "large enough that the shift
itself carries no information about the original date"."""

_DATE_FORMATS: tuple[str, ...] = ("%m/%d/%Y", "%Y-%m-%d")
"""Formats this transform can parse for DATE_SHIFT. A real-world diversity
problem, same as RuleRecognizer's phone pattern -- these two cover the
synthetic corpus's own format (%m/%d/%Y) plus ISO 8601, and extending this
list is a config-level concern, not an architectural one."""


def _hmac_digest(key: bytes, *parts: str) -> bytes:
    mac = hmac.new(key, digestmod=hashlib.sha256)
    for part in parts:
        mac.update(part.encode("utf-8"))
        mac.update(b"\x00")
    return mac.digest()


def _hash_token(key: bytes, category: PHICategory, original: str) -> str:
    digest = _hmac_digest(key, category.value, original).hex()
    return f"[{category.value.upper()}_HASH_{digest[:8]}]"


def _parse_date(value: str) -> date:
    for fmt in _DATE_FORMATS:
        try:
            # .date() immediately discards any time/tz component -- these
            # formats are calendar dates (MM/DD/YYYY, YYYY-MM-DD), which
            # have no timezone concept to begin with.
            return datetime.strptime(value, fmt).date()  # noqa: DTZ007
        except ValueError:
            continue
    raise DeidError(
        "DATE_SHIFT could not parse a detected date in any known format.",
        context={"formats_tried": list(_DATE_FORMATS)},
    )


def _patient_shift_days(key: bytes, patient_id: str) -> int:
    digest = _hmac_digest(key, "date_shift", patient_id)
    span = 2 * _MAX_SHIFT_DAYS + 1
    return (int.from_bytes(digest[:8], "big") % span) - _MAX_SHIFT_DAYS


class Transform:
    """Applies one ``DeidMode`` to every detected span in a document.

    Example:
        >>> from openbtk.core.schemas import TextSpan
        >>> from openbtk.deid.schemas import Detection
        >>> d = Detection(
        ...     category=PHICategory.SSN,
        ...     span=TextSpan(start=5, end=16, label="ssn", confidence=0.9),
        ...     confidence=0.9,
        ...     method="ensemble",
        ... )
        >>> Transform(DeidMode.REDACT).apply("SSN: 123-45-6789.", [d])
        'SSN: [REDACTED].'
    """

    def __init__(
        self,
        mode: DeidMode,
        *,
        key: bytes | None = None,
        consistency_store: ConsistencyStore | None = None,
    ) -> None:
        self._mode = mode
        # Random per-instance by default, exactly like ConsistencyStore's
        # own default -- a fixed, hardcoded key would make HASH/DATE_SHIFT
        # output guessable via a precomputed dictionary, defeating the
        # entire reason this module uses HMAC instead of a plain hash.
        # Pass an explicit key for cross-run reproducibility.
        self._key = key if key is not None else os.urandom(32)
        self._store = (
            consistency_store if consistency_store is not None else ConsistencyStore()
        )
        self._surrogate_counters: dict[PHICategory, int] = {}

    def apply(
        self,
        text: str,
        detections: Sequence[Detection],
        *,
        patient_id: str = "",
    ) -> str:
        """Replace every detected span in ``text`` per this transform's mode.

        Args:
            text: The original document text.
            detections: Non-overlapping detections, any order. Overlap is
                detected and raises rather than silently producing
                corrupted output -- SpanMerger's job is to guarantee this
                never happens; this is a defence, not a recovery path.
            patient_id: Required for SURROGATE and DATE_SHIFT (both need a
                stable per-patient identity); ignored by the other three
                modes.

        Raises:
            DeidError: If two detections overlap, or (DATE_SHIFT only) a
                detected date cannot be parsed in any known format.
        """
        ordered = sorted(detections, key=lambda d: d.span.start)
        pieces: list[str] = []
        cursor = 0
        for detection in ordered:
            span = detection.span
            if span.start < cursor:
                raise DeidError(
                    "Transform received overlapping detections -- "
                    "SpanMerger must run first.",
                    context={"span_start": span.start, "previous_end": cursor},
                )
            pieces.append(text[cursor : span.start])
            original = text[span.start : span.end]
            pieces.append(self._replacement(original, detection.category, patient_id))
            cursor = span.end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _replacement(
        self, original: str, category: PHICategory, patient_id: str
    ) -> str:
        if self._mode is DeidMode.REDACT:
            return "[REDACTED]"
        if self._mode is DeidMode.TAG:
            return f"[{category.value.upper()}]"
        if self._mode is DeidMode.HASH:
            return _hash_token(self._key, category, original)
        if self._mode is DeidMode.SURROGATE:
            return self._store.get_or_create_surrogate(
                patient_id=patient_id,
                category=category,
                original=original,
                surrogate_factory=lambda: self._next_surrogate(category),
            )
        # DeidMode.DATE_SHIFT is the only remaining member.
        shift = _patient_shift_days(self._key, patient_id)
        shifted = _parse_date(original) + timedelta(days=shift)
        return shifted.isoformat()

    def _next_surrogate(self, category: PHICategory) -> str:
        self._surrogate_counters[category] = (
            self._surrogate_counters.get(category, 0) + 1
        )
        return f"[{category.value.upper()}_{self._surrogate_counters[category]}]"
