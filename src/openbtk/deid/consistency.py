"""``ConsistencyStore``: stable original-to-surrogate mapping that never
stores an original value (ADR-0006).

Maps ``HMAC-SHA256(key, patient_id \x00 category \x00 original)`` to
whatever surrogate a caller (``openbtk.deid.transforms``, in
``DeidMode.SURROGATE``) decided to use the first time that exact triple was
seen. The point of the HMAC: even someone with full read access to a
persisted store's contents (digest -> surrogate pairs) cannot recover
*which* original value produced *which* digest without also knowing the
key -- "the store is PHI-free at rest by construction" is a real property
of this design, not just a comment.

**Why HMAC and not plain SHA-256** (unlike ``core.logging``'s
record-id hashing, which is a deliberately unkeyed backstop -- see that
module's docstring): a plain hash of ``"John Smith"`` is the same for every
user of this library forever, so anyone could pre-compute a rainbow table
of common names and de-anonymise a leaked store. A per-store, per-key HMAC
makes that infeasible without the key.

**Persistence is the caller's job, not this class's** (CLAUDE.md rule 11:
constructors do no I/O). ``state`` exposes the current digest -> surrogate
mapping for a caller to serialise however it likes; pass the same ``key``
and a previously-saved ``initial_state`` back in to resume cross-run
consistency (ADR-0006: "optional keyed persistence for cross-run
consistency, requiring a user-supplied key"). With no ``key`` given, a
fresh random one is generated -- consistency holds only within that one
``ConsistencyStore`` instance, matching the documented default ("in-memory
per run by default").
"""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from openbtk.deid.schemas import PHICategory

_KEY_BYTES = 32


class ConsistencyStore:
    """See module docstring.

    Example:
        >>> from openbtk.deid.schemas import PHICategory
        >>> store = ConsistencyStore(key=b"0" * 32)
        >>> first = store.get_or_create_surrogate(
        ...     patient_id="p1",
        ...     category=PHICategory.NAME,
        ...     original="Jane Doe",
        ...     surrogate_factory=lambda: "PATIENT_1",
        ... )
        >>> again = store.get_or_create_surrogate(
        ...     patient_id="p1",
        ...     category=PHICategory.NAME,
        ...     original="Jane Doe",
        ...     surrogate_factory=lambda: "PATIENT_2",  # never called this time
        ... )
        >>> first == again
        True
    """

    def __init__(
        self,
        *,
        key: bytes | None = None,
        initial_state: Mapping[str, str] | None = None,
    ) -> None:
        self._key = key if key is not None else os.urandom(_KEY_BYTES)
        self._store: dict[str, str] = dict(initial_state) if initial_state else {}

    def _digest(self, *, patient_id: str, category: PHICategory, original: str) -> str:
        mac = hmac.new(self._key, digestmod=hashlib.sha256)
        for part in (patient_id, category.value, original):
            mac.update(part.encode("utf-8"))
            mac.update(b"\x00")
        return mac.hexdigest()

    def get_or_create_surrogate(
        self,
        *,
        patient_id: str,
        category: PHICategory,
        original: str,
        surrogate_factory: Callable[[], str],
    ) -> str:
        """Return the stable surrogate for this exact
        ``(patient_id, category, original)`` triple.

        Calls ``surrogate_factory()`` only the first time this triple is
        seen; every later call for the same triple (within this store, or a
        later one constructed with the same ``key`` and a persisted
        ``initial_state``) returns the same surrogate without calling the
        factory again.
        """
        digest = self._digest(
            patient_id=patient_id, category=category, original=original
        )
        if digest not in self._store:
            self._store[digest] = surrogate_factory()
        return self._store[digest]

    @property
    def state(self) -> dict[str, str]:
        """The current digest -> surrogate mapping, safe to persist:
        contains no original value, patient id, or key -- only opaque
        digests and whatever surrogate strings the caller chose."""
        return dict(self._store)

    def __len__(self) -> int:
        return len(self._store)
