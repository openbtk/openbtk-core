"""The deprecation mechanism behind OpenBTK's stability policy (M11 task 11.3).

The policy itself is in ``mkdocs/stability.md``. In one sentence: **a public name
is never removed without first being deprecated, still working and warning, for at
least one minor release** -- and a registry key or config field is never removed
within a major version at all.

This module is the machinery:

* :class:`OpenBTKDeprecationWarning` -- the warning category. It subclasses
  ``DeprecationWarning``, so Python hides it by default outside ``__main__`` and
  tests; run with ``-W default::openbtk.core.deprecation.OpenBTKDeprecationWarning``
  (or ``-W error::...``) to see or enforce it. OpenBTK's own test suite turns it
  into an **error**, so nothing inside OpenBTK may call a deprecated API.
* :func:`deprecated` -- a decorator for a function, method or class.
* :func:`warn_deprecated` -- for a deprecation that is not a call (a renamed
  registry key, a config field).
* ``Registry.register_alias(..., since=..., removal=...)`` -- the mechanism for
  renaming a registry key: the old key keeps resolving and warns.
"""

from __future__ import annotations

import functools
import re
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

_F = TypeVar("_F")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


class OpenBTKDeprecationWarning(DeprecationWarning):
    """Raised (as a warning) when a deprecated OpenBTK API is used."""


def _check_version(label: str, value: str) -> None:
    if not _VERSION.match(value):
        raise ValueError(f"{label} must look like '1.2.0', got {value!r}")


def _message(name: str, since: str, removal: str, use: str | None) -> str:
    text = (
        f"{name} is deprecated since OpenBTK {since} and will be removed in {removal}"
    )
    return f"{text}; use {use} instead." if use else f"{text}."


def warn_deprecated(
    name: str, *, since: str, removal: str, use: str | None = None, stacklevel: int = 2
) -> None:
    """Emit an :class:`OpenBTKDeprecationWarning` for ``name``.

    Args:
        name: What is deprecated, as the caller knows it (``"registry key 'x'"``).
        since: The release that deprecated it, e.g. ``"1.2.0"``.
        removal: The release that will remove it, e.g. ``"2.0.0"``.
        use: What to use instead, if anything.
        stacklevel: As for :func:`warnings.warn`; the default points at the caller.

    Example:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     warn_deprecated("old()", since="1.2.0", removal="2.0.0", use="new()")
        >>> message = str(caught[0].message)
        >>> "1.2.0" in message and "2.0.0" in message and "new()" in message
        True
    """
    _check_version("since", since)
    _check_version("removal", removal)
    warnings.warn(
        _message(name, since, removal, use),
        OpenBTKDeprecationWarning,
        stacklevel=stacklevel,
    )


def deprecated(
    *, since: str, removal: str, use: str | None = None
) -> Callable[[_F], _F]:
    """Mark a function, method or class as deprecated.

    Calling it (or instantiating the class) still works and emits an
    :class:`OpenBTKDeprecationWarning` naming the release that removes it and what
    to use instead. A "Deprecated" note is appended to its docstring so the API
    reference shows it, and ``__deprecated__`` holds the message.

    Example:
        >>> import warnings
        >>> @deprecated(since="1.2.0", removal="2.0.0", use="new()")
        ... def old() -> int:
        ...     return 1
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     result = old()
        >>> result, len(caught)
        (1, 1)
    """
    _check_version("since", since)
    _check_version("removal", removal)

    def decorate(obj: Any) -> Any:
        name = (
            f"{obj.__qualname__}()" if not isinstance(obj, type) else obj.__qualname__
        )
        message = _message(name, since, removal, use)
        note = f"\n\nDeprecated since {since}: {message}"
        obj.__doc__ = (obj.__doc__ or "") + note
        obj.__deprecated__ = message

        if isinstance(obj, type):
            target: Any = obj
            original_init = target.__init__

            @functools.wraps(original_init)
            def init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(message, OpenBTKDeprecationWarning, stacklevel=2)
                original_init(self, *args, **kwargs)

            target.__init__ = init
            return obj

        @functools.wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, OpenBTKDeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        wrapper.__deprecated__ = message  # type: ignore[attr-defined]
        wrapper.__doc__ = obj.__doc__
        return wrapper

    return decorate
