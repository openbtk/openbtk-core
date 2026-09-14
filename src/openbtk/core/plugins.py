"""Third-party plugin discovery via Python entry points.

A third-party package registers new components without OpenBTK ever
importing it directly, by declaring an entry point in its own
``pyproject.toml``:

    [project.entry-points."openbtk.providers"]
    my_eeg = "my_package.eeg:register"

and providing the callable it points to:

    # my_package/eeg.py
    def register() -> None:
        from openbtk.core.registry import LOADER_REGISTRY
        LOADER_REGISTRY.register("loader.biosignals.my_format")(MyEEGLoader)

:func:`load_plugins` discovers every entry point in the ``openbtk.providers``
group and calls it. It is triggered automatically from
``openbtk.core.registry``'s lookup methods (``get``, ``list_keys``, and
friends -- never from ``register``, which plugins themselves call; wiring it
there would make plugin loading re-entrant), so a user never calls this
directly under normal use.
"""

from __future__ import annotations

from importlib.metadata import entry_points

from openbtk.core.logging import get_logger

log = get_logger(__name__)

_ENTRY_POINT_GROUP = "openbtk.providers"

# Process-wide, checked fresh by load_plugins() itself (not by callers) --
# same pattern as core/logging.py's _allow_phi. No public reset function:
# tests that need to exercise this twice in one process poke this module
# attribute directly, which is an accepted, ordinary pattern for a
# module-level singleton flag.
_loaded = False


def load_plugins() -> None:
    """Discover and load every entry point in the ``openbtk.providers`` group.

    Runs at most once per process; a second call is a no-op regardless of
    whether the first call found anything. A plugin that fails to load logs
    a warning and is skipped -- a broken third-party plugin must never
    prevent ``import openbtk``, or any registry lookup, from succeeding.
    Uses a bare ``except Exception`` deliberately: ``SystemExit`` and
    ``KeyboardInterrupt`` are not ``Exception`` subclasses and still
    propagate normally.

    Example:
        The first call's output depends on what plugins happen to be
        installed, so it runs for real (setting the one-shot flag) but its
        output is discarded here rather than checked; the second call is
        then a genuine, reliably-empty-output no-op:

        >>> import contextlib, io
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     load_plugins()
        >>> load_plugins()
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    try:
        discovered = entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as e:
        log.warning("plugins.discovery_failed", error=str(e))
        return

    for ep in discovered:
        try:
            register_fn = ep.load()
            register_fn()
        except Exception as e:
            # "plugin_name", not "name": bare "name" is in the PHI deny-list
            # (core/logging.py) and would be silently dropped, defeating the
            # entire point of this warning -- discovered by actually running
            # this against a real broken entry point, not by inspection.
            log.warning(
                "plugins.load_failed",
                plugin_name=ep.name,
                value=ep.value,
                error=str(e),
            )
        else:
            log.debug("plugins.loaded", plugin_name=ep.name, value=ep.value)
