"""Compatibility patches for Python's :mod:`dataclasses` module."""

from __future__ import annotations

from dataclasses import MISSING, Field, field, is_dataclass
import dataclasses
from typing import Any, Callable


def ensure_hydra_dataclass_compatibility() -> None:
    """Patch ``dataclasses`` to tolerate Hydra's nested defaults on Python 3.12.

    Hydra 1.3.x defines a number of nested dataclasses with default values that
    are instantiated inline (for example ``override_dirname = OverrideDirname()``).
    Python 3.12 tightened the runtime validation for dataclasses and now treats
    these defaults as mutable, raising ``ValueError`` during class creation.

    The upstream fix landed in newer Hydra releases, but the vendored GigaAM
    dependency still imports the affected version.  To keep the runtime working
    we opportunistically convert such defaults to an equivalent
    ``default_factory`` before ``dataclasses`` performs its validation.
    """

    if getattr(dataclasses, "_glados_patch_applied", False):  # pragma: no cover - trivial guard
        return

    original_get_field: Callable[[type[Any], str, type[Any], Any], Field[Any]] = dataclasses._get_field

    def _patched_get_field(cls: type[Any], a_name: str, a_type: type[Any], default_kw_only: Any) -> Field[Any]:
        current_default = getattr(cls, a_name, MISSING)

        if not isinstance(current_default, Field) and current_default is not MISSING:
            default_cls = current_default.__class__
            # Hydra's problematic defaults are dataclass instances with an
            # explicitly disabled ``__hash__`` implementation.  Switching them to
            # ``default_factory`` makes them compliant with Python 3.12.
            if getattr(default_cls, "__hash__", None) is None and is_dataclass(current_default):
                setattr(cls, a_name, field(default_factory=default_cls))

        return original_get_field(cls, a_name, a_type, default_kw_only)

    dataclasses._get_field = _patched_get_field  # type: ignore[attr-defined]
    dataclasses._glados_patch_applied = True  # type: ignore[attr-defined]

