"""Compatibility patches for Python's :mod:`dataclasses` module."""

from __future__ import annotations

import sys
import types
from dataclasses import MISSING, Field, field, is_dataclass
import dataclasses
from typing import Any


def ensure_hydra_dataclass_compatibility() -> None:
    """Patch ``dataclasses`` to tolerate Hydra's nested defaults on Python 3.12.

    Hydra defines a number of nested dataclasses with default values that are
    instantiated inline (for example ``override_dirname = OverrideDirname()``).
    Python 3.12 tightened the runtime validation for dataclasses and now treats
    these defaults as mutable, raising ``ValueError`` during class creation.

    The upstream fix landed in newer Hydra releases, but several of our
    dependencies still import the affected versions.  We relax the runtime check
    to allow dataclass instances to be used as defaults while keeping their
    semantics intact for consumers such as Hydra's config store.
    """

    if getattr(dataclasses, "_glados_patch_applied", False):  # pragma: no cover - trivial guard
        return

    def _patched_get_field(cls: type[Any], a_name: str, a_type: type[Any], default_kw_only: Any) -> Field[Any]:
        default = getattr(cls, a_name, MISSING)
        if isinstance(default, Field):
            f = default
        else:
            if isinstance(default, types.MemberDescriptorType):
                default = MISSING
            f = field(default=default)

        f.name = a_name
        f.type = a_type
        f._field_type = dataclasses._FIELD

        typing_module = sys.modules.get("typing")
        if typing_module:
            if (
                dataclasses._is_classvar(a_type, typing_module)
                or (
                    isinstance(f.type, str)
                    and dataclasses._is_type(
                        f.type,
                        cls,
                        typing_module,
                        typing_module.ClassVar,
                        dataclasses._is_classvar,
                    )
                )
            ):
                f._field_type = dataclasses._FIELD_CLASSVAR

        if f._field_type is dataclasses._FIELD:
            dataclasses_module = sys.modules[dataclasses.__name__]
            if (
                dataclasses._is_initvar(a_type, dataclasses_module)
                or (
                    isinstance(f.type, str)
                    and dataclasses._is_type(
                        f.type,
                        cls,
                        dataclasses_module,
                        dataclasses_module.InitVar,
                        dataclasses._is_initvar,
                    )
                )
            ):
                f._field_type = dataclasses._FIELD_INITVAR

        if f._field_type in (dataclasses._FIELD_CLASSVAR, dataclasses._FIELD_INITVAR):
            if f.default_factory is not MISSING:
                raise TypeError(f"field {f.name} cannot have a default factory")
        if f._field_type in (dataclasses._FIELD, dataclasses._FIELD_INITVAR):
            if f.kw_only is MISSING:
                f.kw_only = default_kw_only
        else:
            if f.kw_only is not MISSING:
                raise TypeError(f"field {f.name} is a ClassVar but specifies kw_only")

        if f._field_type is dataclasses._FIELD and f.default is not MISSING:
            if f.default.__class__.__hash__ is None and not is_dataclass(f.default):
                raise ValueError(
                    f"mutable default {type(f.default)} for field {f.name} is not allowed: use default_factory"
                )

        return f

    dataclasses._get_field = _patched_get_field  # type: ignore[attr-defined]
    dataclasses._glados_patch_applied = True  # type: ignore[attr-defined]

