"""Generic dataclass <-> plain-JSON conversion used by the Phase 2 models.

``to_plain`` turns dataclasses/enums/tuples into JSON-safe values (floats are never rounded).
``from_plain`` rebuilds a dataclass from a dict using its type hints; unknown keys are
ignored and missing keys fall back to field defaults, which is what makes additive schema
changes backwards compatible.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from enum import Enum
from typing import Any, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


def to_plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(v) for v in value]
    return value


def _convert(tp: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        return _convert(args[0], value) if len(args) == 1 else value
    if origin is list:
        (item,) = get_args(tp)
        return [_convert(item, v) for v in value]
    if origin is dict:
        _k, item = get_args(tp)
        return {k: _convert(item, v) for k, v in value.items()}
    if origin is tuple:
        args = get_args(tp)
        return tuple(_convert(a, v) for a, v in zip(args, value))
    if isinstance(tp, type):
        if issubclass(tp, Enum):
            return tp(value)
        if dataclasses.is_dataclass(tp):
            return from_plain(tp, value)
    return value


def from_plain(cls: type[T], data: dict[str, Any]) -> T:
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        if f.name in data:
            kwargs[f.name] = _convert(hints[f.name], data[f.name])
    return cls(**kwargs)
