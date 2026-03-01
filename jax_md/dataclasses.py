# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for defining dataclasses that can be used with jax transformations.

This code was copied and adapted from https://github.com/google/flax/struct.py.

Accessed on 04/29/2020.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict as _asdict
from dataclasses import astuple as _astuple
from dataclasses import field as _field
from dataclasses import fields as _fields
from dataclasses import is_dataclass as _is_dataclass
from dataclasses import replace as _replace
from typing import Any, Callable, Optional, TypeVar, cast, overload

from typing_extensions import dataclass_transform

import jax

__all__ = (
  'dataclass',
  'Settable',
  'static_field',
  'unpack',
  'replace',
  'asdict',
  'astuple',
  'is_dataclass',
  'fields',
  'field',
)


T = TypeVar('T')
S = TypeVar('S', bound='Settable')
_MISSING = dataclasses.MISSING


class Settable:
  """Mixin that provides a typed persistent update helper for dataclasses."""

  def set(self: S, **kwargs: Any) -> S:
    return cast(S, _replace(cast(Any, self), **kwargs))


def static_field(
  *,
  default: Any = _MISSING,
  default_factory: Any = _MISSING,
  init: bool = True,
  repr: bool = True,
  hash: bool | None = None,
  compare: bool = True,
  metadata: dict[str, Any] | None = None,
  kw_only: bool = False,
  **field_kwargs: Any,
) -> Any:
  """Create a field that is treated as static (non-pytree) by JAX."""
  combined_metadata = dict(metadata or {})
  combined_metadata.setdefault('static', True)
  combined_metadata['pytree_node'] = False
  return _field(
    default=default,
    default_factory=default_factory,
    init=init,
    repr=repr,
    hash=hash,
    compare=compare,
    metadata=combined_metadata,
    kw_only=kw_only,
    **field_kwargs,
  )


@overload
def dataclass(
  clz: type[T], *, frozen: bool = True, **dataclass_kwargs: Any
) -> type[T]: ...


@overload
def dataclass(
  *, frozen: bool = True, **dataclass_kwargs: Any
) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(field_specifiers=(static_field, _field))
def dataclass(
  clz: Optional[type[T]] = None,
  *,
  frozen: bool = True,
  **dataclass_kwargs: Any,
) -> type[T] | Callable[[type[T]], type[T]]:
  """Create a class which can be passed to functional transformations.

  Jax transformations such as `jax.jit` and `jax.grad` require objects that are
  immutable and can be mapped over using the `jax.tree_util` methods.

  The `dataclass` decorator makes it easy to define custom classes that can be
  passed safely to Jax by relying on `jax.tree_util.register_dataclass`.

  Args:
      clz: the class that will be transformed by the decorator.
      frozen: whether the resulting dataclass should be frozen. Defaults to True.
      **dataclass_kwargs: additional keyword arguments forwarded to
          `dataclasses.dataclass`.
  Returns:
      The new class.
  """

  if 'frozen' in dataclass_kwargs:
    requested_frozen = dataclass_kwargs.pop('frozen')
    if requested_frozen != frozen:
      raise TypeError(
        "'frozen' must match the decorator argument when provided in dataclass_kwargs"
      )

  def decorate(target_clz: type[T]) -> type[T]:
    data_clz = dataclasses.dataclass(frozen=frozen, **dataclass_kwargs)(
      target_clz
    )
    registered_clz = jax.tree_util.register_dataclass(data_clz)
    if not hasattr(registered_clz, 'set'):
      setattr(registered_clz, 'set', Settable.set)
    return registered_clz

  if clz is None:
    return decorate

  return decorate(clz)


def unpack(dc: Any) -> tuple[Any, ...]:
  """Return a tuple of dataclass attribute values.

  This is a lightweight alternative to :func:`dataclasses.astuple` that avoids
  recursion and respects custom attribute access defined on the dataclass.
  """
  return tuple(getattr(dc, field.name) for field in _fields(dc))


replace = _replace
asdict = _asdict
astuple = _astuple
is_dataclass = _is_dataclass
fields = _fields
field = _field
