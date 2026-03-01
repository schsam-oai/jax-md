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

"""Defines utility functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any, Optional, TypeGuard, cast

import jax
from jax.tree_util import register_pytree_node
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import jit

from functools import partial

import numpy as onp
import numpy.typing as npt

try:
  get_backend = cast(
    Callable[[], Any], import_module('jax.extend.backend').get_backend
  )
except ModuleNotFoundError:
  get_backend = cast(
    Callable[[], Any], import_module('jax.lib.xla_bridge').get_backend
  )

Array = jax.Array | npt.NDArray[Any]
PyTree = Any
Scalar = int | float
Numeric = Scalar | Array

i16 = jnp.int16
i32 = jnp.int32
i64 = jnp.int64

f32 = jnp.float32
f64 = jnp.float64


CUSTOM_SIMULATION_TYPE: list[type[Any]] = []


def register_custom_simulation_type(t: type[Any]) -> None:
  global CUSTOM_SIMULATION_TYPE
  CUSTOM_SIMULATION_TYPE += [t]


def check_custom_simulation_type(x: Any) -> None:
  if type(x) in CUSTOM_SIMULATION_TYPE:
    raise ValueError()


def static_cast(*xs: ArrayLike) -> tuple[Array, ...]:
  """Function to cast a value to the lowest dtype that can express it."""
  # NOTE(schsam): static_cast is so named because it cannot be jit.
  if get_backend().platform == 'tpu':
    return tuple(jnp.array(x, jnp.float32) for x in xs)
  return tuple(jnp.array(x, dtype=onp.min_scalar_type(x)) for x in xs)


def register_pytree_namedtuple(cls: type[Any]) -> None:
  register_pytree_node(
    cls, lambda xs: (tuple(xs), None), lambda _, xs: cls(*xs)
  )


def merge_dicts(
  a: dict[Any, Any],
  b: dict[Any, Any],
  ignore_unused_parameters: bool = False,
) -> dict[Any, Any]:
  if not ignore_unused_parameters:
    return {**a, **b}

  merged = dict(a)
  for key in merged.keys():
    b_val = b.get(key)
    if b_val is not None:
      merged[key] = b_val
  return merged


@partial(jit, static_argnums=(1,))
def safe_mask(
  mask: Array,
  fn: Callable[[Array], Array],
  operand: Array,
  placeholder: ArrayLike = 0,
) -> Array:
  masked = jnp.where(mask, operand, 0)
  return jnp.where(mask, fn(masked), placeholder)


def high_precision_sum(
  X: Array,
  axis: Optional[int | Sequence[int]] = None,
  keepdims: bool = False,
):
  """Sums over axes at 64-bit precision then casts back to original dtype."""
  if jnp.issubdtype(X.dtype, jnp.integer):
    dtyp = jnp.int64
  elif jnp.issubdtype(X.dtype, jnp.complexfloating):
    dtyp = jnp.complex128
  else:
    dtyp = jnp.float64

  return jnp.array(
    jnp.sum(X, axis=axis, dtype=dtyp, keepdims=keepdims), dtype=X.dtype
  )


def maybe_downcast(x: Any) -> Array:
  if isinstance(x, jnp.ndarray) and x.dtype is jnp.dtype('float64'):
    return x
  return jnp.array(x, f32)


def is_array(x: object) -> TypeGuard[Array]:
  return isinstance(x, jnp.ndarray | onp.ndarray)


def safe_norm(
  x: Array, axis: int = -1, epsilon: float = 1e-6, keepdims: bool = False
) -> Array:
  """Compute norm with numerical safety for gradients.

  Args:
      x: Input array.
      axis: Axis along which to compute norm.
      epsilon: Small value to avoid division by zero.
      keepdims: Whether to keep reduced dimensions.

  Returns:
      Norm of x with lower bound at epsilon.
  """
  norm = jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=keepdims))
  return jnp.maximum(norm, epsilon)


def safe_arccos(x: Array, epsilon: float = 1e-6) -> Array:
  """Compute arccos with clipping to avoid domain errors.

  Args:
      x: Input array (should be in [-1, 1]).
      epsilon: Safety margin for clipping.

  Returns:
      arccos(x) with input clipped to valid domain.
  """
  x = jnp.clip(x, -1.0 + epsilon, 1.0 - epsilon)
  return jnp.arccos(x)


def normalize(x: Array, axis: int = -1, epsilon: float = 1e-6) -> Array:
  """Normalize vector with numerical safety.

  Args:
      x: Input array.
      axis: Axis along which to normalize.
      epsilon: Small value to avoid division by zero.

  Returns:
      Normalized vector.
  """
  norm = safe_norm(x, axis=axis, epsilon=epsilon, keepdims=True)
  return x / norm
