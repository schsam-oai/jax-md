# Copyright 2022 Google LLC
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

from __future__ import annotations

from typing import Any, NamedTuple, Tuple, cast

import os

import jax
from jax import eval_shape
from jax import random
from jax.tree_util import tree_map

import jax.numpy as jnp

from jax_md import util

import json
import jraph

import e3nn_jax as e3nn

from . import nequip

import optax

from ml_collections import ConfigDict

from flax import serialization
import flax.linen as nn


f32 = jnp.float32
i32 = jnp.int32

GraphsTuple = jraph.GraphsTuple

IrrepsArray = e3nn.IrrepsArray

NUM_ELEMENTS = 94

PyTree = util.PyTree

Array = util.Array


def model_from_config(cfg: ConfigDict) -> nn.Module:
  model_family = cfg.get('model_family', 'nequip')
  if model_family == 'nequip':
    return nequip.model_from_config(cfg)
  else:
    raise ValueError(f'Unrecognized model family: {model_family}')


def minimum_batch_size(cfg: ConfigDict) -> int:
  train_batch_size = cfg.get('train_batch_size')
  if train_batch_size is None:
    return 1
  if isinstance(train_batch_size, int):
    return train_batch_size
  if isinstance(train_batch_size, list | tuple):
    return min(int(batch_size) for batch_size in train_batch_size)
  return int(train_batch_size)


class ScaleLROnPlateau(NamedTuple):
  step_size: jax.Array
  minimum_loss: jax.Array
  steps_without_reduction: jax.Array
  max_steps_without_reduction: jax.Array
  reduction_factor: jax.Array


def _cfg_str(cfg: ConfigDict, key: str, default: str | None = None) -> str:
  value = cfg.get(key, default)
  if value is None:
    raise ValueError(f'Missing config value: {key}')
  return str(value)


def _cfg_int(cfg: ConfigDict, key: str, default: int | None = None) -> int:
  value = cfg.get(key, default)
  if value is None:
    raise ValueError(f'Missing config value: {key}')
  if isinstance(value, list | tuple):
    if not value:
      raise ValueError(f'Empty config value: {key}')
    value = value[0]
  return int(value)


def _cfg_float(
  cfg: ConfigDict, key: str, default: float | None = None
) -> float:
  value = cfg.get(key, default)
  if value is None:
    raise ValueError(f'Missing config value: {key}')
  if isinstance(value, list | tuple):
    if not value:
      raise ValueError(f'Empty config value: {key}')
    value = value[0]
  return float(value)


def scale_lr_on_plateau(
  initial_step_size: float,
  max_steps_without_reduction: int,
  reduction_factor: float,
) -> optax.GradientTransformation:
  def init_fn(params):
    del params
    return ScaleLROnPlateau(
      jnp.array(initial_step_size, dtype=f32),
      jnp.array(jnp.inf, dtype=f32),
      jnp.array(0, dtype=i32),
      jnp.array(max_steps_without_reduction, dtype=i32),
      jnp.array(reduction_factor, dtype=f32),
    )

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_util.tree_map(lambda g: g * state.step_size, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def optimizer(cfg: ConfigDict) -> optax.GradientTransformation:
  epoch_size = _cfg_int(cfg, 'epoch_size', -1)
  # TODO:
  # if epoch_size < 0:
  #   epoch_size = aggregate_dataset_size(cfg.train_dataset)
  # Maybe replace stubbed in value.

  batch_size = minimum_batch_size(cfg)
  total_steps = _cfg_int(cfg, 'epochs') * (epoch_size // batch_size)
  warmup_steps = _cfg_int(cfg, 'warmup_steps', 0)
  schedule_name = _cfg_str(cfg, 'schedule')
  learning_rate = _cfg_float(cfg, 'learning_rate')

  if schedule_name == 'constant':
    schedule = learning_rate
  elif schedule_name == 'linear_decay':
    schedule = optax.polynomial_schedule(learning_rate, 0.0, 1, total_steps)
  elif schedule_name == 'cosine_decay':
    schedule = optax.cosine_decay_schedule(learning_rate, total_steps)
  elif schedule_name == 'warmup_cosine_decay':
    schedule = optax.warmup_cosine_decay_schedule(
      1e-7, learning_rate, warmup_steps, total_steps
    )
  elif schedule_name == 'scale_on_plateau':
    max_plateau_steps = _cfg_int(
      cfg, 'max_lr_plateau_epochs'
    ) // _cfg_int(cfg, 'epochs_per_eval')
    return optax.chain(
      optax.scale_by_adam(),
      scale_lr_on_plateau(-learning_rate, max_plateau_steps, 0.8),
    )
  else:
    raise ValueError(f'Unknown learning rate schedule, "{schedule_name}".')

  l2_regularization = _cfg_float(cfg, 'l2_regularization', 0.0)
  if l2_regularization == 0.0:
    return optax.adam(schedule)

  return optax.adamw(schedule, weight_decay=l2_regularization)


def load_model(directory: str) -> Tuple[ConfigDict, nn.Module, PyTree]:
  with open(os.path.join(directory, 'config.json'), 'r') as f:
    c = json.loads(json.loads(f.read()))
    c = ConfigDict(c)

  # Now initialize the model and the optimizer functions.
  model = model_from_config(c)
  opt_init, _ = optimizer(c)

  graph = GraphsTuple(
    jnp.zeros((1, NUM_ELEMENTS), dtype=f32),  # Nodes     (nodes, features)
    jnp.zeros((1, 3), dtype=f32),  # dR        (edges, spatial)
    jnp.zeros((1,), dtype=i32),  # senders   (edges,)
    jnp.zeros((1,), dtype=i32),  # receivers (edges,)
    jnp.zeros((1, 1), dtype=f32),  # globals   (graphs,)
    jnp.zeros((1,), dtype=i32),  # n_node    (graphs,)
    jnp.zeros((1,), dtype=i32),
  )  # n_edge    (graphs,)

  def init_opt_and_model(graph):
    key = random.PRNGKey(0)
    params = model.init(key, graph)
    state = opt_init(params)
    return params, state

  abstract_params, abstract_state = eval_shape(init_opt_and_model, graph)

  # Now that we have the structure, load the data using FLAX checkpointing.
  ckpt_data = (0, abstract_params, abstract_state)

  checkpoints = [c for c in os.listdir(directory) if 'checkpoint' in c]
  assert len(checkpoints) == 1

  checkpoint = os.path.join(directory, checkpoints[0])

  with open(checkpoint, 'rb') as f:
    ckpt = cast(tuple[Any, PyTree, Any], serialization.from_bytes(ckpt_data, f.read()))

  params = tree_map(lambda x: x.astype(f32), ckpt[1])
  return c, model, params
