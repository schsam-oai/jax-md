#!/usr/bin/env python3
"""Benchmark cell-list and neighbor-list workloads from issue #377.

This script is intentionally small and dependency-light so it can serve as a
repeatable harness for trying alternative partition backends, including
experimental JAX or Pallas implementations.

Examples:
  python benchmarks/partition_benchmark.py
  python benchmarks/partition_benchmark.py --n-rep 50 --format ordered_sparse
  python benchmarks/partition_benchmark.py \
    --neighbor-factory jax_md.partition:neighbor_list \
    --cell-factory jax_md.partition:cell_list
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

from jax_md import energy
from jax_md import partition
from jax_md import space

jax.config.update('jax_enable_x64', True)


Factory = Callable[..., Any]

ISSUE_URL = 'https://github.com/jax-md/jax-md/issues/377'
DEFAULT_CELL_FACTORY = 'jax_md.partition:cell_list'
DEFAULT_NEIGHBOR_FACTORY = 'jax_md.partition:neighbor_list'


@dataclass(frozen=True)
class TimingSummary:
  samples_ms: list[float]
  min_ms: float
  median_ms: float
  max_ms: float


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=(
      'Benchmark the partition code paths discussed in '
      f'jax-md issue #377 ({ISSUE_URL}).'
    )
  )
  parser.add_argument(
    '--scenario',
    choices=('lattice',),
    default='lattice',
    help='Workload generator. Only the issue #377 lattice case is supported.',
  )
  parser.add_argument(
    '--n-rep',
    type=int,
    default=30,
    help='Particles per box edge for the cubic lattice.',
  )
  parser.add_argument(
    '--lattice-constant',
    type=float,
    default=2.15443,
    help='Lattice spacing in the synthetic condensed-phase workload.',
  )
  parser.add_argument(
    '--r-cutoff',
    type=float,
    default=8.0,
    help='Neighbor-list cutoff radius.',
  )
  parser.add_argument(
    '--dr-threshold',
    type=float,
    default=2.0,
    help='Neighbor-list halo / rebuild threshold.',
  )
  parser.add_argument(
    '--capacity-multiplier',
    type=float,
    default=1.0,
    help='Capacity multiplier passed to cell and neighbor list factories.',
  )
  parser.add_argument(
    '--format',
    choices=('dense', 'sparse', 'ordered_sparse'),
    default='ordered_sparse',
    help='Neighbor-list storage format.',
  )
  parser.add_argument(
    '--dtype',
    choices=('float32', 'float64'),
    default='float64',
    help='Position dtype.',
  )
  parser.add_argument(
    '--warmup',
    type=int,
    default=1,
    help='Untimed warmup iterations before each measurement.',
  )
  parser.add_argument(
    '--repeats',
    type=int,
    default=5,
    help='Timed iterations per measurement.',
  )
  parser.add_argument(
    '--cell-factory',
    default=DEFAULT_CELL_FACTORY,
    help=(
      'Factory path for cell-list construction, formatted as module:attribute. '
      'The callable must be compatible with jax_md.partition.cell_list.'
    ),
  )
  parser.add_argument(
    '--neighbor-factory',
    default=DEFAULT_NEIGHBOR_FACTORY,
    help=(
      'Factory path for neighbor-list construction, formatted as '
      'module:attribute. The callable must be compatible with '
      'jax_md.partition.neighbor_list.'
    ),
  )
  parser.add_argument(
    '--disable-cell-list',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Disable the internal cell-list path in the neighbor-list backend.',
  )
  parser.add_argument(
    '--measure-energy',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Also benchmark a simple Lennard-Jones neighbor energy evaluation.',
  )
  parser.add_argument(
    '--jit-updates',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Benchmark cell/neighbor update paths through jitted wrappers.',
  )
  parser.add_argument(
    '--json',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Print the report as JSON instead of a human-readable summary.',
  )
  return parser.parse_args()


def resolve_factory(spec: str) -> Factory:
  module_name, attr_name = spec.split(':', 1)
  module = importlib.import_module(module_name)
  return getattr(module, attr_name)


def dtype_from_name(name: str) -> jnp.dtype:
  return jnp.dtype(getattr(jnp, name))


def format_from_name(name: str) -> partition.NeighborListFormat:
  return {
    'dense': partition.Dense,
    'sparse': partition.Sparse,
    'ordered_sparse': partition.OrderedSparse,
  }[name]


def block_until_ready(value: Any) -> Any:
  try:
    return jax.block_until_ready(value)
  except TypeError:
    leaves = tree_leaves(value)
    for leaf in leaves:
      if hasattr(leaf, 'block_until_ready'):
        leaf.block_until_ready()
    return value


def device_scalar(value: Any) -> Any:
  value = jax.device_get(value)
  return value.item() if hasattr(value, 'item') else value


def estimate_nbytes(tree: Any) -> int:
  total = 0
  for leaf in tree_leaves(tree):
    if hasattr(leaf, 'nbytes'):
      total += int(leaf.nbytes)
    elif hasattr(leaf, 'size') and hasattr(leaf, 'dtype'):
      total += int(leaf.size * leaf.dtype.itemsize)
  return total


def maybe_memory_analysis(compiled: Any) -> dict[str, int] | None:
  if not hasattr(compiled, 'memory_analysis'):
    return None
  try:
    stats = compiled.memory_analysis()
  except Exception:
    return None
  return {
    key: int(getattr(stats, key))
    for key in dir(stats)
    if key.endswith('_bytes') and not key.startswith('_')
  }


def memory_stats_snapshot() -> dict[str, int] | None:
  device = jax.devices()[0]
  if not hasattr(device, 'memory_stats'):
    return None
  try:
    stats = device.memory_stats()
  except Exception:
    return None
  return {key: int(value) for key, value in stats.items()}


def human_bytes(num_bytes: int) -> str:
  units = ('B', 'KiB', 'MiB', 'GiB', 'TiB')
  value = float(num_bytes)
  for unit in units:
    if value < 1024.0 or unit == units[-1]:
      return f'{value:.2f} {unit}'
    value /= 1024.0
  return f'{num_bytes} B'


def time_call(
  fn: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[TimingSummary, Any]:
  for _ in range(warmup):
    block_until_ready(fn())

  samples_ms = []
  result = None
  for _ in range(repeats):
    start = time.perf_counter()
    result = fn()
    block_until_ready(result)
    samples_ms.append((time.perf_counter() - start) * 1000.0)

  ordered = sorted(samples_ms)
  median_ms = float(statistics.median(ordered))
  return (
    TimingSummary(
      samples_ms=samples_ms,
      min_ms=min(samples_ms),
      median_ms=median_ms,
      max_ms=max(samples_ms),
    ),
    result,
  )


def lattice_positions(
  n_rep: int, lattice_constant: float, dtype: jnp.dtype
) -> tuple[jnp.ndarray, float, float]:
  axes = [jnp.arange(n_rep, dtype=dtype)] * 3
  positions = jnp.stack(jnp.meshgrid(*axes, indexing='ij'), axis=-1)
  positions = positions.reshape((-1, 3)) * lattice_constant
  box_size = float(n_rep * lattice_constant)
  density = positions.shape[0] / box_size**3
  return positions, box_size, density


def moved_positions(
  positions: jnp.ndarray, box_size: float, dr_threshold: float, dtype: jnp.dtype
) -> jnp.ndarray:
  delta = dr_threshold if dr_threshold > 0 else float(dtype.type(0.25))
  moved = positions.at[0, 0].add(delta)
  return jnp.mod(moved, dtype.type(box_size))


def summarize_cell_list(cell_list: partition.CellList) -> dict[str, Any]:
  named_bytes = estimate_nbytes(cell_list.named_buffer)
  total_bytes = (
    estimate_nbytes(cell_list.position_buffer)
    + estimate_nbytes(cell_list.id_buffer)
    + named_bytes
  )
  return {
    'cell_capacity': int(cell_list.cell_capacity),
    'buffer_bytes': total_bytes,
    'buffer_human': human_bytes(total_bytes),
    'position_buffer_shape': list(cell_list.position_buffer.shape),
    'id_buffer_shape': list(cell_list.id_buffer.shape),
    'did_buffer_overflow': bool(device_scalar(cell_list.did_buffer_overflow)),
  }


def summarize_neighbor_list(neighbor: partition.NeighborList) -> dict[str, Any]:
  mask = partition.neighbor_list_mask(neighbor)
  valid_entries = int(device_scalar(jnp.sum(mask)))
  idx_slots = int(neighbor.idx.size)
  total_bytes = (
    estimate_nbytes(neighbor.idx)
    + estimate_nbytes(neighbor.reference_position)
    + estimate_nbytes(neighbor.error.code)
  )
  return {
    'format': neighbor.format.name,
    'max_occupancy': int(neighbor.max_occupancy),
    'idx_shape': list(neighbor.idx.shape),
    'valid_entries': valid_entries,
    'slot_count': idx_slots,
    'utilization': valid_entries / idx_slots if idx_slots else 0.0,
    'buffer_bytes': total_bytes,
    'buffer_human': human_bytes(total_bytes),
    'did_buffer_overflow': bool(device_scalar(neighbor.did_buffer_overflow)),
    'cell_list_capacity': (
      None
      if neighbor.cell_list_capacity is None
      else int(neighbor.cell_list_capacity)
    ),
    'cell_size': (
      None if neighbor.cell_size is None else float(jax.device_get(neighbor.cell_size))
    ),
  }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
  dtype = dtype_from_name(args.dtype)
  fmt = format_from_name(args.format)
  cell_factory = resolve_factory(args.cell_factory)
  neighbor_factory = resolve_factory(args.neighbor_factory)

  positions, box_size, density = lattice_positions(
    args.n_rep, args.lattice_constant, dtype
  )
  displaced_positions = moved_positions(
    positions, box_size, args.dr_threshold, dtype
  )
  displacement_fn, _ = space.periodic(dtype.type(box_size))
  cell_size = dtype.type(args.r_cutoff + args.dr_threshold)

  cell_list_fn = cell_factory(
    dtype.type(box_size), cell_size, args.capacity_multiplier
  )
  cell_allocate_timing, cell_list = time_call(
    lambda: cell_list_fn.allocate(positions),
    warmup=args.warmup,
    repeats=args.repeats,
  )
  cell_update = lambda pos, cl: cl.update(pos)
  if args.jit_updates:
    cell_update = jax.jit(cell_update)
  cell_update_compiled = None
  if args.jit_updates:
    cell_update_compiled = cell_update.lower(positions, cell_list).compile()
  cell_update_same_timing, _ = time_call(
    lambda: cell_update(positions, cell_list),
    warmup=args.warmup,
    repeats=args.repeats,
  )
  cell_update_rebuild_timing, _ = time_call(
    lambda: cell_update(displaced_positions, cell_list),
    warmup=args.warmup,
    repeats=args.repeats,
  )

  neighbor_fn = neighbor_factory(
    displacement_fn,
    dtype.type(box_size),
    r_cutoff=dtype.type(args.r_cutoff),
    dr_threshold=dtype.type(args.dr_threshold),
    capacity_multiplier=args.capacity_multiplier,
    disable_cell_list=args.disable_cell_list,
    format=fmt,
  )
  neighbor_allocate_timing, neighbors = time_call(
    lambda: neighbor_fn.allocate(positions),
    warmup=args.warmup,
    repeats=args.repeats,
  )
  neighbor_update = lambda pos, nbrs: nbrs.update(pos)
  if args.jit_updates:
    neighbor_update = jax.jit(neighbor_update)
  neighbor_update_compiled = None
  if args.jit_updates:
    neighbor_update_compiled = neighbor_update.lower(positions, neighbors).compile()
  neighbor_update_same_timing, _ = time_call(
    lambda: neighbor_update(positions, neighbors),
    warmup=args.warmup,
    repeats=args.repeats,
  )
  neighbor_update_rebuild_timing, _ = time_call(
    lambda: neighbor_update(displaced_positions, neighbors),
    warmup=args.warmup,
    repeats=args.repeats,
  )

  timings = {
    'cell_allocate': asdict(cell_allocate_timing),
    'cell_update_same': asdict(cell_update_same_timing),
    'cell_update_rebuild': asdict(cell_update_rebuild_timing),
    'neighbor_allocate': asdict(neighbor_allocate_timing),
    'neighbor_update_same': asdict(neighbor_update_same_timing),
    'neighbor_update_rebuild': asdict(neighbor_update_rebuild_timing),
  }

  if args.measure_energy:
    _, energy_fn = energy.lennard_jones_neighbor_list(
      displacement_fn,
      dtype.type(box_size),
      r_cutoff=dtype.type(args.r_cutoff),
      dr_threshold=dtype.type(args.dr_threshold),
      format=fmt,
      disable_cell_list=args.disable_cell_list,
      capacity_multiplier=args.capacity_multiplier,
      neighbor_list_fn=neighbor_factory,
    )
    energy_eval = jax.jit(lambda pos, neighbor: energy_fn(pos, neighbor=neighbor))
    energy_compiled = energy_eval.lower(positions, neighbors).compile()
    energy_timing, energy_value = time_call(
      lambda: energy_eval(positions, neighbors),
      warmup=args.warmup,
      repeats=args.repeats,
    )
    timings['neighbor_energy_eval'] = asdict(energy_timing)
    energy_summary = {'lennard_jones_energy': float(jax.device_get(energy_value))}
  else:
    energy_compiled = None
    energy_summary = {}

  observed_memory = memory_stats_snapshot()
  compiled_memory = {
    'cell_update': maybe_memory_analysis(cell_update_compiled),
    'neighbor_update': maybe_memory_analysis(neighbor_update_compiled),
    'neighbor_energy_eval': maybe_memory_analysis(energy_compiled),
  }

  return {
    'issue_url': ISSUE_URL,
    'device': str(jax.devices()[0]),
    'backend': jax.default_backend(),
    'scenario': args.scenario,
    'n_rep': args.n_rep,
    'particle_count': int(positions.shape[0]),
    'box_size': box_size,
    'density': density,
    'dtype': args.dtype,
    'format': fmt.name,
    'disable_cell_list': args.disable_cell_list,
    'jit_updates': args.jit_updates,
    'cell_factory': args.cell_factory,
    'neighbor_factory': args.neighbor_factory,
    'compiled_memory_bytes': compiled_memory,
    'observed_device_memory': observed_memory,
    'cell_list': summarize_cell_list(cell_list),
    'neighbor_list': summarize_neighbor_list(neighbors),
    'timings_ms': timings,
    **energy_summary,
  }


def print_human_report(report: dict[str, Any]) -> None:
  print(f'Issue: {report["issue_url"]}')
  print(f'Device: {report["device"]} ({report["backend"]})')
  print(
    'Workload: '
    f'{report["particle_count"]} particles, '
    f'n_rep={report["n_rep"]}, '
    f'box={report["box_size"]:.5f}, '
    f'density={report["density"]:.6f}'
  )
  print(
    'Backend factories: '
    f'cell={report["cell_factory"]}, '
    f'neighbor={report["neighbor_factory"]}'
  )
  print(
    'Neighbor format: '
    f'{report["format"]}, disable_cell_list={report["disable_cell_list"]}, '
    f'jit_updates={report["jit_updates"]}'
  )
  print()
  print('Cell list:')
  print(
    f'  capacity={report["cell_list"]["cell_capacity"]}, '
    f'overflow={report["cell_list"]["did_buffer_overflow"]}, '
    f'buffer={report["cell_list"]["buffer_human"]}, '
    f'position_shape={report["cell_list"]["position_buffer_shape"]}'
  )
  print('Neighbor list:')
  print(
    f'  max_occupancy={report["neighbor_list"]["max_occupancy"]}, '
    f'valid_entries={report["neighbor_list"]["valid_entries"]}, '
    f'utilization={report["neighbor_list"]["utilization"]:.4f}, '
    f'overflow={report["neighbor_list"]["did_buffer_overflow"]}, '
    f'buffer={report["neighbor_list"]["buffer_human"]}, '
    f'idx_shape={report["neighbor_list"]["idx_shape"]}'
  )
  if report.get('observed_device_memory') is not None:
    observed = report['observed_device_memory']
    peak = observed.get('peak_bytes_in_use')
    current = observed.get('bytes_in_use')
    if peak is not None:
      print(
        f'Observed device memory: current={human_bytes(current)}, '
        f'peak={human_bytes(peak)}'
      )
  print()
  print('Timings (ms):')
  for name, timing in report['timings_ms'].items():
    print(
      f'  {name}: '
      f'median={timing["median_ms"]:.3f}, '
      f'min={timing["min_ms"]:.3f}, '
      f'max={timing["max_ms"]:.3f}, '
      f'samples={",".join(f"{sample:.3f}" for sample in timing["samples_ms"])}'
    )
  if 'lennard_jones_energy' in report:
    print()
    print(f'LJ energy: {report["lennard_jones_energy"]:.6f}')
  compiled = report.get('compiled_memory_bytes', {})
  for name, stats in compiled.items():
    if stats is None:
      continue
    temp = human_bytes(stats.get('temp_size_in_bytes', 0))
    args_size = human_bytes(stats.get('argument_size_in_bytes', 0))
    out_size = human_bytes(stats.get('output_size_in_bytes', 0))
    print(
      f'{name} compiled memory: args={args_size}, output={out_size}, temp={temp}'
    )


def main() -> None:
  args = parse_args()
  report = build_report(args)
  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
  else:
    print_human_report(report)


if __name__ == '__main__':
  main()
