"""Experimental sparse-first neighbor-list backend.

This module is a thin wrapper around :mod:`jax_md.partition` that opts into the
direct sparse backend. It exists so benchmarks can keep using a distinct
factory path while the implementation lives in one place.
"""

from __future__ import annotations

from typing import Optional

from jax_md import partition


Array = partition.Array
Box = partition.Box
DisplacementOrMetricFn = partition.DisplacementOrMetricFn
MaskFn = partition.MaskFn
EdgeMaskFn = partition.EdgeMaskFn
NeighborListFns = partition.NeighborListFns
NeighborListFormat = partition.NeighborListFormat


def neighbor_list(
  displacement_or_metric: DisplacementOrMetricFn,
  box: Box,
  r_cutoff: float,
  dr_threshold: float = 0.0,
  capacity_multiplier: float = 1.25,
  disable_cell_list: bool = False,
  mask_self: bool = True,
  custom_mask_function: Optional[MaskFn] = None,
  custom_edge_mask: Optional[EdgeMaskFn] = None,
  fractional_coordinates: bool = False,
  format: NeighborListFormat = NeighborListFormat.Dense,
  **static_kwargs,
) -> NeighborListFns:
  static_kwargs.setdefault('sparse_backend', 'direct')
  return partition.neighbor_list(
    displacement_or_metric,
    box,
    r_cutoff,
    dr_threshold=dr_threshold,
    capacity_multiplier=capacity_multiplier,
    disable_cell_list=disable_cell_list,
    mask_self=mask_self,
    custom_mask_function=custom_mask_function,
    custom_edge_mask=custom_edge_mask,
    fractional_coordinates=fractional_coordinates,
    format=format,
    **static_kwargs,
  )
