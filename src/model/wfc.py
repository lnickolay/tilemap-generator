"""Implements the core WFC algorithm as a parallel worker process."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import heapq
import math
from multiprocessing import Process
import random
import sys
from typing import Any, cast, TYPE_CHECKING

import numpy as np

from constants import WFC_MAX_ATTEMPTS_PER_SEGMENT
from enums import Direction, WFCCellCollapseOrder, WFCSegmentOrder, WFCUpdateMode, WFCUpdateType

if TYPE_CHECKING:
    from multiprocessing import Queue
    from multiprocessing.synchronize import Event as EventType

    from numpy.typing import NDArray

    from model.pattern_data import PatternData


class WFC(Process):
    """Worker process that executes the WFC algorithm on a specific biome.

    The WFC algorithm is executed iteratively, segment by segment, to fill the output mask defined for the current
    biome. It communicates updates (collapsed cells, finished segments, finished biome) back to the main process via a
    multiprocessing Queue. When a specific segment cannot be solved after a certain amount of attemts (specified by
    constants.WFC_MAX_ATTEMPTS_PER_SEGMENT), this means that the segment is very hard or even impossible to solve. In
    this case the algorithm backtracks and resets all already solved neighboring segments in order to solve them
    differently, resulting in different tile patterns on the borders to the current segment, which might make it easier
    to solve. Inherits from 'multiprocessing.Process' to allow parallel generation of different biomes.
    """

    # === CONSTRUCTOR PARAMETERS (initialized in __init__()) ===

    # The unique ID of the worker process (equal to the index of the biome it solves).
    _index: int
    # The boolean mask defining the area to be generated for this biome (True for each cell that is part of the biome).
    _output_mask: NDArray[np.bool_]
    # Tile patterns, their frequencies and their adjacency rules, derived from the sample array of the biome.
    _pattern_data: PatternData
    # Seed used for random number generation within the process.
    _random_seed: int
    # Queue used to send generation progress updates back to the main tilemap generator process.
    _update_queue: Queue
    # Event object used to signal the worker process to abort generation immediately.
    _abort_event: EventType
    # Dimensions of the square segments to be processed iteratively (in cells).
    _segment_size: int
    # Strategy for selecting the next cell (e.g., lowest entropy).
    _cell_collapse_order: WFCCellCollapseOrder
    # Strategy for selecting the next segment to process (e.g., row-by-row).
    _segment_order: WFCSegmentOrder
    # Granularity of updates sent back to the main tilemap generator process.
    _update_mode: WFCUpdateMode = WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT

    # === RUNTIME STATE (initialized in run()) ===

    # (height, width) of the unpadded output area (in cells).
    _output_size: tuple[int, int]
    # The width of the padding of default patterns that each segment needs to be padded with before solving it.
    _padding: int
    # The full (padded) output grid storing the pattern indices (-1 for uncollapsed cells).
    _pattern_grid: NDArray[np.int_]

    # === SEGMENT STATE (initialized in _initialize_segment()) ===

    # (height, width) of the currently processed segment, including padding (in cells).
    _grid_size: tuple[int, int]
    # 2D array of '_Cell' objects storing the WFC cell state (coefficients, entropy, enabler counts).
    _cell_grid: NDArray[Any]
    # Stack for the propagation phase (pattern possibilities removed from a cell).
    _pattern_removals: list[_RemovalUpdate]
    # Heap structure of cell coords for efficient cell selection (ordered by entropy/position).
    _uncollapsed_cells_coords: list[_HeapItem]
    # List of cell coords of cells with an already predetermined pattern.
    _predetermined_cells_coords: list[tuple[int, int]]
    # Total number of cells that need to be solved within the current segment.
    _total_uncollapsed_cells: int

    def __init__(
        self,
        index: int,
        output_mask: NDArray[np.bool_],
        pattern_data: PatternData,
        random_seed: int,
        update_queue: Queue,
        abort_event: EventType,
        segment_size: int,
        cell_collapse_order: WFCCellCollapseOrder,
        segment_order: WFCSegmentOrder,
        update_mode: WFCUpdateMode = WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT,
    ) -> None:
        """Initializes the WFC worker process with all necessary config data.

        Args:
            index: The unique ID of the worker process (equal to the index of the biome it solves).
            output_mask: The boolean mask defining the area to be generated for this biome (True for each cell that is
                part of the biome).
            pattern_data: Tile patterns, their frequencies and their adjacency rules, derived from the sample array of
                the biome.
            random_seed: Seed used for random number generation within the process.
            update_queue: Queue used to send generation progress updates back to the main tilemap generator process.
            abort_event: Event object used to signal the worker process to abort generation immediately.
            segment_size: Dimensions of the square segments to be processed iteratively (in cells).
            cell_collapse_order: Strategy for selecting the next cell (e.g., lowest entropy).
            segment_order: Strategy for selecting the next segment to process (e.g., row-by-row).
            update_mode: Granularity of updates sent back to the main tilemap generator process.
        """
        super().__init__()

        self._index = index
        self._output_mask = output_mask
        self._pattern_data = pattern_data
        self._random_seed = random_seed
        self._update_queue = update_queue
        self._abort_event = abort_event
        self._segment_size = segment_size
        self._cell_collapse_order = cell_collapse_order
        self._segment_order = segment_order
        self._update_mode = update_mode

    def run(self) -> None:
        """The main entry point for the process, overriding 'multiprocessing.Process.run()'.

        This method sets up the initial grid, determines the segment processing order, and iteratively processes all
        segments until the biome area is fully collapsed. When processing a segment was unsuccessful, the algorithm
        backtracks and resets all already solved neighboring segments in order to solve them differently, resulting in
        different tile patterns on the borders to the current segment, which might make it easier to solve.
        """
        # Casting from tuple[int, ...] to tuple[int, int] so Mypy doesn't raise errors (self.output_mask is a 2D array).
        self._output_size = cast(tuple[int, int], self._output_mask.shape)
        self._padding = self._pattern_data.pattern_size - 1

        self._pattern_grid = np.pad(np.where(self._output_mask, -1, 0), self._padding, constant_values=0)

        segment_rows = int(math.ceil(self._output_size[0] / self._segment_size))
        segment_cols = int(math.ceil(self._output_size[1] / self._segment_size))

        remaining_segments_coords = []
        finished_segments_coords = []
        if self._segment_order == WFCSegmentOrder.COL_BY_COL:
            for segment_col in range(segment_cols):
                for segment_row in range(segment_rows):
                    segment_coords = segment_row, segment_col
                    if not self._is_segment_empty(segment_coords):
                        remaining_segments_coords.append(segment_coords)
        else:
            for segment_row in range(segment_rows):
                for segment_col in range(segment_cols):
                    segment_coords = segment_row, segment_col
                    if not self._is_segment_empty(segment_coords):
                        remaining_segments_coords.append(segment_coords)

        if self._segment_order == WFCSegmentOrder.RANDOM:
            random.shuffle(remaining_segments_coords)
        else:
            # Reverse the list because the next segment to solve is always popped from the end of the list.
            remaining_segments_coords.reverse()

        while remaining_segments_coords:
            segment_coords = remaining_segments_coords.pop()

            processing_successful = self._process_segment(segment_coords)

            if processing_successful:
                finished_segments_coords.append(segment_coords)

                if self._update_mode == WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT:
                    pattern_grid_segment = self._get_pattern_grid_segment(segment_coords)
                    tilemap_segment = self._pattern_data.get_tile_grid_from_pattern_grid(pattern_grid_segment)
                    self._update_queue.put(
                        [
                            WFCUpdateType.SEGMENT_FINISHED,
                            tilemap_segment,
                            self._get_segment_offset(segment_coords),
                            self._index,
                        ]
                    )

            else:
                remaining_segments_coords.append(segment_coords)

                for direction in Direction:
                    neighbor_segment_row = segment_coords[0] + direction.to_vector()[0]
                    neighbor_segment_col = segment_coords[1] + direction.to_vector()[1]
                    neighbor_segment_coords = (neighbor_segment_row, neighbor_segment_col)

                    if neighbor_segment_coords in finished_segments_coords:
                        finished_segments_coords.remove(neighbor_segment_coords)
                        self._reset_segment(neighbor_segment_coords)
                        remaining_segments_coords.append(neighbor_segment_coords)

                        if (
                            self._update_mode == WFCUpdateMode.ON_COLLAPSED_CELL
                            or self._update_mode == WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT
                        ):
                            tilemap_segment = np.full(
                                self._get_segment_size(neighbor_segment_coords), -1, dtype=np.int_
                            )
                            self._update_queue.put(
                                [
                                    WFCUpdateType.SEGMENT_FINISHED,
                                    tilemap_segment,
                                    self._get_segment_offset(neighbor_segment_coords),
                                    self._index,
                                ]
                            )

        pattern_grid_unpadded = self._pattern_grid[self._padding : -self._padding, self._padding : -self._padding]
        tilemap = self._pattern_data.get_tile_grid_from_pattern_grid(pattern_grid_unpadded)
        self._update_queue.put([WFCUpdateType.FINISHED, tilemap, self._index])

        # Terminate the worker process gracefully after the entire biome (its assigned task) has been generated.
        sys.exit()

    def _is_segment_empty(self, segment_coords: tuple[int, int]) -> bool:
        """Checks if a segment contains no cells from the biome to solve."""
        min_row = segment_coords[0] * self._segment_size
        row_size = min(min_row + self._segment_size, self._output_size[0])
        min_col = segment_coords[1] * self._segment_size
        col_size = min(min_col + self._segment_size, self._output_size[1])

        return not np.any(self._output_mask[min_row : min_row + row_size, min_col : min_col + col_size])

    def _get_slice2d_from_segment_coords(
        self, segment_coords: tuple[int, int]
    ) -> tuple[slice[int, int], slice[int, int]]:
        """Calculates the cell coords slices demarcating an unpadded segment."""
        min_row = segment_coords[0] * self._segment_size
        max_row = min(min_row + self._segment_size, self._output_size[0])
        min_col = segment_coords[1] * self._segment_size
        max_col = min(min_col + self._segment_size, self._output_size[1])
        return slice(min_row, max_row), slice(min_col, max_col)

    def _get_slice2d_padded_from_segment_coords(
        self, segment_coords: tuple[int, int]
    ) -> tuple[slice[int, int], slice[int, int]]:
        """Calculates the cell coords slices demarcating a padded segment."""
        min_row = segment_coords[0] * self._segment_size + self._padding
        max_row = min(min_row + self._segment_size, self._output_size[0] + self._padding)
        min_col = segment_coords[1] * self._segment_size + self._padding
        max_col = min(min_col + self._segment_size, self._output_size[1] + self._padding)
        return slice(min_row, max_row), slice(min_col, max_col)

    def _reset_segment(self, segment_coords: tuple[int, int]) -> None:
        """Resets a segment (to the unsolved state) in the main pattern grid."""
        slice2d = self._get_slice2d_from_segment_coords(segment_coords)
        slice2d_padded = self._get_slice2d_padded_from_segment_coords(segment_coords)

        self._pattern_grid[slice2d_padded] = np.where(self._output_mask[slice2d], -1, 0)

    def _get_segment_offset(self, segment_coords: tuple[int, int]) -> tuple[int, int]:
        """Returns the (row, col) offset of a segment's top-left corner."""
        return segment_coords[0] * self._segment_size, segment_coords[1] * self._segment_size

    def _get_segment_size(self, segment_coords: tuple[int, int]) -> tuple[int, int]:
        """Calculates the precise size of a segment."""
        segment_size = [self._segment_size, self._segment_size]
        if (segment_coords[0] + 1) * self._segment_size > self._output_size[0]:
            segment_size[0] = self._output_size[0] % self._segment_size
        if (segment_coords[1] + 1) * self._segment_size > self._output_size[1]:
            segment_size[1] = self._output_size[1] % self._segment_size
        return segment_size[0], segment_size[1]

    def _get_pattern_grid_segment(
        self, segment_coords: tuple[int, int], include_padding: bool = False
    ) -> NDArray[np.int_]:
        """Returns the 2D array slice of the pattern grid for a segment."""
        segment_offset = self._get_segment_offset(segment_coords)
        if not include_padding:
            min_row = segment_offset[0] + self._padding
            max_row = min(min_row + self._segment_size, self._pattern_grid.shape[0] - self._padding)
            min_col = segment_offset[1] + self._padding
            max_col = min(min_col + self._segment_size, self._pattern_grid.shape[1] - self._padding)
        else:
            min_row = segment_offset[0]
            max_row = min(min_row + self._segment_size + 2 * self._padding, self._pattern_grid.shape[0])
            min_col = segment_offset[1]
            max_col = min(min_col + self._segment_size + 2 * self._padding, self._pattern_grid.shape[1])
        return self._pattern_grid[min_row:max_row, min_col:max_col]

    def _process_segment(self, segment_coords: tuple[int, int]) -> bool:
        """Reapeatedly attempts to solve the current segment using WFC."""
        initialization_successful = self._initialize_segment(segment_coords)

        if not initialization_successful:
            return False

        initial_cell_grid = copy.deepcopy(self._cell_grid)
        initial_uncollapsed_cells_coords = self._uncollapsed_cells_coords.copy()

        failed_attempts = 0
        while failed_attempts < WFC_MAX_ATTEMPTS_PER_SEGMENT:
            solution_successful = self._solve_segment(segment_coords)

            if not solution_successful:
                failed_attempts += 1
                self._cell_grid = copy.deepcopy(initial_cell_grid)
                self._uncollapsed_cells_coords = initial_uncollapsed_cells_coords.copy()

                if (
                    self._update_mode == WFCUpdateMode.ON_COLLAPSED_CELL
                    or self._update_mode == WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT
                ):
                    tilemap_segment = np.full(self._get_segment_size(segment_coords), -1, dtype=np.int_)
                    self._update_queue.put(
                        [
                            WFCUpdateType.SEGMENT_FINISHED,
                            tilemap_segment,
                            self._get_segment_offset(segment_coords),
                            self._index,
                        ]
                    )
            else:
                solved_segment = np.array(
                    [
                        [
                            self._cell_grid[row, col]._pattern_index
                            for col in range(self._padding, self._grid_size[1] - self._padding)
                        ]
                        for row in range(self._padding, self._grid_size[0] - self._padding)
                    ]
                )
                self._get_pattern_grid_segment(segment_coords)[:] = solved_segment
                return True

        return False

    def _initialize_segment(self, segment_coords: tuple[int, int]) -> bool:
        """Sets up initial '_Cell' grid; propagates predertermined cells."""
        input_pattern_grid_segment = self._get_pattern_grid_segment(segment_coords, include_padding=True)
        # Casting from tuple[int, ...] to tuple[int, int] so Mypy doesn't raise errors (input_pattern_grid_segment is a
        # 2D array).
        self._grid_size = cast(tuple[int, int], input_pattern_grid_segment.shape)
        self._cell_grid = np.empty(self._grid_size, dtype=object)

        initial_pattern_enabler_counts = self._determine_initial_pattern_enabler_counts()

        self._pattern_removals = []
        self._uncollapsed_cells_coords = []
        self._predetermined_cells_coords = []

        for row in range(self._grid_size[0]):
            for col in range(self._grid_size[1]):
                input_pattern_index = input_pattern_grid_segment[row, col]
                self._cell_grid[row, col] = _Cell(
                    input_pattern_index, self._pattern_data._frequency_hints, initial_pattern_enabler_counts
                )

                if input_pattern_index == -1:
                    if (
                        self._padding <= row < self._grid_size[0] - self._padding
                        and self._padding <= col < self._grid_size[1] - self._padding
                    ):
                        self._add_to_uncollapsed_cells_coords(self._cell_grid[row, col], (row, col))
                    else:
                        # The uncollapsed cells that are part of the padding don't need to be collapsed.
                        self._cell_grid[row, col]._is_collapsed = True
                else:
                    self._predetermined_cells_coords.append((row, col))

        self._total_uncollapsed_cells = len(self._uncollapsed_cells_coords)

        while self._predetermined_cells_coords:
            coords = self._predetermined_cells_coords.pop()
            for i in range(self._pattern_data.pattern_count):
                if not i == self._cell_grid[coords]._pattern_index:
                    self._pattern_removals.append(_RemovalUpdate(i, coords))

        # Trigger propagation of all predetermined cells' patterns.
        initialization_successful = self._propagate()
        return initialization_successful

    def _solve_segment(self, segment_coords: tuple[int, int]) -> bool:
        """Runs the collapse and propagation loop for a single segment."""
        collapsed_cells = 0
        while collapsed_cells < self._total_uncollapsed_cells:

            if self._abort_event.is_set():
                break

            next_coords = self._choose_next_cell()
            next_cell = self._cell_grid[next_coords]

            self._collapse_cell_at(next_coords)
            collapsed_cells += 1

            if self._update_mode == WFCUpdateMode.ON_COLLAPSED_CELL:
                segment_offset = self._get_segment_offset(segment_coords)
                absolute_row = next_coords[0] + segment_offset[0] - self._padding
                absolute_col = next_coords[1] + segment_offset[1] - self._padding
                self._update_queue.put(
                    [
                        WFCUpdateType.OUTPUT_CELL_COLLAPSED,
                        (absolute_row, absolute_col),
                        self._pattern_data.get_tile_index_from_pattern_index(next_cell._pattern_index),
                    ]
                )

            propagation_successful = self._propagate()

            if not propagation_successful:
                return False

        return True

    def _add_to_uncollapsed_cells_coords(self, cell: _Cell, coords: tuple[int, int]) -> None:
        """Adds cell coords to the priority queue based on collapse order."""
        match self._cell_collapse_order:
            case WFCCellCollapseOrder.LOWEST_ENTROPY_FIRST:
                heapq.heappush(self._uncollapsed_cells_coords, _HeapItem(cell._get_entropy(), coords))
            case WFCCellCollapseOrder.HIGHEST_ENTROPY_FIRST:
                # Use negative entropy to implement a max-heap (highest entropy first).
                heapq.heappush(self._uncollapsed_cells_coords, _HeapItem(-cell._get_entropy(), coords))
            case WFCCellCollapseOrder.ROW_BY_ROW:
                heapq.heappush(
                    self._uncollapsed_cells_coords, _HeapItem(coords[0] * self._segment_size + coords[1], coords)
                )
            case WFCCellCollapseOrder.COL_BY_COL:
                heapq.heappush(
                    self._uncollapsed_cells_coords, _HeapItem(coords[1] * self._segment_size + coords[0], coords)
                )
            case WFCCellCollapseOrder.RANDOM:
                heapq.heappush(self._uncollapsed_cells_coords, _HeapItem(random.random(), coords))

    def _choose_next_cell(self) -> tuple[int, int]:
        """Pops and returns the coordinates of the next cell to collapse."""
        uncollapsed_cell_found = False
        while not uncollapsed_cell_found:
            next_coords = heapq.heappop(self._uncollapsed_cells_coords)._coords
            uncollapsed_cell_found = not self._cell_grid[next_coords]._is_collapsed
        return next_coords

    def _collapse_cell_at(self, coords: tuple[int, int]) -> None:
        """Collapses a cell; adds removed patterns to the propagation stack."""
        cell = self._cell_grid[coords]
        cell._choose_pattern_index()

        # Add all patterns that were NOT chosen to the removal stack for propagation.
        for i in range(self._pattern_data.pattern_count):
            if not i == cell._pattern_index and cell._coefficients[i]:
                cell._coefficients[i] = False
                self._pattern_removals.append(_RemovalUpdate(i, coords))

    def _propagate(self) -> bool:
        """Performs the constraint propagation caused by removed patterns."""
        while self._pattern_removals:
            removal_update = self._pattern_removals.pop()
            for direction in Direction:
                neighbor_row = removal_update._coords[0] + direction.to_vector()[0]
                neighbor_col = removal_update._coords[1] + direction.to_vector()[1]

                if (
                    neighbor_row < 0
                    or neighbor_row >= self._grid_size[0]
                    or neighbor_col < 0
                    or neighbor_col >= self._grid_size[1]
                ):
                    continue

                neighbor_coords = (neighbor_row, neighbor_col)
                neighbor_cell = self._cell_grid[neighbor_coords]

                if not neighbor_cell._is_collapsed:

                    for compatible_pattern_index in self._pattern_data.get_compatible_patterns(
                        removal_update._pattern_index, direction
                    ):
                        opposite_direction = direction.reverse()
                        enabler_counts = neighbor_cell._pattern_enabler_counts[compatible_pattern_index]

                        # Check if the removed pattern was the only remaining enabler in this direction.
                        if enabler_counts[opposite_direction.value] == 1:
                            # Check if pattern is currently still enabled from all other directions (no count of 0). If
                            # this is not the case, it has already been removed from the cell.
                            if 0 not in enabler_counts:
                                contradiction = neighbor_cell._remove_pattern(compatible_pattern_index)
                                if contradiction:
                                    return False

                                if (
                                    self._cell_collapse_order == WFCCellCollapseOrder.LOWEST_ENTROPY_FIRST
                                    or self._cell_collapse_order == WFCCellCollapseOrder.HIGHEST_ENTROPY_FIRST
                                ):
                                    self._add_to_uncollapsed_cells_coords(neighbor_cell, neighbor_coords)
                                self._pattern_removals.append(_RemovalUpdate(compatible_pattern_index, neighbor_coords))

                        neighbor_cell._pattern_enabler_counts[compatible_pattern_index, opposite_direction.value] -= 1

        return True

    def _determine_initial_pattern_enabler_counts(self) -> NDArray[np.int_]:
        """Calculates the initial count of patterns that enable each pattern."""
        initial_pattern_enabler_counts = np.full((self._pattern_data.pattern_count, len(Direction)), 0, dtype=np.int_)
        for pattern_index in range(self._pattern_data.pattern_count):
            for direction in Direction:
                initial_pattern_enabler_counts[pattern_index, direction.value] = len(
                    self._pattern_data.get_compatible_patterns(pattern_index, direction)
                )

        return initial_pattern_enabler_counts


class _Cell:
    """Represents a single cell in the WFC grid state."""

    # The final pattern index if the cell is collapsed, otherwise -1.
    _pattern_index: int
    # The total number of possible patterns.
    _pattern_count: int

    # The frequency of each pattern (its number of occurrences on the biome sample array).
    _frequency_hints: NDArray[np.int_] | None
    # For each pattern index and direction, tracks how many possible patterns of the neighbor cell in that direction
    # still enable the pattern with that index.
    _pattern_enabler_counts: NDArray[np.int_] | None

    # Sum of weights of all patterns still possible.
    _sum_of_possible_pattern_weights: int | None
    # Part of the Shannon entropy calculation.
    _sum_of_possible_pattern_weight_log_weights: float | None

    # Boolean array wich contains True for each index of a pattern that is still possible, False otherwise.
    _coefficients: NDArray[np.bool_] | None
    # True if a final pattern has been chosen for this cell.
    _is_collapsed: bool

    # Small random value added to entropy to break ties.
    _entropy_noise: float

    def __init__(
        self, pattern_index: int, frequency_hints: NDArray[np.int_], initial_pattern_enabler_counts: NDArray[np.int_]
    ) -> None:
        """Initializes the cell's WFC state."""
        self._pattern_index = pattern_index
        self._pattern_count = len(frequency_hints)

        if pattern_index == -1:
            self._frequency_hints = frequency_hints
            self._pattern_enabler_counts = initial_pattern_enabler_counts.copy()

            possible_frequency_hints = self._frequency_hints

            self._sum_of_possible_pattern_weights = possible_frequency_hints.sum()
            self._sum_of_possible_pattern_weight_log_weights = (
                possible_frequency_hints * np.log2(possible_frequency_hints)
            ).sum()

            self._coefficients = np.full(self._pattern_count, True, dtype=bool)
            self._is_collapsed = False
        else:
            self._is_collapsed = True

        self._entropy_noise = random.uniform(0.0, 0.00000001)

    def _remove_pattern(self, pattern_index: int) -> bool:
        """Removes a possible pattern and updates the entropy sums."""
        assert self._coefficients is not None
        assert self._frequency_hints is not None
        if self._coefficients[pattern_index]:
            self._coefficients[pattern_index] = False

            removed_freq = self._frequency_hints[pattern_index]
            if removed_freq > 0:
                self._sum_of_possible_pattern_weights -= removed_freq
                # Using math.log2() instead of numpy.log2() here because it is faster for single values.
                self._sum_of_possible_pattern_weight_log_weights -= removed_freq * math.log2(removed_freq)

            # Returns True if the last possible pattern for this cell was just removed, False otherwise.
            return bool((~self._coefficients).all())
        else:
            return False

    def _get_entropy(self) -> float:
        """Calculates the Shannon entropy for the cell."""
        if not self._is_collapsed:
            assert self._sum_of_possible_pattern_weights is not None
            assert self._sum_of_possible_pattern_weight_log_weights is not None
            # Using math.log2() instead of numpy.log2() here because it is faster for single values.
            entropy = math.log2(self._sum_of_possible_pattern_weights) - (
                self._sum_of_possible_pattern_weight_log_weights / self._sum_of_possible_pattern_weights
            )
        else:
            entropy = 1
        return entropy + self._entropy_noise

    def _choose_pattern_index(self) -> None:
        """Randomly picks the cell's pattern, weighed by pattern frequency."""
        assert self._coefficients is not None
        assert self._frequency_hints is not None
        assert self._sum_of_possible_pattern_weights is not None
        remaining = random.randrange(self._sum_of_possible_pattern_weights)
        for i in range(self._pattern_count):
            if self._coefficients[i]:
                if remaining >= self._frequency_hints[i]:
                    remaining -= self._frequency_hints[i]
                else:
                    self._pattern_index = i
                    break
        self._is_collapsed = True


class _RemovalUpdate:
    """Container tracking a removed pattern for propagation."""

    # The index of the pattern that was removed.
    _pattern_index: int
    # The coords of the cell where the pattern was removed.
    _coords: tuple[int, int]

    def __init__(self, pattern_index: int, coords: tuple[int, int]) -> None:
        """Creates a new removal update instance and initializes it."""
        self._pattern_index = pattern_index
        self._coords = coords


@dataclass(order=True)
class _HeapItem:
    """Dataclass storing cell coordinates for the priority queue."""

    # The priority value (e.g. entropy or derived from coordinates).
    _priority: float
    # The coordinates of the cell.
    _coords: tuple[int, int] = field(compare=False)
