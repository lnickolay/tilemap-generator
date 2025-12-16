"""Manages tile pattern data for the WFC algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from enums import Direction

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PatternData:
    """Abstract base class for managing tile patterns and adjacency rules.

    This class provides the core framework for processing the patterns (the local arrangements of tiles) from a sample
    input array. It defines the abstract methods necessary to extract and count the patterns and to determine which
    patterns can be placed next to each other. It stores the extracted patterns and the determined adjacency rules,
    serving as the source of this data for the Wave Function Collapse algorithm.

    Attributes:
        pattern_size: The width and height of the square patterns extracted (in tiles).
        pattern_count: The total number of unique patterns discovered.
    """

    pattern_size: int
    pattern_count: int

    # The 2D sample tile array of a biome which is used for pattern extraction.
    _sample_array: NDArray[np.int_]

    # The tile index of the tile used for padding/default behavior.
    _default_tile_index: np.int_

    # An array storing the frequency count for each pattern (used as probability weights).
    _frequency_hints: NDArray[np.int_]

    # A sorted list of unique pattern objects, where the index corresponds to the pattern ID.
    _patterns: list[_Pattern]

    # The 3D boolean array defining compatibility: [p1, p2, direction] is True exactly if pattern p2 can be placed next
    # to pattern p1 in the specified direction.
    _adjacency_rules: NDArray[np.bool_]

    def __init__(self, sample_array: NDArray[np.int_], pattern_size: int) -> None:
        """Initializes the pattern data structures.

        Stores the input array and pattern size, sets the default tile index, and then calls the abstract methods for
        extracting and counting the tile patterns in the sample array and for determining the adjacency rules.

        Args:
            sample_array: The 2D sample tile array of a biome which is used for pattern extraction.
            pattern_size: The width and height of the square patterns extracted (in tiles).
        """
        self.pattern_size = pattern_size

        self._sample_array = sample_array

        self._default_tile_index = self._sample_array[0, 0]

        self._extract_and_count_patterns()
        self._frequency_hints = np.array([pattern._frequency for pattern in self._patterns], dtype=np.int_)
        self._determine_adjacency_rules()

    def get_compatible_patterns(self, pattern_index: int, direction: Direction) -> list[int]:
        """Returns all compatible pattern indices for a pattern and direction.

        Returns a list of all pattern indices that can legally be placed adjacent to the pattern with the specified
        index in the specified direction, based on the precalculated adjacency rules matrix.

        Args:
            pattern_index: The index of the pattern to check compatibility for.
            direction: The direction to check compatibility for.

        Returns:
            A list of all pattern indices that can legally be placed adjacent to the pattern with the specified index
                in the specified direction.
        """
        compatible_patterns = []
        for other_pattern_index in range(self.pattern_count):
            if self._adjacency_rules[pattern_index, other_pattern_index, direction.value]:
                compatible_patterns.append(other_pattern_index)
        return compatible_patterns

    def get_tile_index_from_pattern_index(self, pattern_index: int) -> int:
        """Returns the tile index at the pattern's top-left corner (0, 0).

        Args:
            pattern_index: The index of the pattern.

        Returns:
            The tile index at the top-left corner (0, 0) of the pattern specified by the given pattern index, or -1 if
                the pattern index is invalid.
        """
        if pattern_index >= 0:
            return int(self._patterns[pattern_index]._tile_arrangement[0, 0])
        else:
            return -1

    def get_tile_grid_from_pattern_grid(self, pattern_grid: NDArray[np.int_]) -> NDArray[np.int_]:
        """Converts a grid of pattern indices into a grid of tile indices.

        Each element in the pattern index grid is mapped to the tile index located at the top-left corner (0, 0) of
        that pattern, effectively generating the final visual tile map.

        Args:
            pattern_grid: A 2D array where each element is a pattern index.

        Returns:
            A 2D array where each element is the tile index at the top-left corner (0, 0) of the corresponding pattern.
        """
        return np.array(
            [
                [self.get_tile_index_from_pattern_index(pattern_grid[row, col]) for col in range(pattern_grid.shape[1])]
                for row in range(pattern_grid.shape[0])
            ]
        )

    @abstractmethod
    def _extract_and_count_patterns(self) -> None:
        """Defines how patterns are extracted from the sample array."""
        pass

    @abstractmethod
    def _determine_adjacency_rules(self) -> None:
        """Defines the logic for determining pattern compatibility."""
        pass

    def _hash_tile_arrangement(self, array: NDArray[np.int_]) -> int:
        """Generates a unique hash for a pattern's tile arrangement."""
        return hash(np.array2string(array))


class PatternDataOverlapping(PatternData):
    """Pattern data implementation for the Overlapping WFC model.

    Patterns of size NxN are extracted by sliding a window over the sample image. Adjacency is determined by checking
    if the (N-1)xN / Nx(N-1) overlap regions match exactly.
    """

    def __init__(self, sample_array: NDArray[np.int_], pattern_size: int) -> None:
        """Initializes the Overlapping pattern data structure."""
        super().__init__(sample_array, pattern_size)

    def _extract_and_count_patterns(self) -> None:
        """Extracts all unique NxN patterns and counts their frequency."""
        self._patterns = []
        patterns_by_hash = {}
        self.pattern_count = 0

        sample_array_padded = np.pad(
            self._sample_array, self.pattern_size - 1, constant_values=self._default_tile_index
        )

        # Add the default pattern consisting of default tiles only (because it might not be featured in the sample array
        # and it is needed for masked areas).
        default_pattern = _Pattern(
            self.pattern_count, np.full((self.pattern_size, self.pattern_size), self._default_tile_index, dtype=np.int_)
        )
        self._patterns.append(default_pattern)
        patterns_by_hash[self._hash_tile_arrangement(default_pattern._tile_arrangement)] = default_pattern
        self.pattern_count += 1

        for row in range(self._sample_array.shape[0] + self.pattern_size - 1):
            for col in range(self._sample_array.shape[1] + self.pattern_size - 1):
                tile_arrangement = sample_array_padded[row : row + self.pattern_size, col : col + self.pattern_size]

                array_transformations = {self._hash_tile_arrangement(tile_arrangement): tile_arrangement}

                for hash_value, array_transformation in array_transformations.items():
                    if hash_value not in patterns_by_hash:
                        new_pattern = _Pattern(self.pattern_count, array_transformation)
                        self._patterns.append(new_pattern)
                        self.pattern_count += 1
                        patterns_by_hash[hash_value] = new_pattern
                    else:
                        patterns_by_hash[hash_value]._frequency += 1

    def _determine_adjacency_rules(self) -> None:
        """Calculates compatibility by checking overlapping pattern regions."""
        self._adjacency_rules = np.full((self.pattern_count, self.pattern_count, len(Direction)), False, dtype=bool)

        # adjacency_rules[p1, p2, direction] is True if and only if it is legal for p2 to be positioned one step to the
        # left of / to the right of / above / below p1 (according to the specified direction).
        for p1 in self._patterns:
            for p2 in self._patterns:
                # Check LEFT compatibility: p1's left N-1 columns must match p2's right N-1 columns.
                if (p1._tile_arrangement[:, :-1] == p2._tile_arrangement[:, 1:]).all():
                    self._adjacency_rules[p1._index, p2._index, Direction.LEFT.value] = True
                # Check RIGHT compatibility: p1's right N-1 columns must match p2's left N-1 columns.
                if (p1._tile_arrangement[:, 1:] == p2._tile_arrangement[:, :-1]).all():
                    self._adjacency_rules[p1._index, p2._index, Direction.RIGHT.value] = True
                # Check UP compatibility: p1's top N-1 rows must match p2's bottom N-1 rows.
                if (p1._tile_arrangement[:-1, :] == p2._tile_arrangement[1:, :]).all():
                    self._adjacency_rules[p1._index, p2._index, Direction.UP.value] = True
                # Check DOWN compatibility: p1's bottom N-1 rows must match p2's top N-1 rows.
                if (p1._tile_arrangement[1:, :] == p2._tile_arrangement[:-1, :]).all():
                    self._adjacency_rules[p1._index, p2._index, Direction.DOWN.value] = True


class PatternDataSimpleTiled(PatternData):
    """Pattern data implementation for the Simple Tiled WFC model.

    Each unique tile in the sample array is treated as a pattern of size 1x1. Adjacency is determined by checking the
    neighboring tiles in the sample array.

    Attributes:
        patterns_by_tile_index: Maps the tile index to its corresponding pattern object.
    """

    patterns_by_tile_index: dict[int, _Pattern]

    def __init__(self, sample_array: NDArray[np.int_]) -> None:
        """Initializes the Simple Tiled pattern data structure."""
        super().__init__(sample_array, 2)

    def _extract_and_count_patterns(self) -> None:
        """Extracts all unique 1x1 patterns and counts their frequency."""
        self._patterns = []
        self.patterns_by_tile_index = {}
        self.pattern_count = 0

        for row in range(self._sample_array.shape[0]):
            for col in range(self._sample_array.shape[1]):
                # This is a 2D 1x1 array containing a single value in the simple tiled version.
                tile_arrangement = self._sample_array[row : row + 1, col : col + 1]
                tile_index = self._sample_array[row, col]

                if tile_index not in self.patterns_by_tile_index:
                    new_pattern = _Pattern(self.pattern_count, tile_arrangement)
                    self._patterns.append(new_pattern)
                    self.pattern_count += 1
                    self.patterns_by_tile_index[tile_index] = new_pattern
                else:
                    self.patterns_by_tile_index[tile_index]._frequency += 1

    def _determine_adjacency_rules(self) -> None:
        """Extracts allowed tile adjacencies directly from the sample array."""
        self._adjacency_rules = np.full((self.pattern_count, self.pattern_count, len(Direction)), False, dtype=bool)

        # Pad the array with default tiles allow adjacency to the default tile for every other tile.
        sample_array_padded = np.pad(self._sample_array, 1, constant_values=self._default_tile_index)

        # adjacency_rules[p1, p2, direction] is True if and only if it is legal for p2 to be positioned one step to the
        # left of / to the right of / above / below p1 (according to the specified direction).
        for row in range(1, self._sample_array.shape[0] + 1):
            for col in range(1, self._sample_array.shape[1] + 1):
                pattern_index = self.patterns_by_tile_index[sample_array_padded[row, col]]._index
                for direction in Direction:
                    neighbor_row = row + direction.to_vector()[0]
                    neighbor_col = col + direction.to_vector()[1]

                    neighbor_pattern_index = self.patterns_by_tile_index[
                        sample_array_padded[neighbor_row, neighbor_col]
                    ]._index
                    # In the Simple Tiled version, [p1, p2, direction] gets set to True only if p1 and p2 occur directly
                    # next to each other at least once in specified direction in the sample array.
                    self._adjacency_rules[pattern_index, neighbor_pattern_index, direction.value] = True


class _Pattern(ABC):
    """Internal class to represent a single unique NxN tile pattern."""

    # The unique integer ID for this pattern.
    _index: int
    # The NxN array of tile indices that define the pattern.
    _tile_arrangement: NDArray[np.int_]
    # The number of times this pattern was found in the sample array.
    _frequency: int

    def __init__(self, index: int, tile_arrangement: NDArray[np.int_]) -> None:
        """Initializes a pattern object. Frequency starts at 1 upon creation."""
        self._index = index
        self._tile_arrangement = tile_arrangement.copy()
        self._frequency = 1
