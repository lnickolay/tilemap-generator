"""Contains all global enumeration classes used throughout the project."""

from __future__ import annotations

from enum import Enum


class NoiseType(Enum):
    """Defines the available noise functions used for noise map creation."""

    PERLIN = "Perlin Noise"
    """Standard Perlin noise function, known for its soft, cloud-like gradients."""
    OPENSIMPLEX = "OpenSimplex Noise"
    """Unpatented alternative to simplex noise, offers improvements over Perlin noise in terms of visual artifacts."""


class MapType(Enum):
    """Defines the types of noise maps used to determine biome placement."""

    ALTITUDE = "Altitude"
    """Represents the height distribution (in meters above sea level)."""

    PRECIPITATION = "Precipitation"
    """Represents the precipitation distribution (in total millimeters per year)."""

    TEMPERATURE = "Temperature"
    """Represents the temperature distribution. (in average degrees Celsius per year)"""


class WFCMode(Enum):
    """Defines the available tiling modes used by the WFC algorithm."""

    SIMPLE_TILED = "Simple Tiled"
    """Adjacency rules only specify which single tiles might be placed next to each other."""
    OVERLAPPING = "Overlapping"
    """Adjacency rules specify which overlapping NxN (N >= 2) tile patterns might be placed next to each other."""


class WFCCellCollapseOrder(Enum):
    """Defines the order in which cells of an output segment are collapsed."""

    LOWEST_ENTROPY_FIRST = "Lowest Entropy First (Default)"
    """Always picks the cell with the lowest entropy as the cell to collapse next."""
    HIGHEST_ENTROPY_FIRST = "Highest Entropy First"
    """Always picks the cell with the highest entropy as the cell to collapse next."""
    ROW_BY_ROW = "Row by Row"
    """Starts with the cell in the upper left corner. Proceeds from left to right, then from top to bottom."""
    COL_BY_COL = "Column by Column"
    """Starts with the cell in the upper left corner. Proceeds from top to bottom, then from left to right."""
    RANDOM = "Random"
    """Always picks the next cell to collapse randomly."""


class WFCSegmentOrder(Enum):
    """Defines the order in which the segments of the output are processed."""

    ROW_BY_ROW = "Row by Row (Default)"
    """Starts with the segment in the upper left corner. Proceeds from left to right, then from top to bottom."""
    COL_BY_COL = "Column by Column"
    """Starts with the segment in the upper left corner. Proceeds from top to bottom, then from left to right."""
    RANDOM = "Random"
    """Always picks the next segment to process randomly."""


class WFCUpdateType(Enum):
    """Defines the types of update messages the WFC worker processes send."""

    SEGMENT_FINISHED = 0
    """Used when a WFC worker process has finished generating a biome segment."""
    FINISHED = 1
    """Used when a WFC worker process has finished generating an entire biome."""
    OUTPUT_CELL_COLLAPSED = 2
    """Used when a WFC worker process has collapsed a tilemap cell."""


class WFCUpdateMode(Enum):
    """Defines the frequency at which the shown output should be updated."""

    ON_COLLAPSED_CELL = "On Each Collapsed Cell"
    """Updates the UI whenever a cell gets collapsed. Considerably slows down tilemap generation."""
    ON_FINISHED_BIOME_SEGMENT = "On Each Finished Biome Segment"
    """Updates the UI whenever a segment of any biome is finished."""
    ON_FINISHED_BIOME = "On Each Finished Biome"
    """Updates the UI whenever an entire biome is finished."""
    ONLY_WHEN_DONE = "Only When Done"
    """Updates the UI once when the entire output tilemap is finished."""


class Direction(Enum):
    """Defines the cardinal directions used for tile adjacency and movement."""

    LEFT = 0
    """Left direction."""
    RIGHT = 1
    """Right direction."""
    UP = 2
    """Upward direction."""
    DOWN = 3
    """Downward direction."""

    def reverse(self) -> Direction:
        """Returns the opposite direction of the current direction."""
        match self:
            case Direction.LEFT:
                return Direction.RIGHT
            case Direction.RIGHT:
                return Direction.LEFT
            case Direction.UP:
                return Direction.DOWN
            case Direction.DOWN:
                return Direction.UP

    def to_vector(self) -> tuple[int, int]:
        """Returns the vector representation for the direction."""
        match self:
            case Direction.LEFT:
                return (0, -1)
            case Direction.RIGHT:
                return (0, 1)
            case Direction.UP:
                return (-1, 0)
            case Direction.DOWN:
                return (1, 0)


class ExampleBiomes(Enum):
    """Defines predefined biome configurations for quick testing."""

    HOMOGENOUS = "Homogenous"
    """A homogenous configuration featuring only the temperate forest biome."""
    SIMPLE = "Simple"
    """A simple configuration featuring the ocean, taiga, temperate forest and rainforest biomes."""
    COMPLEX = "Complex"
    """A complex configuration featuring eleven different biomes."""
