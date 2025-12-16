"""Contains the class for defining environmental zones."""

from __future__ import annotations

from typing import TYPE_CHECKING

import constants

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from enums import MapType


class Biome:
    """Represents a distinct environmental zone of a tilemap.

    This class represents a distinct environmental zone defined by a unique index and a set of altitude, precipitation,
    and temperature constraints (min/max limits). These constraints determine where the biome will be placed when
    generating a biome map using noise maps. The sample array further defines the allowed tiles and their adjacency
    rules within this zone.

    Attributes:
        index: An index that serves as the identifier for the biome. Must be unique.
        name: The name of the biome.
        sample_array: The 2D array of tile indices serving as a sample of what tiles the biome should contain and in
            what ways they can be placed next to each other.
        min_limits: The lower altitude, precipitation and temperature limits below which this biome cannot appear.
        max_limits: The upper altitude, precipitation and temperature limits above which this biome cannot appear.
        color_rgb: The color representing this biome on the biome color map in the noise maps widget.
    """

    index: int
    name: str
    sample_array: NDArray[np.int_] | None
    min_limits: dict[MapType, int]
    max_limits: dict[MapType, int]
    color_rgb: tuple[int, int, int]

    def __init__(
        self,
        index: int,
        name: str,
        sample_array: NDArray[np.int_] | None,
        min_limits: dict[MapType, int] = constants.NOISE_MAP_MIN_DEFAULTS.copy(),
        max_limits: dict[MapType, int] = constants.NOISE_MAP_MAX_DEFAULTS.copy(),
        color_rgb: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """Initializes a new Biome instance.

        The constructor sets all configuration parameters for the biome instance. Refer to the Attributes section
        of the class docstring for detailed descriptions.

        Args:
            index: An index that serves as the identifier for the biome. Must be unique.
            name: The name of the biome.
            sample_array: The 2D array of tile indices serving as a sample of what tiles the biome should contain and in
                what ways they can be placed next to each other.
            min_limits: The lower altitude, precipitation and temperature limits below which this biome cannot appear.
                Defaults to constants.NOISE_MAP_MIN_DEFAULTS.copy().
            max_limits: The upper altitude, precipitation and temperature limits above which this biome cannot appear.
                Defaults to constants.NOISE_MAP_MAX_DEFAULTS.copy().
            color_rgb: The color representing this biome on the biome color map in the noise maps widget. Defaults to
                (255, 255, 255).
        """
        self.index = index
        self.name = name
        self.sample_array = sample_array
        self.min_limits = min_limits
        self.max_limits = max_limits
        self.color_rgb = color_rgb

    def update(
        self,
        name: str,
        sample_array: NDArray[np.int_] | None,
        min_limits: dict[MapType, int],
        max_limits: dict[MapType, int],
        color_rgb: tuple[int, int, int],
    ) -> None:
        """Updates the attributes of the instance with the provided values.

        This method overwrites all the existing biome instance attributes except for its index with the newly provided
        arguments, performing a full state refresh.

        Args:
            name: The new name of the biome.
            sample_array: The new 2D array of tile indices serving as a sample of what tiles the biome should contain
                and in what ways they can be placed next to each other.
            min_limits: The new lower altitude, precipitation and temperature limits below which this biome cannot
                appear.
            max_limits: The new upper altitude, precipitation and temperature limits above which this biome cannot
                appear.
            color_rgb: The new color representing this biome on the biome color map in the noise maps widget.
        """
        self.name = name
        self.sample_array = sample_array
        self.min_limits = min_limits
        self.max_limits = max_limits
        self.color_rgb = color_rgb
