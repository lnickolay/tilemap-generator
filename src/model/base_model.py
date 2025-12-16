"""Manages tile/tilemap size and noise/biome maps."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import opensimplex
from perlin_noise import PerlinNoise
from PyQt6 import QtCore as qtc

import constants
from enums import MapType, NoiseType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from model.biome import Biome


class BaseModel(qtc.QObject):
    """
    Manages the state for tile/tilemap size and noise/biome maps.

    This class handles the tracking of tile and tilemap sizes, the generation and management of all required noise maps
    (altitude, precipitation, temperature), and the derivation and storage of the final biome map. It inherits from
    QObject to facilitate communication via signals.

    Signals:
        noise_map_toggled: Emitted when a noise map is enabled or disabled.
        noise_map_limits_changed: Emitted when noise map limits are updated.
        noise_map_updated: Emitted after a noise map has been updated.
        biome_color_map_updated: Emitted after the biome color map has been updated.

    Attributes:
        tilemap_width: The width of the tilemap (in tiles).
        tilemap_height: The height of the tilemap (in tiles).
        random_seed: The current random seed used for noise generation.
        noise_map_min_limits: Current minimum interpolation limits per MapType.
        noise_map_max_limits: Current maximum interpolation limits per MapType.
        biomes: A collection of user-defined Biome objects, indexed by their ID.
        biome_map: The map of biome indices assigning each tile of the tilemap to the biome it belongs to.
    """

    noise_map_toggled = qtc.pyqtSignal(MapType, bool)
    noise_map_limits_changed = qtc.pyqtSignal(MapType, int, int)
    noise_map_updated = qtc.pyqtSignal(MapType, np.ndarray)
    biome_color_map_updated = qtc.pyqtSignal(np.ndarray)

    tilemap_width: int
    tilemap_height: int

    random_seed: int

    noise_map_min_limits: dict[MapType, int]
    noise_map_max_limits: dict[MapType, int]

    biomes: dict[int, Biome]

    biome_map: NDArray[np.int_]

    # The width of each individual tile in pixels.
    _tile_width: int
    # The height of each individual tile in pixels.
    _tile_height: int

    # Flags indicating which noise maps are currently active for biome determination.
    _noise_map_flags: dict[MapType, bool]
    # Stores the generated noise maps (altitude, precipitation, temperature).
    _noise_maps: dict[MapType, NDArray[np.double]]

    # Array of color values which is used to generate a visual representation of the biome map.
    _biome_color_map: NDArray[np.int_]

    def __init__(self) -> None:
        """Initializes the model with default tilemap size and noise map limits.

        Initializes internal attributes and calls _init_noise_and_biome_maps to set up the data structures.
        """
        super().__init__()

        self.tilemap_width = constants.TILEMAP_SIZE_DEFAULT
        self.tilemap_height = constants.TILEMAP_SIZE_DEFAULT

        self.random_seed = random.randint(0, constants.RANDOM_SEED_MAX)

        self.noise_map_min_limits = {map_type: constants.NOISE_MAP_MIN_DEFAULTS[map_type] for map_type in MapType}
        self.noise_map_max_limits = {map_type: constants.NOISE_MAP_MAX_DEFAULTS[map_type] for map_type in MapType}

        self.biomes = {}

        self._tile_width = constants.TILE_SIZE_DEFAULT
        self._tile_height = constants.TILE_SIZE_DEFAULT

        self._noise_map_flags = {map_type: True for map_type in MapType}

        self._init_noise_and_biome_maps()

    def generate_noise_map(self, map_type: MapType, noise_type_str: str, noise_octaves: float) -> None:
        """Generates a new noise map using the specified noise type and octaves.

        The generated values (initially in [-1, 1]) are interpolated to match the user-defined min and max limits for
        the given MapType.

        Args:
            map_type: The type of noise map to generate.
            noise_type_str: The type of noise algorithm to use.
            noise_octaves: The octave setting for the noise function. Higher values result in more high-frequency detail
            and roughness on the map.
        """
        noise_map_pre_interpolation = np.full((self.tilemap_height, self.tilemap_width), 0.0, dtype=np.double)

        # Noise map generation for Perlin noise.
        if noise_type_str == NoiseType.PERLIN.value:
            noise = PerlinNoise(octaves=noise_octaves)
            for col in range(self.tilemap_width):
                for row in range(self.tilemap_height):
                    noise_map_pre_interpolation[row, col] = noise([col / self.tilemap_width, row / self.tilemap_height])
            # Normalize noise array values so that they are in the range of [-1.0, 1.0].
            noise_map_pre_interpolation *= math.sqrt(2)
            # Clipping Perlin noise array values because they can rarely be slightly outside of [-1.0, 1.0].
            noise_map_pre_interpolation.clip(-1.0, 1.0)

        # Noise map generation for OpenSimplex noise.
        elif noise_type_str == NoiseType.OPENSIMPLEX.value:
            opensimplex.random_seed()
            for col in range(self.tilemap_width):
                for row in range(self.tilemap_height):
                    noise_map_pre_interpolation[row, col] = opensimplex.noise2(
                        noise_octaves * col / self.tilemap_width, noise_octaves * row / self.tilemap_height
                    )
            # Normalize noise array values so that they are in the range of [-1.0, 1.0].
            noise_map_pre_interpolation *= 2 / math.sqrt(3)

        # Interpolate the noise map values so they lie between the min and max values for the given noise map type.
        self._noise_maps[map_type] = np.interp(
            noise_map_pre_interpolation,
            [-1.0, 1.0],
            [self.noise_map_min_limits[map_type], self.noise_map_max_limits[map_type]],
        )

        self.noise_map_updated.emit(map_type, self._noise_maps[map_type])

    def generate_biome_map(self) -> None:
        """Generates the biome map based on the noise maps and biome limits.

        Iterates over every tile, determines the correct biome based on the noise values at that position, and updates
        the biome map and the biome color map.
        """
        for col in range(self.tilemap_width):
            for row in range(self.tilemap_height):
                noise_values = {map_type: self._noise_maps[map_type][row, col] for map_type in MapType}
                biome_at_current_pos = self._determine_biome(noise_values)
                if biome_at_current_pos is not None:
                    self.biome_map[row, col] = biome_at_current_pos.index
                    self._biome_color_map[row, col] = biome_at_current_pos.color_rgb
                else:
                    self.biome_map[row, col] = -1
                    self._biome_color_map[row, col] = (0, 0, 0)
        self.biome_color_map_updated.emit(self._biome_color_map)

    def set_tilemap_size(self, tilemap_width: int, tilemap_height: int) -> None:
        """Sets the tilemap size, then reinitializes the noise and biome maps.

        Args:
            tilemap_width: The new width of the tilemap (in tiles).
            tilemap_height: The new height of the tilemap (in tiles).
        """
        self.tilemap_width = tilemap_width
        self.tilemap_height = tilemap_height
        self._init_noise_and_biome_maps()

    def toggle_noise_map(self, map_type: MapType, flag: bool) -> None:
        """Sets the flag that determines if a noise map is used for biome checks.

        Args:
            map_type: The map type to toggle.
            flag: If True, the map is enabled, otherwise, it is disabled.
        """
        self._noise_map_flags[map_type] = flag
        self.noise_map_toggled.emit(map_type, flag)

    def set_noise_map_limits(self, map_type: MapType, min_limit: int, max_limit: int) -> None:
        """Sets the minimum and maximum limits for a specific noise map type.

        Args:
            map_type: The type of noise map to configure.
            min_limit: The new minimum value for the map (inclusive).
            max_limit: The new maximum value for the map (inclusive).
        """
        self.noise_map_min_limits[map_type] = min_limit
        self.noise_map_max_limits[map_type] = max_limit
        self.noise_map_limits_changed.emit(map_type, min_limit, max_limit)

    def _init_noise_and_biome_maps(self) -> None:
        """Initializes/resets all internal noise/biome map data structures."""
        self._noise_maps = {}
        for map_type in MapType:
            self._noise_maps[map_type] = np.full((self.tilemap_height, self.tilemap_width), 0.0, dtype=np.double)
            self.noise_map_updated.emit(map_type, self._noise_maps[map_type])

        self.biome_map = np.full((self.tilemap_height, self.tilemap_width), -1, dtype=np.int_)
        self._biome_color_map = np.full((self.tilemap_height, self.tilemap_width, 3), 0, dtype=np.int_)
        self.biome_color_map_updated.emit(self._biome_color_map)

    def _determine_biome(self, noise_values: dict[MapType, np.double]) -> Biome | None:
        """Returns the first biome matching the given noise values."""
        for biome in self.biomes.values():
            valid = True
            for map_type in MapType:
                # For noise maps that are actually enabled, check if the noise value is within the biome's limits.
                valid = (
                    not self._noise_map_flags[map_type]
                    or biome.min_limits[map_type] <= float(noise_values[map_type]) <= biome.max_limits[map_type]
                )
                if not valid:
                    break
            if valid:
                return biome
        # Returns None if there is no fitting biome.
        return None
