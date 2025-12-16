"""Manages the visual representation of tilesets and tilemaps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class TilesetManager:
    """Manages the extraction and rendering of tiles from a tileset image.

    This class handles loading the source image (tileset), splitting it into individual tile images, and providing
    methods to convert a tilemap array (a grid of tile indices) into a final, visualized tilemap image.
    """

    # The dimensions (width, height) of a single tile in pixels.
    _tile_size: tuple[int, int]
    # A completely black tile image used to represent uncollapsed (undetermined) cells.
    _uncollapsed_tile: Image.Image
    # A dictionary mapping tile indices (int) to their corresponding PIL Image objects.
    _tiles: dict[int, Image.Image]
    # The source image containing all individual tiles arranged in a grid.
    _tileset_img: Image.Image

    def __init__(self, tileset_img_path: str, tile_size: tuple[int, int]) -> None:
        """Initializes the manager by setting the initial tileset.

        This method immediately calls 'set_tileset' to process the input path and extract all individual tile images
        based on the provided dimensions.

        Args:
            tileset_img_path: The file path to the source tileset image.
            tile_size: The dimensions (width, height) of a single tile in pixels.
        """
        self.set_tileset(tileset_img_path, tile_size)

    def set_tileset(self, tileset_img_path: str, tile_size: tuple[int, int]) -> None:
        """Loads a new tileset image and extracts individual tile images.

        The image is opened and then sliced into individual tiles based on 'tile_size'. Each tile is stored in '_tiles'
        with its index, calculated by reading the tileset image row by row.

        Args:
            tileset_img_path: The file path to the source tileset image.
            tile_size: The dimensions (width, height) of a single tile in pixels.
        """
        self._tile_size = tile_size

        self._uncollapsed_tile = Image.new("RGB", self._tile_size)

        self._tiles = {}

        with Image.open(tileset_img_path) as img:
            self._tileset_img = img.copy()

        rows = self._tileset_img.size[1] // self._tile_size[1]
        cols = self._tileset_img.size[0] // self._tile_size[0]
        for row in range(rows):
            for col in range(cols):
                box = (
                    col * self._tile_size[0],
                    row * self._tile_size[1],
                    (col + 1) * self._tile_size[0],
                    (row + 1) * self._tile_size[1],
                )
                self._tiles[row * cols + col] = self._tileset_img.crop(box)

    def get_tileset_img(self) -> Image.Image:
        """Returns the source PIL image containing all individual tiles.

        Returns:
            The source PIL Image object containing all individual tiles.
        """
        return self._tileset_img

    def get_tilemap_img(self, tilemap_array: NDArray[np.int_]) -> Image.Image:
        """Renders a tilemap array into a complete PIL Image object.

        The tilemap array contains tile indices. The method iterates through the array, fetches the corresponding tile
        images, and pastes them onto a new canvas. An index of -1 is replaced by the black uncollapsed tile.

        Args:
            tilemap_array: A 2D array containing tile indices.

        Returns:
            A PIL Image representing the visual tilemap.
        """
        # tile_size is (width, height) while tilemap_array.shape is (rows, cols), so the indices have to be swapped.
        img_size = (tilemap_array.shape[1] * self._tile_size[0], tilemap_array.shape[0] * self._tile_size[1])
        tilemap_img = Image.new("RGB", img_size)
        for row in range(tilemap_array.shape[0]):
            for col in range(tilemap_array.shape[1]):
                box = (
                    col * self._tile_size[0],
                    row * self._tile_size[1],
                    (col + 1) * self._tile_size[0],
                    (row + 1) * self._tile_size[1],
                )

                if tilemap_array[row, col] != -1:
                    tilemap_img.paste(self._tiles[tilemap_array[row, col]], box)
                else:
                    tilemap_img.paste(self._uncollapsed_tile, box)
        return tilemap_img

    def get_initial_tilemap_img(self, tilemap_size: tuple[int, int]) -> Image.Image:
        """Creates an empty image canvas for a new tilemap of the given size.

        Args:
            tilemap_size: A tuple (rows, columns) defining the dimensions of the target tilemap grid.

        Returns:
            A new, empty (black) PIL Image canvas with the correct pixel dimensions.
        """
        img_size = (tilemap_size[1] * self._tile_size[0], tilemap_size[0] * self._tile_size[1])
        tilemap_img = Image.new("RGB", img_size)
        return tilemap_img

    def update_tilemap_img_tile(
        self, tilemap_img: Image.Image, tile_index: int, tile_coords: tuple[int, int]
    ) -> Image.Image:
        """Updates a single tile at the given coordinates on an existing image.

        Args:
            tilemap_img: The existing image canvas to be modified.
            tile_index: The index of the new tile to paste.
            tile_coords: A tuple (row, column) specifying the location of the tile.

        Returns:
            The modified image canvas (the input image, as the operation is in-place).
        """
        box = (
            tile_coords[1] * self._tile_size[0],
            tile_coords[0] * self._tile_size[1],
            (tile_coords[1] + 1) * self._tile_size[0],
            (tile_coords[0] + 1) * self._tile_size[1],
        )
        tilemap_img.paste(self._tiles[tile_index], box)
        return tilemap_img

    def update_tilemap_img_segment(
        self, tilemap_img: Image.Image, tilemap_segment: NDArray[np.int_], segment_offset: tuple[int, int]
    ) -> Image.Image:
        """Updates a rectangular segment of the tilemap image.

        Args:
            tilemap_img: The existing image canvas to be modified.
            tilemap_segment: A 2D array representing the tile indices of the segment.
            segment_offset: A tuple (row, column) specifying the top-left coordinate of where the segment starts on the
                main canvas.

        Returns:
            The modified image canvas (the input image, as the operation is in-place).
        """
        tilemap_segment_img = self.get_tilemap_img(tilemap_segment)
        box = (
            segment_offset[1] * self._tile_size[0],
            segment_offset[0] * self._tile_size[1],
            (segment_offset[1] + tilemap_segment.shape[1]) * self._tile_size[0],
            (segment_offset[0] + tilemap_segment.shape[0]) * self._tile_size[1],
        )
        tilemap_img.paste(tilemap_segment_img, box)
        return tilemap_img

    def save_tilemap_img(self, tilemap_img: Image.Image, file_path: str) -> None:
        """Saves a generated tilemap image to the specified file path.

        Args:
            tilemap_img: The PIL Image object to be saved.
            file_path: The destination path (including filename and extension).
        """
        tilemap_img.save(file_path)
