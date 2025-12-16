"""Contains global constants and default values used throughout the project."""

from enums import MapType


EXAMPLE_TILESET_IMG_PATH: str = "./assets/tilesets/example_tileset.png"

EXAMPLE_BIOME_OCEAN_PATH: str = "./assets/biome_samples/ocean_30x20.csv"
EXAMPLE_BIOME_SHALLOW_OCEAN_PATH: str = "./assets/biome_samples/shallow_ocean_30x20.csv"
EXAMPLE_BIOME_SHORE_PATH: str = "./assets/biome_samples/shore_30x20.csv"
EXAMPLE_BIOME_POLAR_DESERT_PATH: str = "./assets/biome_samples/polar_desert_30x20.csv"
EXAMPLE_BIOME_GRASSLAND_PATH: str = "./assets/biome_samples/grassland_30x20.csv"
EXAMPLE_BIOME_DESERT_PATH: str = "./assets/biome_samples/desert_30x20.csv"
EXAMPLE_BIOME_TUNDRA_PATH: str = "./assets/biome_samples/tundra_30x20.csv"
EXAMPLE_BIOME_TAIGA_PATH: str = "./assets/biome_samples/taiga_30x20.csv"
EXAMPLE_BIOME_TEMPERATE_FOREST_PATH: str = "./assets/biome_samples/temperate_forest_30x20.csv"
EXAMPLE_BIOME_SAVANNA_PATH: str = "./assets/biome_samples/savanna_30x20.csv"
EXAMPLE_BIOME_RAINFOREST_PATH: str = "./assets/biome_samples/rainforest_30x20.csv"

# === MODEL CONSTANTS ===

WFC_SEGMENT_SIZE_DEFAULT: int = 25
WFC_SEGMENT_SIZE_MIN_LIMIT: int = 5
WFC_SEGMENT_SIZE_MAX_LIMIT: int = 500

WFC_MAX_ATTEMPTS_PER_SEGMENT: int = 10

TILE_SIZE_DEFAULT: int = 16
TILE_SIZE_MIN_LIMIT: int = 8
TILE_SIZE_MAX_LIMIT: int = 256

TILEMAP_SIZE_DEFAULT: int = 200

TILEMAP_SIZE_MIN_LIMIT: int = 10
TILEMAP_SIZE_MAX_LIMIT: int = 500

RANDOM_SEED_MAX: int = 999999999

NOISE_OCTAVES_DEFAULT: float = 5.0
NOISE_OCTAVES_MIN_LIMIT: float = 0.1
NOISE_OCTAVES_MAX_LIMIT: float = 10.0

NOISE_MAP_MIN_DEFAULTS: dict[MapType, int] = {
    MapType.ALTITUDE: -5000,
    MapType.PRECIPITATION: 0,
    MapType.TEMPERATURE: -20,
}
NOISE_MAP_MAX_DEFAULTS: dict[MapType, int] = {
    MapType.ALTITUDE: 5000,
    MapType.PRECIPITATION: 4000,
    MapType.TEMPERATURE: 40,
}
NOISE_MAP_MIN_LIMITS: dict[MapType, int] = {
    MapType.ALTITUDE: -20000,
    MapType.PRECIPITATION: 0,
    MapType.TEMPERATURE: -100,
}
NOISE_MAP_MAX_LIMITS: dict[MapType, int] = {
    MapType.ALTITUDE: 20000,
    MapType.PRECIPITATION: 20000,
    MapType.TEMPERATURE: 100,
}
NOISE_MAP_COLORMAPS: dict[MapType, str] = {
    MapType.ALTITUDE: "gist_earth",
    MapType.PRECIPITATION: "rainbow_r",
    MapType.TEMPERATURE: "plasma",
}
SPIN_BOX_STEP_SIZES: dict[MapType, int] = {
    MapType.ALTITUDE: 100,
    MapType.PRECIPITATION: 100,
    MapType.TEMPERATURE: 1,
}

# === VIEW CONSTANTS ===

LAYOUT_LEFT_SIDE_MAX_WIDTH: int = 350
LAYOUT_LEFT_SIDE_VBOX_SPACING: int = 20
LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH: int = 20
LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH: int = 150
