"""Contains the widget class for managing, adding, and configuring biomes."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import numpy as np
from PyQt6 import QtCore as qtc
from PyQt6 import QtGui as qtg
from PyQt6 import QtWidgets as qtw
from PIL.ImageQt import ImageQt

import constants
from enums import ExampleBiomes, MapType
from model.biome import Biome
from view.int_spin_box import IntSpinBox

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from model.base_model import BaseModel
    from model.tileset_manager import TilesetManager


class BiomesWidget(qtw.QWidget):
    """The widget class for managing, adding, and configuring biomes.

    This widget contains the controls for adding/deleting biomes and a 'QTabWidget' that holds the individual
    '_BiomeWidget' instances for detailed configuration of each biome. It provides slots to react to global setting
    changes like toggling noise maps or changing their min/max limits.
    """

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel
    # The tileset manager responsible for tileset lookup and image manipulation.
    _tileset_manager: TilesetManager

    # Button to add a new biome.
    _add_biome_button: qtw.QPushButton
    # Button to delete the currently selected biome.
    _delete_selected_biome_button: qtw.QPushButton
    # Button to load predefined example biome sets.
    _create_example_biomes_button: qtw.QPushButton
    # Selection field for which set of example biomes to load.
    _example_biomes_combobox: qtw.QComboBox

    # Tab widget for displaying and managing individual biome configuration widgets.
    _biomes_tabs: qtw.QTabWidget
    # Counter used to assign unique indices to new biomes.
    _biome_counter: int

    def __init__(self, model: BaseModel, tileset_manager: TilesetManager) -> None:
        """Initializes the widget and sets up the GUI elements and connections.

        Args:
            model: The base model managing tile/tilemap size and noise/biome maps.
            tileset_manager: The tileset manager responsible for tileset lookup and image manipulation.
        """
        super().__init__()

        self._model = model
        self._tileset_manager = tileset_manager

        # === WIDGETS ===

        self._add_biome_button = qtw.QPushButton("Add Biome")
        self._add_biome_button.clicked.connect(self._add_biome_tab)

        self._delete_selected_biome_button = qtw.QPushButton("Delete Selected Biome")
        self._delete_selected_biome_button.clicked.connect(self._delete_selected_biome_tab)

        self._create_example_biomes_button = qtw.QPushButton("Create Example Biomes")
        self._create_example_biomes_button.clicked.connect(self._create_example_biomes)

        self._example_biomes_combobox = qtw.QComboBox()
        self._example_biomes_combobox.addItems([option.value for option in ExampleBiomes])
        self._example_biomes_combobox.setCurrentText(ExampleBiomes.COMPLEX.value)

        # === LAYOUT ===

        container_top = qtw.QWidget()
        container_top.setMaximumWidth(600)
        container_top_layout = qtw.QHBoxLayout()
        container_top.setLayout(container_top_layout)
        container_top_layout.addWidget(self._add_biome_button)
        container_top_layout.addWidget(self._delete_selected_biome_button)
        container_top_layout.addSpacing(50)
        container_top_layout.addWidget(self._create_example_biomes_button)
        container_top_layout.addWidget(self._example_biomes_combobox)

        self._biomes_tabs = qtw.QTabWidget()
        self._biome_counter = 0

        layout = qtw.QVBoxLayout()
        self.setLayout(layout)
        layout.addWidget(container_top)
        layout.addWidget(self._biomes_tabs)

    def on_biome_name_changed(self, biome_name: str) -> None:
        """Updates the text of the current tab when the biome name changes.

        Args:
            biome_name: The new name of the biome.
        """
        self._biomes_tabs.setTabText(self._biomes_tabs.currentIndex(), biome_name)

    def on_noise_map_toggled(self, map_type: MapType, flag: bool) -> None:
        """Enables/disables noise map input fields for all biomes.

        Triggered when a noise map is enabled or disabled in the global settings.

        Args:
            map_type: The map type that was toggled.
            flag: True to enable the inputs, False to disable them.
        """
        for biome_widget in self._get_biome_widgets():
            biome_widget._noise_map_min_inputs[map_type].setEnabled(flag)
            biome_widget._noise_map_max_inputs[map_type].setEnabled(flag)

    def on_noise_map_limits_changed(self, map_type: MapType, min_limit: int, max_limit: int) -> None:
        """Sets the valid range of the biome limit fields in all biome widgets.

        Triggered when the limits for a noise map are changed in the global settings.

        Args:
            map_type: The map type for which the limits have changed.
            min_limit: The new minimum limit for the allowed values in the biome limit fields for this map type.
            max_limit: The new maximum limit for the allowed values in the biome limit fields for this map type.
        """
        for biome_widget in self._get_biome_widgets():
            biome_widget._noise_map_min_inputs[map_type].setRange(min_limit, max_limit)
            biome_widget._noise_map_max_inputs[map_type].setRange(min_limit, max_limit)

    def _add_biome_tab(self, biome: Biome | None = None) -> None:
        """Adds a new biome tab and registers the biome in the model."""
        if not biome:
            # If biome is None, a new default biome is created.
            new_biome = Biome(self._biome_counter, f"Biome {self._biome_counter}", None)
        else:
            new_biome = biome

        biome_widget = _BiomeWidget(new_biome, self._model, self._tileset_manager)
        biome_widget._name_input.textChanged.connect(self.on_biome_name_changed)
        biome_widget._update_biome_img()

        self._biomes_tabs.addTab(biome_widget, new_biome.name)
        self._biomes_tabs.setCurrentIndex(self._biomes_tabs.count() - 1)

        self._model.biomes[self._biome_counter] = new_biome

        self._biome_counter += 1

    def _delete_selected_biome_tab(self) -> None:
        """Deletes the selected biome tab, removes the biome from the model."""
        # Casting from QWidget | None to BiomeWidget so Mypy recognizes the type and doesn't raise errors.
        current_biome_widget = cast(_BiomeWidget, self._biomes_tabs.currentWidget())
        if current_biome_widget is not None:
            removed_biome = current_biome_widget._biome
            self._model.biomes.pop(removed_biome.index)

            self._biomes_tabs.removeTab(self._biomes_tabs.currentIndex())

    def _get_noise_map_limits(self) -> tuple[dict[MapType, int], dict[MapType, int]]:
        """Returns an independent copy of the default limits for noise maps."""
        return constants.NOISE_MAP_MIN_DEFAULTS.copy(), constants.NOISE_MAP_MAX_DEFAULTS.copy()

    def _get_biome_widgets(self) -> list[_BiomeWidget]:
        """Returns a list of all '_BiomeWidget' instances in the tab widget."""
        biome_widgets = []
        for i in range(self._biomes_tabs.count()):
            # Casting from QWidget | None to BiomeWidget so Mypy recognizes the type and doesn't raise errors.
            current_biome_widget = cast(_BiomeWidget, self._biomes_tabs.widget(i))
            biome_widgets.append(current_biome_widget)
        return biome_widgets

    def _create_example_biomes(self) -> None:
        """Clears existing biomes, loads a set of predefined example biomes."""
        self._biomes_tabs.clear()
        self._model.biomes = {}
        self._biome_counter = 0

        match ExampleBiomes(self._example_biomes_combobox.currentText()):

            case ExampleBiomes.HOMOGENOUS:

                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TEMPERATE_FOREST_PATH, delimiter=",", dtype=np.int_)
                min_limits = constants.NOISE_MAP_MIN_DEFAULTS.copy()
                max_limits = constants.NOISE_MAP_MAX_DEFAULTS.copy()
                self._add_biome_tab(Biome(0, "Temperate Forest", sample, min_limits, max_limits, color_rgb=(0, 187, 0)))

            case ExampleBiomes.SIMPLE:

                # OCEAN
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_OCEAN_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(0, "Ocean", sample, min_limits, max_limits, color_rgb=(0, 0, 255)))

                # TAIGA
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TAIGA_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 0,
                }
                self._add_biome_tab(Biome(7, "Taiga", sample, min_limits, max_limits, color_rgb=(0, 128, 128)))

                # TEMPERATE FOREST
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TEMPERATE_FOREST_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 0,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 20,
                }
                self._add_biome_tab(Biome(2, "Temperate Forest", sample, min_limits, max_limits, color_rgb=(0, 187, 0)))

                # RAINFOREST
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_RAINFOREST_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 20,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(3, "Rainforest", sample, min_limits, max_limits, color_rgb=(0, 100, 0)))

            case ExampleBiomes.COMPLEX:

                shallow_ocean_min_altitude = -200
                shore_max_altitude = 200

                # OCEAN
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_OCEAN_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: shallow_ocean_min_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(0, "Ocean", sample, min_limits, max_limits, color_rgb=(0, 0, 255)))

                # SHALLOW OCEAN
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_SHALLOW_OCEAN_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shallow_ocean_min_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(1, "Shallow Ocean", sample, min_limits, max_limits, color_rgb=(0, 255, 255)))

                # SHORE
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_SHORE_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: 0,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(2, "Shore", sample, min_limits, max_limits, color_rgb=(252, 208, 70)))

                # POLAR DESERT
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_POLAR_DESERT_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: 0,
                }
                self._add_biome_tab(Biome(3, "Polar Desert", sample, min_limits, max_limits, color_rgb=(255, 255, 255)))

                # GRASSLAND
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_GRASSLAND_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 0,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: 20,
                }
                self._add_biome_tab(Biome(4, "Grassland", sample, min_limits, max_limits, color_rgb=(144, 238, 144)))

                # DESERT
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_DESERT_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: constants.NOISE_MAP_MIN_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 20,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(5, "Desert", sample, min_limits, max_limits, color_rgb=(194, 178, 128)))

                # TUNDRA
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TUNDRA_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: constants.NOISE_MAP_MIN_DEFAULTS[MapType.TEMPERATURE],
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 0,
                }
                self._add_biome_tab(Biome(6, "Tundra", sample, min_limits, max_limits, color_rgb=(175, 238, 238)))

                # TAIGA
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TAIGA_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: 0,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 10,
                }
                self._add_biome_tab(Biome(7, "Taiga", sample, min_limits, max_limits, color_rgb=(0, 128, 128)))

                # TEMPERATE FOREST
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_TEMPERATE_FOREST_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: 10,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: 20,
                }
                self._add_biome_tab(Biome(8, "Temperate Forest", sample, min_limits, max_limits, color_rgb=(0, 187, 0)))

                # SAVANNA
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_SAVANNA_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: 1500,
                    MapType.TEMPERATURE: 20,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: 2500,
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(9, "Savanna", sample, min_limits, max_limits, color_rgb=(128, 128, 0)))

                # RAINFOREST
                sample = np.genfromtxt(constants.EXAMPLE_BIOME_RAINFOREST_PATH, delimiter=",", dtype=np.int_)
                min_limits = {
                    MapType.ALTITUDE: shore_max_altitude,
                    MapType.PRECIPITATION: 2500,
                    MapType.TEMPERATURE: 20,
                }
                max_limits = {
                    MapType.ALTITUDE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.ALTITUDE],
                    MapType.PRECIPITATION: constants.NOISE_MAP_MAX_DEFAULTS[MapType.PRECIPITATION],
                    MapType.TEMPERATURE: constants.NOISE_MAP_MAX_DEFAULTS[MapType.TEMPERATURE],
                }
                self._add_biome_tab(Biome(10, "Rainforest", sample, min_limits, max_limits, color_rgb=(0, 100, 0)))


class _BiomeWidget(qtw.QWidget):
    """Inner widget for configuring the parameters of a single biome object."""

    # The underlying biome object.
    _biome: Biome
    # The RGB color representing the biome.
    _color_rgb: tuple[int, int, int]
    # The currently loaded tile sample array of the biome.
    _biome_sample_array: NDArray[np.int_] | None

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel
    # The tileset manager responsible for tileset lookup and image manipulation.
    _tileset_manager: TilesetManager

    # Input field for the biome name.
    _name_input: qtw.QLineEdit
    # Dictionary of spin boxes for the minimum limits of the noise maps.
    _noise_map_min_inputs: dict[MapType, IntSpinBox]
    # Dictionary of spin boxes for the maximum limits of the noise maps.
    _noise_map_max_inputs: dict[MapType, IntSpinBox]
    # Button that opens the dialog for selecting the RGB color representing the biome.
    _color_chooser_button: qtw.QPushButton

    # Button that opens the dialog for loading the biome sample array.
    _load_sample_file_button: qtw.QPushButton

    # Label to display the visual biome sample array.
    _biome_sample_img_label: qtw.QLabel

    def __init__(self, biome: Biome, model: BaseModel, tileset_manager: TilesetManager) -> None:
        """Initializes the biome widget with a specific biome object."""
        super().__init__()

        self._biome = biome
        self._color_rgb = biome.color_rgb
        self._biome_sample_array = biome.sample_array

        self._model = model
        self._tileset_manager = tileset_manager

        # === LEFT SIDE - WIDGETS ===

        self._name_input = qtw.QLineEdit()
        self._name_input.setText(biome.name)
        self._name_input.editingFinished.connect(self._update_biome)

        self._noise_map_min_inputs = {}
        self._noise_map_max_inputs = {}
        for map_type in MapType:
            self._noise_map_min_inputs[map_type] = IntSpinBox(
                biome.min_limits[map_type],
                self._model.noise_map_min_limits[map_type],
                self._model.noise_map_max_limits[map_type],
                constants.SPIN_BOX_STEP_SIZES[map_type],
            )
            self._noise_map_min_inputs[map_type].value_change_commited.connect(self._update_biome)
            self._noise_map_max_inputs[map_type] = IntSpinBox(
                biome.max_limits[map_type],
                self._model.noise_map_min_limits[map_type],
                self._model.noise_map_max_limits[map_type],
                constants.SPIN_BOX_STEP_SIZES[map_type],
            )
            self._noise_map_max_inputs[map_type].value_change_commited.connect(self._update_biome)

        self._color_chooser_button = qtw.QPushButton()
        self._color_chooser_button.setStyleSheet(
            f"background-color: rgb{self._color_rgb}; border-width: 4px; border-color: black; border-style: solid;"
        )
        self._color_chooser_button.clicked.connect(self._choose_color_representation)

        self._load_sample_file_button = qtw.QPushButton("Load Sample (from CSV File)")
        self._load_sample_file_button.clicked.connect(self._load_sample_file)

        # === LEFT SIDE - LAYOUT ===

        container_biome_settings = qtw.QGroupBox("Biome Settings")
        container_biome_settings_layout = qtw.QGridLayout()
        container_biome_settings.setLayout(container_biome_settings_layout)
        container_biome_settings_layout.setColumnStretch(0, 1)
        container_biome_settings_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_biome_settings_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_biome_settings_layout.addWidget(qtw.QLabel("Name"), 0, 0)
        container_biome_settings_layout.addWidget(self._name_input, 0, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Min Altitude"), 1, 0)
        container_biome_settings_layout.addWidget(self._noise_map_min_inputs[MapType.ALTITUDE], 1, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Max Altitude"), 2, 0)
        container_biome_settings_layout.addWidget(self._noise_map_max_inputs[MapType.ALTITUDE], 2, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Min Precipitation"), 3, 0)
        container_biome_settings_layout.addWidget(self._noise_map_min_inputs[MapType.PRECIPITATION], 3, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Max Precipitation"), 4, 0)
        container_biome_settings_layout.addWidget(self._noise_map_max_inputs[MapType.PRECIPITATION], 4, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Min Temperature"), 5, 0)
        container_biome_settings_layout.addWidget(self._noise_map_min_inputs[MapType.TEMPERATURE], 5, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Max Temperature"), 6, 0)
        container_biome_settings_layout.addWidget(self._noise_map_max_inputs[MapType.TEMPERATURE], 6, 2)
        container_biome_settings_layout.addWidget(qtw.QLabel("Color Representation"), 7, 0)
        container_biome_settings_layout.addWidget(self._color_chooser_button, 7, 2)

        container_left = qtw.QWidget()
        container_left.setMaximumWidth(constants.LAYOUT_LEFT_SIDE_MAX_WIDTH)
        container_left_layout = qtw.QVBoxLayout()
        container_left.setLayout(container_left_layout)
        container_left_layout.setSpacing(constants.LAYOUT_LEFT_SIDE_VBOX_SPACING)
        container_left_layout.addWidget(container_biome_settings)
        container_left_layout.addWidget(self._load_sample_file_button)
        container_left_layout.addStretch()

        # === RIGHT SIDE ===

        self._biome_sample_img_label = qtw.QLabel()
        self._biome_sample_img_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)

        container_right = qtw.QWidget()
        container_right_layout = qtw.QVBoxLayout()
        container_right.setLayout(container_right_layout)
        container_right_layout.addWidget(self._biome_sample_img_label)

        # === COMBINE SIDES ===

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(container_left)
        layout.addWidget(container_right)

    def _update_biome(self) -> None:
        """Collects all settings and updates the biome object in the model."""
        self._biome.update(
            self._name_input.text(),
            self._biome_sample_array,
            {map_type: self._noise_map_min_inputs[map_type].value() for map_type in MapType},
            {map_type: self._noise_map_max_inputs[map_type].value() for map_type in MapType},
            self._color_rgb,
        )

    def _choose_color_representation(self) -> None:
        """Opens a QColorDialog, then updates color, button style, and model."""
        # Casting from tuple[int | None, int | None, int | None] to tuple[int, int, int] so Mypy doesn't raise errors
        # (the r, g and b values can never be None).
        self._color_rgb = cast(tuple[int, int, int], qtw.QColorDialog.getColor().getRgb()[:3])
        self._color_chooser_button.setStyleSheet(
            f"background-color: rgb{self._color_rgb}; border-width: 4px; border-color: black; border-style: solid;"
        )
        self._update_biome()

    def _load_sample_file(self) -> None:
        """Loads the biome sample from a .csv file, updates GUI and model."""
        file_path, _ = qtw.QFileDialog.getOpenFileName(self, "Load Sample from...", "", "CSV Files (*.csv)")
        self._biome_sample_array = np.genfromtxt(file_path, delimiter=",", dtype=np.int_)
        self._update_biome()
        self._update_biome_img()

    def _update_biome_img(self) -> None:
        """Renders the biome sample array and updates image shown in the GUI."""
        if self._biome_sample_array is not None:
            tilemap_img = self._tileset_manager.get_tilemap_img(self._biome_sample_array)
            biome_sample_img_pixmap = qtg.QPixmap.fromImage(ImageQt(tilemap_img).copy())
            self._biome_sample_img_label.setPixmap(biome_sample_img_pixmap)
