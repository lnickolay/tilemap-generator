"""Contains the widget class for general configuration options."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL.ImageQt import ImageQt
from PyQt6 import QtCore as qtc
from PyQt6 import QtGui as qtg
from PyQt6 import QtWidgets as qtw

import constants
from enums import MapType
from view.int_spin_box import IntSpinBox

if TYPE_CHECKING:
    from model.base_model import BaseModel
    from model.tileset_manager import TilesetManager


class GeneralSettingsWidget(qtw.QWidget):
    """The widget class for general configuration options.

    This widget contains general configuration options, including loading the tileset image, setting the dimensions of
    the individual tiles and the output tilemap, toggling the world generation noise maps (Altitude, Precipitation,
    Temperature) and setting their min and max limits.
    """

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel
    # The tileset manager responsible for tileset lookup and image manipulation.
    _tileset_manager: TilesetManager

    # Input for the width of individual tiles in the tileset image (in pixels).
    _tile_width_input: IntSpinBox
    # Input for the height of individual tiles in the tileset image (in pixels).
    _tile_height_input: IntSpinBox
    # Button to trigger the file dialog for loading a new tileset.
    _load_tileset_file_button: qtw.QPushButton

    # Input for the desired width of the output tilemap (in tiles).
    _tilemap_width_input: IntSpinBox
    # Input for the desired height of the resulting tilemap (in tiles).
    _tilemap_height_input: IntSpinBox

    # Checkboxes to enable/disable eachnoise map type.
    _noise_map_checkboxes: dict[MapType, qtw.QCheckBox]
    # Inputs for the minimum limits of each noise map.
    _noise_map_min_inputs: dict[MapType, IntSpinBox]
    # Inputs for the maximum limits of each noise map.
    _noise_map_max_inputs: dict[MapType, IntSpinBox]

    # Label used to display the currently loaded tileset image.
    _tileset_img_label: qtw.QLabel

    def __init__(self, model: BaseModel, tileset_manager: TilesetManager) -> None:
        """Initializes the widget and sets up the GUI elements and connections.

        Args:
            model: The base model managing tile/tilemap size and noise/biome maps.
            tileset_manager: The tileset manager responsible for tileset lookup and image manipulation.
        """
        super().__init__()

        self._model = model
        self._tileset_manager = tileset_manager

        # === LEFT SIDE - WIDGETS ===

        self._tile_width_input = IntSpinBox(
            constants.TILE_SIZE_DEFAULT, constants.TILE_SIZE_MIN_LIMIT, constants.TILE_SIZE_MAX_LIMIT, 1
        )
        self._tile_height_input = IntSpinBox(
            constants.TILE_SIZE_DEFAULT, constants.TILE_SIZE_MIN_LIMIT, constants.TILE_SIZE_MAX_LIMIT, 1
        )

        self._load_tileset_file_button = qtw.QPushButton("Load Tileset (from JPEG/PNG File)")
        self._load_tileset_file_button.clicked.connect(self.load_tileset_file)

        self._tilemap_width_input = IntSpinBox(
            constants.TILEMAP_SIZE_DEFAULT, constants.TILEMAP_SIZE_MIN_LIMIT, constants.TILEMAP_SIZE_MAX_LIMIT, 10
        )
        self._tilemap_width_input.value_change_commited.connect(self.on_tilemap_size_input_changed)
        self._tilemap_height_input = IntSpinBox(
            constants.TILEMAP_SIZE_DEFAULT, constants.TILEMAP_SIZE_MIN_LIMIT, constants.TILEMAP_SIZE_MAX_LIMIT, 10
        )
        self._tilemap_height_input.value_change_commited.connect(self.on_tilemap_size_input_changed)

        self._noise_map_checkboxes = {}
        self._noise_map_min_inputs = {}
        self._noise_map_max_inputs = {}
        for map_type in MapType:
            self._noise_map_checkboxes[map_type] = qtw.QCheckBox()
            self._noise_map_checkboxes[map_type].setChecked(True)
            self._noise_map_min_inputs[map_type] = IntSpinBox(
                constants.NOISE_MAP_MIN_DEFAULTS[map_type],
                constants.NOISE_MAP_MIN_LIMITS[map_type],
                constants.NOISE_MAP_MAX_LIMITS[map_type],
                constants.SPIN_BOX_STEP_SIZES[map_type],
            )
            self._noise_map_max_inputs[map_type] = IntSpinBox(
                constants.NOISE_MAP_MAX_DEFAULTS[map_type],
                constants.NOISE_MAP_MIN_LIMITS[map_type],
                constants.NOISE_MAP_MAX_LIMITS[map_type],
                constants.SPIN_BOX_STEP_SIZES[map_type],
            )
        # Can't do the following in the loop above because using the map_type variable on the right side of the signal
        # connection doesn't work, because when the signal is emitted at a later point in time, the loop is already
        # over and map_type is always equal to MapType.TEMPERATURE.
        self._noise_map_checkboxes[MapType.ALTITUDE].toggled.connect(
            lambda checked: self.on_noise_map_checkbox_toggled(MapType.ALTITUDE, checked)
        )
        self._noise_map_checkboxes[MapType.PRECIPITATION].toggled.connect(
            lambda checked: self.on_noise_map_checkbox_toggled(MapType.PRECIPITATION, checked)
        )
        self._noise_map_checkboxes[MapType.TEMPERATURE].toggled.connect(
            lambda checked: self.on_noise_map_checkbox_toggled(MapType.TEMPERATURE, checked)
        )
        self._noise_map_min_inputs[MapType.ALTITUDE].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.ALTITUDE)
        )
        self._noise_map_min_inputs[MapType.PRECIPITATION].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.PRECIPITATION)
        )
        self._noise_map_min_inputs[MapType.TEMPERATURE].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.TEMPERATURE)
        )
        self._noise_map_max_inputs[MapType.ALTITUDE].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.ALTITUDE)
        )
        self._noise_map_max_inputs[MapType.PRECIPITATION].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.PRECIPITATION)
        )
        self._noise_map_max_inputs[MapType.TEMPERATURE].value_change_commited.connect(
            lambda: self.on_noise_map_limits_input_changed(MapType.TEMPERATURE)
        )

        # === LEFT SIDE - LAYOUT ===

        container_tileset_settings = qtw.QGroupBox("Tileset Settings")
        container_tileset_settings_layout = qtw.QGridLayout()
        container_tileset_settings.setLayout(container_tileset_settings_layout)
        container_tileset_settings_layout.setColumnStretch(0, 1)
        container_tileset_settings_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_tileset_settings_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_tileset_settings_layout.addWidget(qtw.QLabel("Tile Width"), 0, 0)
        container_tileset_settings_layout.addWidget(self._tile_width_input, 0, 2)
        container_tileset_settings_layout.addWidget(qtw.QLabel("Tile Height"), 1, 0)
        container_tileset_settings_layout.addWidget(self._tile_height_input, 1, 2)
        container_tileset_settings_layout.addWidget(self._load_tileset_file_button, 2, 0, 1, -1)

        container_output_settings = qtw.QGroupBox("Output Settings")
        container_output_settings_layout = qtw.QGridLayout()
        container_output_settings.setLayout(container_output_settings_layout)
        container_output_settings_layout.setColumnStretch(0, 1)
        container_output_settings_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_output_settings_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_output_settings_layout.addWidget(qtw.QLabel("Tilemap Width"), 0, 0)
        container_output_settings_layout.addWidget(self._tilemap_width_input, 0, 2)
        container_output_settings_layout.addWidget(qtw.QLabel("Tilemap Height"), 1, 0)
        container_output_settings_layout.addWidget(self._tilemap_height_input, 1, 2)

        container_noise_map_settings = qtw.QGroupBox("Noise Map Settings")
        container_noise_map_settings_layout = qtw.QGridLayout()
        container_noise_map_settings.setLayout(container_noise_map_settings_layout)
        container_noise_map_settings_layout.setColumnStretch(0, 1)
        container_noise_map_settings_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_noise_map_settings_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Enable Altitude Map?"), 0, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_checkboxes[MapType.ALTITUDE], 0, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Altitude Limits (in m above sea level)"), 1, 0, 1, -1)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Min Altitude"), 2, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_min_inputs[MapType.ALTITUDE], 2, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Max Altitude"), 3, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_max_inputs[MapType.ALTITUDE], 3, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Enable Precipitation Map?"), 4, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_checkboxes[MapType.PRECIPITATION], 4, 2)
        container_noise_map_settings_layout.addWidget(
            qtw.QLabel("Total Annual Precipitation Limits (in mm)"), 5, 0, 1, -1
        )
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Min Precipitation"), 6, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_min_inputs[MapType.PRECIPITATION], 6, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Max Precipitation"), 7, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_max_inputs[MapType.PRECIPITATION], 7, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Enable Temperature Map?"), 8, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_checkboxes[MapType.TEMPERATURE], 8, 2)
        container_noise_map_settings_layout.addWidget(
            qtw.QLabel("Average Annual Temperature Limits (in °C)"), 9, 0, 1, -1
        )
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Min Temperature"), 10, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_min_inputs[MapType.TEMPERATURE], 10, 2)
        container_noise_map_settings_layout.addWidget(qtw.QLabel("Max Temperature"), 11, 0)
        container_noise_map_settings_layout.addWidget(self._noise_map_max_inputs[MapType.TEMPERATURE], 11, 2)

        container_left = qtw.QWidget()
        container_left.setMaximumWidth(constants.LAYOUT_LEFT_SIDE_MAX_WIDTH)
        container_left_layout = qtw.QVBoxLayout()
        container_left.setLayout(container_left_layout)
        container_left_layout.setSpacing(constants.LAYOUT_LEFT_SIDE_VBOX_SPACING)
        container_left_layout.addWidget(container_tileset_settings)
        container_left_layout.addWidget(container_output_settings)
        container_left_layout.addWidget(container_noise_map_settings)
        container_left_layout.addStretch()

        # === RIGHT SIDE ===

        self._tileset_img_label = qtw.QLabel()
        self._tileset_img_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)

        container_right = qtw.QWidget()
        container_right_layout = qtw.QVBoxLayout()
        container_right.setLayout(container_right_layout)
        container_right_layout.addWidget(self._tileset_img_label)

        # === COMBINE SIDES ===

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(container_left)
        layout.addWidget(container_right)

        tileset_img = self._tileset_manager.get_tileset_img()

        if tileset_img is not None:
            tileset_img_pixmap = qtg.QPixmap.fromImage(ImageQt(tileset_img).copy())
            self._tileset_img_label.setPixmap(tileset_img_pixmap)

    def load_tileset_file(self) -> None:
        """Opens a file dialog for the user to select a new tileset image.

        The selected image (*.jpeg, *.jpg, *.png) is loaded by the tileset manager, and the new tileset image is
        displayed.
        """
        file_path, _ = qtw.QFileDialog.getOpenFileName(
            self, "Load Tileset from...", "", "JPEG/PNG Files (*.jpeg, *.jpg, *.png)"
        )
        tile_size = (int(self._tile_width_input.text()), int(self._tile_height_input.text()))
        self._tileset_manager.set_tileset(file_path, tile_size)
        tileset_img = self._tileset_manager.get_tileset_img()

        if tileset_img is not None:
            tileset_img_pixmap = qtg.QPixmap.fromImage(ImageQt(tileset_img).copy())
            self._tileset_img_label.setPixmap(tileset_img_pixmap)

    def on_tilemap_size_input_changed(self) -> None:
        """Updates the model's tilemap size when the input values change."""
        self._model.set_tilemap_size(self._tilemap_width_input.value(), self._tilemap_height_input.value())

    def on_noise_map_checkbox_toggled(self, map_type: MapType, checked: bool) -> None:
        """Handles the toggling of a noise map checkbox.

        Enables/disables the min/max input fields for the specific noise map type and updates the model.

        Args:
            map_type: The map type being toggled.
            checked: True if the checkbox is checked, False otherwise.
        """
        self._noise_map_min_inputs[map_type].setEnabled(checked)
        self._noise_map_max_inputs[map_type].setEnabled(checked)
        self._model.toggle_noise_map(map_type, checked)

    def on_noise_map_limits_input_changed(self, map_type: MapType) -> None:
        """Updates the min and max limits for a noise map in the model.

        Args:
            map_type: The map type for which the min and max limits have changed.
        """
        self._model.set_noise_map_limits(
            map_type, self._noise_map_min_inputs[map_type].value(), self._noise_map_max_inputs[map_type].value()
        )
