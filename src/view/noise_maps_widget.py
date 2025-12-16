"""Contains the widget class for generating/visualizing noise/biome maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PyQt6 import QtWidgets as qtw

import constants
from enums import MapType, NoiseType
from view.float_spin_box import FloatSpinBox

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from numpy.typing import NDArray

    from model.base_model import BaseModel


class NoiseMapsWidget(qtw.QWidget):
    """The widget class for generating/visualizing noise/biome maps.

    This widget contains the settings for each map (Altitude, Precipitation, Temperature) and a Matplotlib canvas to
    display the resulting noise maps and the resulting biome color map (the visual representation of the biome map).
    """

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel

    # A dictionary od widgets for the settings of each individual noise map.
    _noise_settings_widgets: dict[MapType, _NoiseSettingsWidget]

    # Button to trigger biome map generation.
    _generate_biome_map_button: qtw.QPushButton

    # Matplotlib figure containing all plots.
    _maps_figure: Figure
    # Canvas to display the Matplotlib figure.
    _maps_canvas: FigureCanvasQTAgg
    # A dictionary of axes for displaying individual noise maps.
    _noise_maps_ax: dict[MapType, Axes]
    # Axes for displaying the resulting biome color map (the visual representation of the biome map).
    _biome_color_map_ax: Axes

    def __init__(self, model: BaseModel) -> None:
        """Initializes the widget and sets up the GUI elements and connections.

        Args:
            model: The base model managing tile/tilemap size and noise/biome maps.
        """
        super().__init__()

        self._model = model

        # === LEFT SIDE ===

        self._noise_settings_widgets = {map_type: _NoiseSettingsWidget(map_type, self._model) for map_type in MapType}

        self._generate_biome_map_button = qtw.QPushButton("Generate Biome Map")
        self._generate_biome_map_button.clicked.connect(self.on_generate_biome_map_button_clicked)

        container_left = qtw.QWidget()
        container_left.setMaximumWidth(constants.LAYOUT_LEFT_SIDE_MAX_WIDTH)
        container_left_layout = qtw.QVBoxLayout()
        container_left.setLayout(container_left_layout)
        container_left_layout.setSpacing(constants.LAYOUT_LEFT_SIDE_VBOX_SPACING)
        for noise_settings_widget in self._noise_settings_widgets.values():
            container_left_layout.addWidget(noise_settings_widget)
        container_left_layout.addWidget(self._generate_biome_map_button)
        container_left_layout.addStretch()

        # === RIGHT SIDE ===

        self._maps_figure = Figure(figsize=(6, 6))
        self._maps_canvas = FigureCanvasQTAgg(self._maps_figure)

        self._noise_maps_ax = {}
        self._noise_maps_ax[MapType.ALTITUDE] = self._maps_figure.add_subplot(221)
        self._noise_maps_ax[MapType.ALTITUDE].set_title("Altitude Map")
        self._noise_maps_ax[MapType.ALTITUDE].set_axis_off()
        self._noise_maps_ax[MapType.PRECIPITATION] = self._maps_figure.add_subplot(222)
        self._noise_maps_ax[MapType.PRECIPITATION].set_title("Precipitation Map")
        self._noise_maps_ax[MapType.PRECIPITATION].set_axis_off()
        self._noise_maps_ax[MapType.TEMPERATURE] = self._maps_figure.add_subplot(223)
        self._noise_maps_ax[MapType.TEMPERATURE].set_title("Temperature Map")
        self._noise_maps_ax[MapType.TEMPERATURE].set_axis_off()

        self._biome_color_map_ax = self._maps_figure.add_subplot(224)
        self._biome_color_map_ax.set_title("Biome Color Map")
        self._biome_color_map_ax.set_axis_off()

        container_right = qtw.QWidget()
        container_right_layout = qtw.QVBoxLayout()
        container_right.setLayout(container_right_layout)
        container_right_layout.addWidget(self._maps_canvas)

        # === COMBINE SIDES ===

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(container_left)
        layout.addWidget(container_right)

    def generate_initial_maps(self) -> None:
        """Triggers the initial generation and display of all maps."""
        for map_type in MapType:
            self._noise_settings_widgets[map_type].on_generate_noise_map_button_clicked()
        self.on_generate_biome_map_button_clicked()

    def on_noise_map_updated(self, map_type: MapType, noise_map: NDArray[np.double]) -> None:
        """Displays the newly generated noise map on the corresponding axes.

        Args:
            map_type: The type of noise map that was updated.
            noise_map: The 2D array containing the new noise data.
        """
        if not np.all(noise_map == 0.0):
            self._noise_maps_ax[map_type].imshow(noise_map, cmap=constants.NOISE_MAP_COLORMAPS[map_type])
        else:
            # If all array elements are 0.0, the noise map has just been reset, so a completely black image is shown.
            self._noise_maps_ax[map_type].imshow(np.zeros((1, 1, 3)), interpolation="none")
        self._maps_canvas.draw()

    def on_biome_color_map_updated(self, biome_color_map: NDArray[np.int_]) -> None:
        """Slot: Displays the newly generated biome color map on its axes.

        Args:
            biome_color_map: The 2D array representing the biome map colors.
        """
        self._biome_color_map_ax.imshow(biome_color_map)
        self._maps_canvas.draw()

    def on_generate_biome_map_button_clicked(self) -> None:
        """Triggers the model to generate the biome map."""
        self._model.generate_biome_map()

    def on_noise_map_toggled(self, map_type: MapType, flag: bool) -> None:
        """Enables/disables settings widget and display for a noise map type.

        Args:
            map_type: The type of noise map that was toggled.
            flag: True to enable/show, False to disable/hide.
        """
        self._noise_settings_widgets[map_type].setEnabled(flag)
        self._noise_maps_ax[map_type].set_visible(flag)
        self._maps_canvas.draw()


class _NoiseSettingsWidget(qtw.QGroupBox):
    """Inner widget for configuring the settings of a single noise map."""

    # The type of noise map this widget configures.
    _map_type: MapType

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel

    # Selection for which noise type (Perlin or OpenSimplex) to use when generating the noise map.
    _noise_type_combobox: qtw.QComboBox
    # Input for the number of noise octaves to use when generating the noise map.
    _noise_octaves_input: FloatSpinBox
    # Button to regenerate the noise map this widget configures.
    _generate_noise_map_button: qtw.QPushButton

    def __init__(self, map_type: MapType, model: BaseModel) -> None:
        """Initializes the settings widget with a specific noise map type."""
        super().__init__(map_type.value)

        self._map_type = map_type
        self._model = model

        self._noise_type_combobox = qtw.QComboBox()
        self._noise_type_combobox.addItems([option.value for option in NoiseType])

        self._noise_octaves_input = FloatSpinBox(
            constants.NOISE_OCTAVES_DEFAULT, constants.NOISE_OCTAVES_MIN_LIMIT, constants.NOISE_OCTAVES_MAX_LIMIT, 0.1
        )

        self._generate_noise_map_button = qtw.QPushButton(f"Generate {self._map_type.value} Map")
        self._generate_noise_map_button.clicked.connect(self.on_generate_noise_map_button_clicked)

        layout = qtw.QGridLayout()
        self.setLayout(layout)
        layout.setColumnStretch(0, 1)
        layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        layout.addWidget(qtw.QLabel("Noise Type"), 0, 0)
        layout.addWidget(self._noise_type_combobox, 0, 2)
        layout.addWidget(qtw.QLabel("Noise Octaves"), 1, 0)
        layout.addWidget(self._noise_octaves_input, 1, 2)
        layout.addWidget(self._generate_noise_map_button, 3, 0, 1, -1)

    def on_generate_noise_map_button_clicked(self) -> None:
        """Triggers the model to generate the configured noise map."""
        self._model.generate_noise_map(
            self._map_type, self._noise_type_combobox.currentText(), self._noise_octaves_input.value()
        )
