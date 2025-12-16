"""Contains the main window widget class for the tilemap generator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6 import QtWidgets as qtw

if TYPE_CHECKING:
    from view.biomes_widget import BiomesWidget
    from view.general_settings_widget import GeneralSettingsWidget
    from view.noise_maps_widget import NoiseMapsWidget
    from view.output_widget import OutputWidget


class MainWindow(qtw.QMainWindow):
    """The main window of the application.

    Inherits from QMainWindow and acts as the top-level container, holding all primary view components (the general
    settings, biomes, noise maps and output widgets) within a central QTabWidget interface.
    """

    def __init__(
        self,
        general_settings_widget: GeneralSettingsWidget,
        biomes_widget: BiomesWidget,
        noise_maps_widget: NoiseMapsWidget,
        output_widget: OutputWidget,
    ) -> None:
        """Initializes the main window and sets up the tabbed interface.

        Configures the size and title of the window and integrates the four primary view components into separate tabs
        within a central tab widget.

        Args:
            general_settings_widget: The widget for setting tile and tilemap size and tileset source image as well as
                toggling which noise maps are used and setting their limits.
            biomes_widget: The widget for managing, adding, and configuring biomes.
            noise_maps_widget: The widget for generating and visualizing the noise maps and the biome color map.
            output_widget: The widget for generating and displaying the output tilemap.
        """
        super().__init__()

        self.resize(1200, 800)
        self.setWindowTitle("Tilemap Generator")

        self.main_tabs = qtw.QTabWidget()
        self.main_tabs.addTab(general_settings_widget, "General Settings")
        self.main_tabs.addTab(biomes_widget, "Biomes")
        self.main_tabs.addTab(noise_maps_widget, "Noise Maps")
        self.main_tabs.addTab(output_widget, "Output")

        self.setCentralWidget(self.main_tabs)
