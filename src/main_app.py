"""Serves as the entry point and initializer for the tilemap generator."""

import multiprocessing
import sys

from PyQt6 import QtWidgets as qtw

from constants import EXAMPLE_TILESET_IMG_PATH
from model.base_model import BaseModel
from model.tileset_manager import TilesetManager
from model.wfc_manager import WFCManager
from view.biomes_widget import BiomesWidget
from view.general_settings_widget import GeneralSettingsWidget
from view.main_window import MainWindow
from view.noise_maps_widget import NoiseMapsWidget
from view.output_widget import OutputWidget


class MainApp(qtw.QApplication):
    """The application initializer and integrator for the tilemap generator.

    Inherits from PyQt's QApplication. It handles the initial setup of the application's model and view components (the
    latter being PyQt Widgets), and the signal/slot connections that define the application's reactivity and data flow.
    """

    # The top-level window of the application, which holds all widgets.
    _main_window: MainWindow

    def __init__(self, argv: list[str]) -> None:
        """Initializes the PyQt application and all application components.

        Ensures that the model and view components are created in the correct order and are linked via signals/slots
        before the application starts.

        Args:
            argv: Command line arguments passed to the application (sys.argv).
        """
        super().__init__(argv)

        model = BaseModel()
        tileset_manager = TilesetManager(EXAMPLE_TILESET_IMG_PATH, (16, 16))
        wfc_manager = WFCManager(model, tileset_manager)

        general_settings_widget = GeneralSettingsWidget(model, tileset_manager)
        biomes_widget = BiomesWidget(model, tileset_manager)
        noise_maps_widget = NoiseMapsWidget(model)
        output_widget = OutputWidget(model, wfc_manager, tileset_manager)

        model.noise_map_toggled.connect(biomes_widget.on_noise_map_toggled)
        model.noise_map_toggled.connect(noise_maps_widget.on_noise_map_toggled)
        model.noise_map_limits_changed.connect(biomes_widget.on_noise_map_limits_changed)
        model.noise_map_updated.connect(noise_maps_widget.on_noise_map_updated)
        model.biome_color_map_updated.connect(noise_maps_widget.on_biome_color_map_updated)

        self.aboutToQuit.connect(wfc_manager.on_main_app_about_to_quit)

        noise_maps_widget.generate_initial_maps()

        self._main_window = MainWindow(general_settings_widget, biomes_widget, noise_maps_widget, output_widget)
        self._main_window.show()


if __name__ == "__main__":
    multiprocessing.freeze_support()

    app = MainApp(sys.argv)
    sys.exit(app.exec())
