"""Contains the widget class for generating/displaying the output tilemap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL.ImageQt import ImageQt
from PyQt6 import QtCore as qtc
from PyQt6 import QtGui as qtg
from PyQt6 import QtWidgets as qtw

import constants
from enums import WFCCellCollapseOrder, WFCMode, WFCSegmentOrder, WFCUpdateMode
from view.int_spin_box import IntSpinBox

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

    from model.base_model import BaseModel
    from model.tileset_manager import TilesetManager
    from model.wfc_manager import WFCManager


class OutputWidget(qtw.QWidget):
    """The widget class for generating/displaying the output tilemap.

    This widget manages the configuration of the WFC algorithm, including mode, segment handling, and collapse order. It
    initiates the generation process and displays the resulting tilemap. It also provides options for saving the tilemap
    data (.csv format) and its visual representation (.png format).
    """

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel
    # The WFC manager responsible for creation, execution, and coordination of WFC worker processes.
    _wfc_manager: WFCManager
    # The tileset manager responsible for tileset lookup and image manipulation.
    _tileset_manager: TilesetManager

    # Selection for the WFC mode (Simple Tiled or Overlapping).
    _wfc_mode_combobox: qtw.QComboBox
    # Input for the dimensions of the square segments to be processed iteratively (in cells).
    _segment_size_input: IntSpinBox
    # Selection for the strategy for selecting the next cell (e.g., lowest entropy).
    _cell_collapse_order_combobox: qtw.QComboBox
    # Selection for the strategy for selecting the next segment to process (e.g., row-by-row).
    _segment_order_combobox: qtw.QComboBox

    # Selection for the granularity of updates sent back to the GUI from the WFC worker processes.
    _tilemap_image_update_mode_combobox: qtw.QComboBox
    # Button to start the generating the output tilemap via WFC.
    _generate_tilemap_button: qtw.QPushButton
    # Button to stop the ongoing generation.
    _abort_tilemap_generation_button: qtw.QPushButton

    # Button to save the raw tilemap data (.csv format).
    _save_tilemap_button: qtw.QPushButton
    # Button to save the tilemap image (.png format).
    _save_tilemap_image_button: qtw.QPushButton

    # Label to display the tilemap image.
    _tilemap_img_label: qtw.QLabel

    def __init__(self, model: BaseModel, wfc_manager: WFCManager, tileset_manager: TilesetManager) -> None:
        """Initializes the widget and sets up the GUI elements and connections.

        Args:
            model: The base model managing tile/tilemap size and noise/biome maps.
            wfc_manager: The WFC manager responsible for creation, execution, and coordination of WFC worker processes.
            tileset_manager: The tileset manager responsible for tileset lookup and image manipulation.
        """
        super().__init__()

        self._model = model
        self._wfc_manager = wfc_manager
        self._tileset_manager = tileset_manager

        self.tilemap = np.full((self._model.tilemap_height, self._model.tilemap_width), -1, dtype=np.int_)

        # === LEFT SIDE - WIDGETS ===

        self._wfc_mode_combobox = qtw.QComboBox()
        self._wfc_mode_combobox.addItems([option.value for option in WFCMode])

        self._segment_size_input = IntSpinBox(
            constants.WFC_SEGMENT_SIZE_DEFAULT,
            constants.WFC_SEGMENT_SIZE_MIN_LIMIT,
            constants.WFC_SEGMENT_SIZE_MAX_LIMIT,
            5,
        )

        self._cell_collapse_order_combobox = qtw.QComboBox()
        self._cell_collapse_order_combobox.addItems([option.value for option in WFCCellCollapseOrder])

        self._segment_order_combobox = qtw.QComboBox()
        self._segment_order_combobox.addItems([option.value for option in WFCSegmentOrder])

        tilemap_image_update_mode_label = qtw.QLabel("Tilemap Image Update Mode")
        tilemap_image_update_mode_label.setWordWrap(True)
        tilemap_image_update_mode_label.setMinimumHeight(32)

        self._tilemap_image_update_mode_combobox = qtw.QComboBox()
        self._tilemap_image_update_mode_combobox.addItems([option.value for option in WFCUpdateMode])
        self._tilemap_image_update_mode_combobox.setCurrentText(WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT.value)

        self._generate_tilemap_button = qtw.QPushButton("Generate Tilemap")
        self._generate_tilemap_button.clicked.connect(self.on_generate_tilemap_button_clicked)

        self._abort_tilemap_generation_button = qtw.QPushButton("Abort")
        self._abort_tilemap_generation_button.setEnabled(False)
        self._abort_tilemap_generation_button.clicked.connect(self.on_abort_tilemap_generation_button_clicked)

        self._save_tilemap_button = qtw.QPushButton("Save Tilemap (to CSV File)")
        self._save_tilemap_button.clicked.connect(self.save_tilemap)

        self._save_tilemap_image_button = qtw.QPushButton("Save Tilemap Image")
        self._save_tilemap_image_button.clicked.connect(self.save_tilemap_image)

        # === LEFT SIDE - LAYOUT ===

        container_wfc_settings = qtw.QGroupBox("WFC Settings")
        container_wfc_settings_layout = qtw.QGridLayout()
        container_wfc_settings.setLayout(container_wfc_settings_layout)
        container_wfc_settings_layout.setColumnStretch(0, 1)
        container_wfc_settings_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_wfc_settings_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_wfc_settings_layout.addWidget(qtw.QLabel("Mode"), 0, 0)
        container_wfc_settings_layout.addWidget(self._wfc_mode_combobox, 0, 2)
        container_wfc_settings_layout.addWidget(qtw.QLabel("Segment Size"), 1, 0)
        container_wfc_settings_layout.addWidget(self._segment_size_input, 1, 2)
        container_wfc_settings_layout.addWidget(qtw.QLabel("Cell Collapse Order"), 2, 0)
        container_wfc_settings_layout.addWidget(self._cell_collapse_order_combobox, 2, 2)
        container_wfc_settings_layout.addWidget(qtw.QLabel("Segment Order"), 3, 0)
        container_wfc_settings_layout.addWidget(self._segment_order_combobox, 3, 2)

        container_tilemap_generation = qtw.QGroupBox("Tilemap Generation")
        container_tilemap_generation_layout = qtw.QGridLayout()
        container_tilemap_generation.setLayout(container_tilemap_generation_layout)
        container_tilemap_generation_layout.setColumnStretch(0, 1)
        container_tilemap_generation_layout.setColumnMinimumWidth(1, constants.LAYOUT_GRID_MIDDLE_COLUMN_MIN_WIDTH)
        container_tilemap_generation_layout.setColumnMinimumWidth(2, constants.LAYOUT_GRID_RIGHT_COLUMN_MIN_WIDTH)
        container_tilemap_generation_layout.addWidget(tilemap_image_update_mode_label, 0, 0)
        container_tilemap_generation_layout.addWidget(self._tilemap_image_update_mode_combobox, 0, 2)
        container_tilemap_generation_layout.addWidget(self._generate_tilemap_button, 1, 0)
        container_tilemap_generation_layout.addWidget(self._abort_tilemap_generation_button, 1, 2)

        container_tilemap_storage = qtw.QGroupBox("Tilemap Storage")
        container_tilemap_storage_layout = qtw.QGridLayout()
        container_tilemap_storage.setLayout(container_tilemap_storage_layout)
        container_tilemap_storage_layout.addWidget(self._save_tilemap_button, 0, 0, 1, -1)
        container_tilemap_storage_layout.addWidget(self._save_tilemap_image_button, 1, 0, 1, -1)

        container_left = qtw.QWidget()
        container_left.setMaximumWidth(constants.LAYOUT_LEFT_SIDE_MAX_WIDTH)
        container_left_layout = qtw.QVBoxLayout()
        container_left.setLayout(container_left_layout)
        container_left_layout.setSpacing(constants.LAYOUT_LEFT_SIDE_VBOX_SPACING)
        container_left_layout.addWidget(container_wfc_settings)
        container_left_layout.addWidget(container_tilemap_generation)
        container_left_layout.addWidget(container_tilemap_storage)
        container_left_layout.addStretch()
        container_left_layout.addStretch()
        container_left_layout.addStretch()

        # === RIGHT SIDE ===

        self._tilemap_img_label = qtw.QLabel()
        self._tilemap_img_label.setAlignment(qtc.Qt.AlignmentFlag.AlignCenter)

        container_right = qtw.QWidget()
        container_right_layout = qtw.QVBoxLayout()
        container_right.setLayout(container_right_layout)
        container_right_layout.addWidget(self._tilemap_img_label)

        # === COMBINE SIDES ===

        layout = qtw.QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(container_left)
        layout.addWidget(container_right)

        self._wfc_manager.tilemap_img_updated.connect(self.on_wfc_manager_tilemap_img_updated)
        self._wfc_manager.finished.connect(self.on_wfc_manager_finished)

    def on_generate_tilemap_button_clicked(self) -> None:
        """Orders the WFC manager to start the tilemap generation process.

        Gathers all relevant WFC settings from the input fields of the output widget and sends them to the WFC manager
        in order to start generating the output tilemap. Disables the 'Generate' button and enables the 'Abort' button.
        """
        self._generate_tilemap_button.setEnabled(False)
        self._abort_tilemap_generation_button.setEnabled(True)

        self.wfc_wait_cond = qtc.QWaitCondition()
        self._wfc_manager.generate_tilemap(
            self.wfc_wait_cond,
            WFCMode(self._wfc_mode_combobox.currentText()),
            2,
            self._segment_size_input.value(),
            WFCCellCollapseOrder(self._cell_collapse_order_combobox.currentText()),
            WFCSegmentOrder(self._segment_order_combobox.currentText()),
            update_mode=WFCUpdateMode(self._tilemap_image_update_mode_combobox.currentText()),
        )

    def on_abort_tilemap_generation_button_clicked(self) -> None:
        """Aborts the currently running WFC tilemap generation process.

        Enables the 'Generate' button and disables the 'Abort' button.
        """
        self._wfc_manager.abort_tilemap_generation()

        self._generate_tilemap_button.setEnabled(True)
        self._abort_tilemap_generation_button.setEnabled(False)

    def on_wfc_manager_tilemap_img_updated(self, tilemap_img: Image.Image) -> None:
        """Receives an updated tilemap image and draws it.

        Draws the updated tilemap image and wakes the update listener thread so that it can continue handling incoming
        updates coming from the WFC worker threads.

        Args:
            tilemap_img: The updated rendered tilemap image.
        """
        self._draw_tilemap_img(tilemap_img)
        self.wfc_wait_cond.wakeAll()

    def on_wfc_manager_finished(self, tilemap: NDArray[np.int_], tilemap_img: Image.Image) -> None:
        """Receives the final tilemap data and image and draws the latter.

        Receives the final tilemap data, draws the final tilemap image and wakes the update listener thread. Enables the
        'Generate' button and disables the 'Abort' button.

        Args:
            tilemap: The final 2D array representing the tile indices.
            tilemap_img: The final rendered tilemap image.
        """
        self.tilemap = tilemap
        self.tilemap_img = tilemap_img
        self._draw_tilemap_img(tilemap_img)

        self.wfc_wait_cond.wakeAll()

        self._generate_tilemap_button.setEnabled(True)
        self._abort_tilemap_generation_button.setEnabled(False)

    def save_tilemap(self) -> None:
        """Opens a file dialog and saves the tilemap data as a .csv file."""
        file_path, _ = qtw.QFileDialog.getSaveFileName(self, "Save Tilemap to...", "tilemap", "CSV Files (*.csv)")
        np.savetxt(file_path, self.tilemap, fmt="%i", delimiter=",")

    def save_tilemap_image(self) -> None:
        """Opens a file dialog and saves the tilemap image as a .png file."""
        # Saving to .jpg instead of .png results in blurry images, so the .jpg option was removed (this might be because
        # the source tileset is also in .png format).
        file_path, _ = qtw.QFileDialog.getSaveFileName(self, "Save Tilemap Image to...", "tilemap", "PNG Files (*.png)")
        self._tileset_manager.save_tilemap_img(self.tilemap_img, file_path)

    def _draw_tilemap_img(self, tilemap_img: Image.Image) -> None:
        """Converts the PIL Image to a QPixmap and displays it in the label."""
        tilemap_img_pixmap = qtg.QPixmap.fromImage(ImageQt(tilemap_img).copy())
        if (
            tilemap_img_pixmap.width() > self._tilemap_img_label.width()
            or tilemap_img_pixmap.height() > self._tilemap_img_label.height()
        ):
            # Subtracting one pixel from the image label height to prevent a bug where the tilemap image grows
            # vertically by one pixel per draw step.
            tilemap_img_pixmap = tilemap_img_pixmap.scaled(
                self._tilemap_img_label.width(),
                self._tilemap_img_label.height() - 1,
                qtc.Qt.AspectRatioMode.KeepAspectRatio,
            )
        self._tilemap_img_label.setPixmap(tilemap_img_pixmap)
