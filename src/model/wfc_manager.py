"""Contains the class that generates and manages the WFC worker processes."""

from __future__ import annotations

from multiprocessing import Event, Queue
from typing import Any, TYPE_CHECKING

import numpy as np
import psutil
from PyQt6 import QtCore as qtc

from enums import WFCMode, WFCUpdateMode
from model.pattern_data import PatternDataOverlapping, PatternDataSimpleTiled
from model.wfc import WFC
from model.wfc_update_listener import WFCUpdateListener

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventType

    from numpy.typing import NDArray
    from PIL import Image

    from enums import WFCCellCollapseOrder, WFCSegmentOrder
    from model.base_model import BaseModel
    from model.pattern_data import PatternData
    from model.tileset_manager import TilesetManager


class WFCManager(qtc.QObject):
    """Manages creation, execution, and coordination of WFC worker processes.

    The manager distributes the generation task by creating a WFC worker process for each biome. It coordinates the flow
    of updates from the WFC update listener back to the GUI, manages concurrency, and handles abortion.

    Signals:
        tilemap_img_updated: Emitted when the tilemap image has been updated (e.g., cell collapsed, segment finished).
        finished: Emitted when all WFC worker processes have successfully completed their assigned biomes.
        main_app_about_to_quit: Emitted when the main application starts its shutdown phase.
    """

    tilemap_img_updated = qtc.pyqtSignal(object)
    finished = qtc.pyqtSignal(np.ndarray, object)
    main_app_about_to_quit = qtc.pyqtSignal()

    # The base model managing tile/tilemap size and noise/biome maps.
    _model: BaseModel
    # The tileset manager responsible for tileset lookup and image manipulation.
    _tileset_manager: TilesetManager

    # Current count of WFC worker processes running concurrently.
    _active_worker_count: int
    # Maximum number of concurrent workers (based on physical CPU count).
    _max_active_worker_count: int
    # List of WFC worker processes awaiting execution.
    _workers_waiting_for_execution: list[WFC]

    # Event used to signal all workers to stop immediately.
    _abort_event: EventType

    # The wait condition used to synchronize the update listener with GUI rendering.
    _wait_cond: qtc.QWaitCondition
    # Defines how often the WFC worker processes send updates.
    _update_mode: WFCUpdateMode

    # The currently assembled tilemap (combining results from all biomes).
    _tilemap: NDArray[np.int_]
    # Total number of cells in the entire tilemap.
    _total_cells: int
    # Count of cells that have been successfully collapsed so far.
    _collapsed_cells: int

    # The PIL image representation of the current tilemap state.
    _tilemap_img: Image.Image

    def __init__(self, model: BaseModel, tileset_manager: TilesetManager) -> None:
        """Initializes the WFC Manager.

        Args:
            model: The base model managing tile/tilemap size and noise/biome maps.
            tileset_manager: The tileset manager responsible for tileset lookup and image manipulation.
        """
        super().__init__()

        self._model = model
        self._tileset_manager = tileset_manager

        self._max_active_worker_count = psutil.cpu_count(logical=False)

        self._abort_event = Event()

    def generate_tilemap(
        self,
        wait_cond: qtc.QWaitCondition,
        mode: WFCMode,
        pattern_size: int,
        segment_size: int,
        cell_collapse_order: WFCCellCollapseOrder,
        segment_order: WFCSegmentOrder,
        update_mode: WFCUpdateMode = WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT,
    ) -> None:
        """Initiates the WFC tilemap generation process.

        This method sets up the initial state, starts the update listener, creates the WFC worker processes (one for
        each biome), and starts the first batch of workers up to the maximum active worker count.

        Args:
            wait_cond: The wait condition used by the output widget to signal the listener to proceed.
            mode: The WFC variant to use (Simple Tiled or Overlapping).
            pattern_size: The size of the NxN patterns to extract (only used in Overlapping mode).
            segment_size: The side length of the square segments for segmented WFC.
            cell_collapse_order: Defines the order cells are selected for collapse (e.g., lowest entropy first).
            segment_order: Defines the order segments are processed in (e.g., row-by-row).
            update_mode: Defines the frequency of updates sent back to the GUI. Defaults to
                WFCUpdateMode.ON_FINISHED_BIOME_SEGMENT.
        """
        self._wait_cond = wait_cond
        self._update_mode = update_mode

        # Reset the abort state for a new generation run.
        self._abort_event.clear()

        # Initialize the global tilemap grid with an uncollapsed state (pattern index = -1).
        self._tilemap = np.full((self._model.tilemap_height, self._model.tilemap_width), -1, dtype=np.int_)
        self._total_cells = self._tilemap.shape[0] * self._tilemap.shape[1]
        self._collapsed_cells = 0

        # Initialize and emit the blank/initial tilemap image.
        self._tilemap_img = self._tileset_manager.get_initial_tilemap_img(
            (self._model.tilemap_height, self._model.tilemap_width)
        )
        self.tilemap_img_updated.emit(self._tilemap_img)

        update_listener_thread = qtc.QThread(self)
        update_queue: Queue[Any] = Queue()
        update_listener = WFCUpdateListener(update_queue, self._wait_cond, self._abort_event)

        # Move listener object to the QThread and set up execution and cleanup logic.
        update_listener.moveToThread(update_listener_thread)
        update_listener_thread.started.connect(update_listener.run)
        update_listener.finished.connect(update_listener_thread.quit)
        update_listener.finished.connect(update_listener.deleteLater)
        update_listener_thread.finished.connect(update_listener_thread.deleteLater)

        # Connect listener signals to manager slots to handle incoming data.
        update_listener.output_cell_collapsed.connect(self.on_wfc_output_cell_collapsed)
        update_listener.segment_finished.connect(self.on_segmented_wfc_segment_finished)
        update_listener.finished.connect(self.on_segmented_wfc_finished)

        update_listener_thread.start()

        self._workers_waiting_for_execution = []

        for biome in self._model.biomes.values():
            assert biome.sample_array is not None

            # Create an output mask defining the region the worker for this biome is responsible for.
            output_mask = np.full(self._model.biome_map.shape, False, dtype=bool)
            output_mask[self._model.biome_map == biome.index] = True

            pattern_data: PatternData
            if np.any(output_mask):
                if mode == WFCMode.SIMPLE_TILED:
                    pattern_data = PatternDataSimpleTiled(biome.sample_array)
                elif mode == WFCMode.OVERLAPPING:
                    pattern_data = PatternDataOverlapping(biome.sample_array, pattern_size)

                worker = WFC(
                    biome.index,
                    output_mask,
                    pattern_data,
                    self._model.random_seed,
                    update_queue,
                    self._abort_event,
                    segment_size,
                    cell_collapse_order,
                    segment_order,
                    self._update_mode,
                )

                self._workers_waiting_for_execution.append(worker)

        self._active_worker_count = 0
        # Start only min(max_cpus, number_of_biomes) workers.
        for _ in range(min(self._max_active_worker_count, len(self._workers_waiting_for_execution))):
            self._active_worker_count += 1
            self._workers_waiting_for_execution.pop(0).start()

    def abort_tilemap_generation(self) -> None:
        """Sets the abort event, signaling all WFC workers to terminate."""
        self._abort_event.set()

    def on_main_app_about_to_quit(self) -> None:
        """Handler for the application shutdown signal."""
        self.main_app_about_to_quit.emit()

    def on_wfc_output_cell_collapsed(
        self, collapsed_cell_coords: tuple[int, int], collapsed_cell_tile_index: int
    ) -> None:
        """Handles updates for a single collapsed cell (used in per-cell update mode).

        This updates the internal tilemap array, increments the collapse counter, and updates the tilemap image for
        visualization.

        Args:
            collapsed_cell_coords: (row, col) coords of the collapsed cell.
            collapsed_cell_tile_index: The tile index assigned to that cell.
        """
        self._tilemap[collapsed_cell_coords] = collapsed_cell_tile_index
        self._collapsed_cells += 1

        self._tilemap_img = self._tileset_manager.update_tilemap_img_tile(
            self._tilemap_img, collapsed_cell_tile_index, collapsed_cell_coords
        )
        self.tilemap_img_updated.emit(self._tilemap_img)

    def on_segmented_wfc_finished(self, output_array: NDArray[np.int_], biome_index: int) -> None:
        """Handles updates for when a worker process has finished its biome.

        This integrates the results for the biome into the global tilemap, updates the tilemap image and manages the
        worker queue, starting the next worker if available, or finalizing the application if all workers are done.

        Args:
            output_array: The output tilemap array generated by the worker.
            biome_index: The index of the biome that was just completed.
        """
        self._tilemap[self._model.biome_map == biome_index] = output_array[self._model.biome_map == biome_index]

        if self._update_mode == WFCUpdateMode.ON_FINISHED_BIOME:
            self._tilemap_img = self._tileset_manager.get_tilemap_img(self._tilemap)
            self.tilemap_img_updated.emit(self._tilemap_img)
        else:
            self._wait_cond.wakeAll()

        if self._workers_waiting_for_execution:
            self._workers_waiting_for_execution.pop(0).start()
        else:
            self._active_worker_count -= 1
            if self._active_worker_count == 0:
                # All workers are finished. Finalize the tilemap image and emit the completion signal.
                self._tilemap_img = self._tileset_manager.get_tilemap_img(self._tilemap)
                self.finished.emit(self._tilemap, self._tilemap_img)

    def on_segmented_wfc_segment_finished(
        self, tilemap_segment: NDArray[np.int_], segment_offset: tuple[int, int], biome_index: int
    ) -> None:
        """Handles updates for when a worker process has finished a segment.

        This integrates the segment into the global tilemap and updates the tilemap image.

        Args:
            tilemap_segment: The successfully collapsed segment array.
            segment_offset: The (row, col) cell coordinates of the top-left corner of the segment in the global tilemap.
            biome_index: The index of the biome to which the segment belongs.
        """
        row_slice = slice(segment_offset[0], segment_offset[0] + tilemap_segment.shape[0])
        col_slice = slice(segment_offset[1], segment_offset[1] + tilemap_segment.shape[1])

        # Create a mask to ensure only cells belonging to the worker's biome are updated.
        mask = self._model.biome_map[row_slice, col_slice] == biome_index
        self._tilemap[row_slice, col_slice][mask] = tilemap_segment[mask]

        self._tilemap_img = self._tileset_manager.update_tilemap_img_segment(
            self._tilemap_img, self._tilemap[row_slice, col_slice], segment_offset
        )
        self.tilemap_img_updated.emit(self._tilemap_img)
