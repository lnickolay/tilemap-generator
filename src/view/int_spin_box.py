"""Contains a specialized PyQt widget class for integer input."""

from PyQt6 import QtCore as qtc
from PyQt6 import QtWidgets as qtw


class IntSpinBox(qtw.QSpinBox):
    """A QSpinBox that signals only when a committed value is new.

    This class inherits from QSpinBox but adds logic to the 'editingFinished' event. It checks if the new value differs
    from the previously committed value ('_last_value') before emitting the custom 'value_change_commited' signal. This
    ensures that signals are only emitted for meaningful value changes.

    Signals:
        value_change_commited: Emitted when the user finishes editing and the value has changed since the last commit.
    """

    value_change_commited = qtc.pyqtSignal(int)

    # The last successfully committed int value, used to determine if a change is meaningful.
    _last_value: int

    def __init__(self, default_value: int, min_value: int, max_value: int, step_size: int) -> None:
        """Initializes the specialized integer spin box.

        This sets the range, step size, initial value, and connects the PyQt signal 'editingFinished' to capture user
        input commitment.

        Args:
            default_value: The initial value displayed by the spin box.
            min_value: The lowest integer value allowed in the spin box.
            max_value: The highest integer value allowed in the spin box.
            step_size: The amount to increase or decrease the value by when using the up/down arrow buttons.
        """
        super().__init__()

        self._last_value = default_value

        self.setRange(min_value, max_value)
        self.setValue(default_value)
        self.setSingleStep(step_size)
        self.setCorrectionMode(qtw.QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.editingFinished.connect(self.on_editing_finished)

    def on_editing_finished(self) -> None:
        """Handles the completion of editing by the user.

        Compares the current value to the cached '_last_value'. If the values differ, the new value is cached, and the
        'value_change_commited' signal is emitted. This prevents unnecessary signal emissions when the user confirms
        the existing value.
        """
        if self._last_value != self.value():
            self._last_value = self.value()
            self.value_change_commited.emit(self.value())
