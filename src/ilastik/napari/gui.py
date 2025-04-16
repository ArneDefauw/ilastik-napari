import collections.abc
from typing import Iterable, Mapping, Sequence, Set, Tuple

from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItem, QStandardItemModel, QIntValidator
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QTableView,
    QVBoxLayout,
    QMessageBox,
    QCheckBox,
    QGroupBox,
    QPushButton,
    QLineEdit,
    QLabel,
    QSizePolicy,
    QFileDialog,
    QListWidget,
    QHBoxLayout
)

def rc_pairs(nrows: int, ncolumns: int) -> Iterable[Tuple[int, int]]:
    """Yield pairs of (row, column) indices."""
    yield from ((r, c) for r in range(nrows) for c in range(ncolumns))


class ModelDict(collections.abc.MutableMapping):
    source: QStandardItemModel

    def __init__(self, rows: Iterable[str], columns: Iterable[str], parent=None):
        self.source = QStandardItemModel(parent)
        self.source.setVerticalHeaderLabels(rows)
        self.source.setHorizontalHeaderLabels(columns)
        for k in rc_pairs(self.source.rowCount(), self.source.columnCount()):
            item = QStandardItem()
            item.setFlags(Qt.NoItemFlags)
            self.source.setItem(*k, item)

    def __getitem__(self, key: Tuple[int, int]) -> bool:
        item = self.source.item(*key)
        if item.isEnabled():
            return item.checkState() == Qt.Checked
        raise KeyError(key)

    def __setitem__(self, key: Tuple[int, int], value: bool) -> None:
        item = self.source.item(*key)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def __delitem__(self, key: Tuple[int, int]) -> None:
        self.source.item(*key).setFlags(Qt.NoItemFlags)

    def __iter__(self, *, row=None, column=None) -> Iterable[Tuple[int, int]]:
        rows = range(self.source.rowCount()) if row is None else (row,)
        columns = range(self.source.columnCount()) if column is None else (column,)
        yield from (
            (r, c) for r in rows for c in columns if self.source.item(r, c).isEnabled()
        )

    def __len__(self) -> int:
        return sum(1 for _k in self)


class CheckboxTableDialog(QDialog):
    selected: Set[Tuple[int, int]]

    def __init__(
        self,
        parent=None,
        *,
        rows: Sequence[str],
        cols: Sequence[str],
        state: Mapping[Tuple[int, int], bool],
    ):
        if not (rows and cols and state):
            raise ValueError("rows, cols, and state should be non-empty")

        super().__init__(parent)

        model = ModelDict(rows, cols, parent)
        model.update(state)

        table = QTableView()
        table.setModel(model.source)
        table.resizeRowsToContents()
        table.resizeColumnsToContents()
        table.clicked.connect(lambda _index: self._update_widgets())

        vheader = table.verticalHeader()
        vheader.setSectionsClickable(True)
        vheader.sectionClicked.connect(lambda index: self._handle_header(row=index))

        hheader = table.horizontalHeader()
        hheader.setSectionsClickable(True)
        hheader.sectionClicked.connect(lambda index: self._handle_header(column=index))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        select_all = buttons.addButton("Select All", QDialogButtonBox.ActionRole)
        select_all.clicked.connect(lambda: self._handle_select(True))

        deselect_all = buttons.addButton("Deselect All", QDialogButtonBox.ActionRole)
        deselect_all.clicked.connect(lambda: self._handle_select(False))

        layout = QVBoxLayout()
        layout.addWidget(table)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._model = model
        self._ok_button = buttons.button(QDialogButtonBox.Ok)
        self.selected = set(k for k, v in self._model.items() if v)
        self._update_widgets()

    def accept(self):
        self.selected = set(k for k, v in self._model.items() if v)
        super().accept()

    def _update_widgets(self):
        self._ok_button.setEnabled(any(self._model.values()))

    def _handle_header(self, row=None, column=None):
        value = not all(
            self._model[k] for k in self._model.__iter__(row=row, column=column)
        )
        for k in self._model.__iter__(row=row, column=column):
            self._model[k] = value
        self._update_widgets()

    def _handle_select(self, value):
        for k in self._model:
            self._model[k] = value
        self._update_widgets()

class StoredQCheckbox(QCheckBox):
    def __init__(self, item, parent=None, update_function=None, **kwargs):
        super().__init__(parent)
        self.setText(item)
        self.item = item
        if update_function:
            self.stateChanged.connect(update_function)

class NumberLineEdit(QLineEdit):
    def __init__(self, default, parent=None):
        super().__init__(parent)
        validator = QIntValidator(0, 2147483647)
        self.setValidator(validator)

        self.setText(default)

    def get_value(self):
        return int(self.text())

class CheckboxDialog(QDialog):
    def __init__(self, item_list, default=True, parent=None):
        super().__init__(parent)

        stat_layout = QVBoxLayout()

        self.stats = []

        for i in item_list:
            checkbox = StoredQCheckbox(i, update_function=self._update_widgets)
            stat_layout.addWidget(checkbox)
            self.stats.append(checkbox)

        self.depth_label = NumberLineEdit("100")
        depth_layout = QHBoxLayout()
        label = QLabel("Depth:")
        depth_layout.addWidget(label)
        depth_layout.addWidget(self.depth_label)

        stat_layout.addLayout(depth_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        select_all = buttons.addButton("Select All", QDialogButtonBox.ActionRole)
        select_all.clicked.connect(lambda: self._handle_select(True))

        deselect_all = buttons.addButton("Deselect All", QDialogButtonBox.ActionRole)
        deselect_all.clicked.connect(lambda: self._handle_select(False))

        layout = QVBoxLayout()
        layout.addLayout(stat_layout)  # Corrected from addWidget to addLayout
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._ok_button = buttons.button(QDialogButtonBox.Ok)

        self._handle_select(default)

    def get_stat_functions(self):
        result = []

        for i in self.stats:
            if i.isChecked():
                result.append(i.item)
        return result

    def get_depth(self):
        return self.depth_label.get_value()

    def _update_widgets(self):
        self._ok_button.setEnabled(any([i.isChecked() for i in self.stats]))

    def _handle_select(self, value):
        state = Qt.Checked if value else Qt.Unchecked
        for k in self.stats:
            k.setCheckState(state)

        self._update_widgets()

class ErrorMessageBox(QMessageBox):

    def __init__(self, message, **kwargs):
        super().__init__(**kwargs)
        self.setIcon(QMessageBox.Critical)
        self.setText("An error occurred:")
        self.setInformativeText(message)
        self.setWindowTitle("Error")

class ListeningQLineEdit(QLineEdit):
    def __init__(self, update_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_widgets = update_function

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        if self._update_widgets and event.key() in {Qt.Key_Enter, Qt.Key_Return}:
            self._update_widgets()

class ImageViewQListWidget(QListWidget):
    def __init__(self, update_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_widgets = update_function

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._update_widgets:
            self._update_widgets()

class FileOutputGroup(QGroupBox):

    def __init__(self, classifier_controler, update_function=None, **kwargs):
        super().__init__(**kwargs)

        self.classifier_controler = classifier_controler
        self._update_other_widgets = update_function

        self.setTitle("Output folder")

        folder_button = QPushButton("select folder")
        folder_button.clicked.connect(self._select_folder)

        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        self.folder_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.prefix_group = QGroupBox("Prefix")

        self.prefix_button = QPushButton("confirm prefix")
        self.prefix_button.clicked.connect(self._select_prefix)

        self.prefix_line_edit = ListeningQLineEdit(update_function=self._select_prefix)
        self.prefix_line_edit.setPlaceholderText("Please set prefix...")

        prefix_layout = QVBoxLayout()
        prefix_layout.addWidget(self.prefix_line_edit)
        prefix_layout.addWidget(self.prefix_button)
        self.prefix_group.setLayout(prefix_layout)

        self.overwrite = QCheckBox("overwrite")
        self.overwrite.setChecked(False)

        output_file_layout = QVBoxLayout()
        output_file_layout.addWidget(folder_button)
        output_file_layout.addWidget(self.folder_label)
        output_file_layout.addWidget(self.prefix_group)
        output_file_layout.addWidget(self.overwrite)
        self.setLayout(output_file_layout)

    def _update_widgets(self):
        self.prefix_group.setEnabled(bool(self.classifier_controler.folder_path))
        self.folder_label.setText(
            self.classifier_controler.folder_path if self.classifier_controler.folder_path else "No folder selected"
        )
        self.prefix_line_edit.setText(self.classifier_controler.prefix_name)
        self.setEnabled(bool(self.classifier_controler.folder_path))

        self.classifier_controler.overwrite = self.overwrite.isChecked()

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path is not None:
            self.classifier_controler.folder_path = folder_path

        self.classifier_controler.prefix_name = None

        self._update_widgets()

    def setEnabled(self, condition):
        self.prefix_button.setEnabled(condition)
        self.prefix_line_edit.setEnabled(condition)

    def _select_prefix(self):
        self.classifier_controler.prefix_name = self.prefix_line_edit.text()
        if self._update_other_widgets:
            self._update_other_widgets()
