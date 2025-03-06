import os
from typing import Any
import dask.array as da
import loguru
import numpy
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)
from spatialdata import read_zarr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from ilastik.napari import filters
from ilastik.napari.filters import FilterSet
from ilastik.napari.gui import CheckboxTableDialog, rc_pairs
from napari import Viewer
from napari.components import LayerList
from napari.layers import Image, Labels, Layer
from ilastik.napari.object_classification import Pixel_Classifier, Object_Classifier

logger = loguru.logger


filter_names = {
    filters.GaussianDask: "Gaussian Smoothing",
    filters.LaplacianOfGaussianDask: "Laplacian of Gaussian",
    filters.GaussianGradientMagnitudeDask: "Gaussian Gradient Magnitude",
    filters.DifferenceOfGaussiansDask: "Difference of Gaussians",
    # filters.StructureTensorEigenvaluesDask: "Structure Tensor Eigenvalues",
    # filters.HessianOfGaussianEigenvaluesDask: "Hessian of Gaussian Eigenvalues",
}
filter_list = (
    filters.GaussianDask,
    filters.LaplacianOfGaussianDask,
    filters.GaussianGradientMagnitudeDask,
    filters.DifferenceOfGaussiansDask,
    # filters.StructureTensorEigenvaluesDask,
    # filters.HessianOfGaussianEigenvaluesDask,
)
scale_list = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)


class LayerModel(QSortFilterProxyModel):
    def __init__(self, layers: LayerList, parent=None):
        super().__init__(parent)
        self.setSourceModel(QStandardItemModel())
        self.napari_layers = layers
        self.update_model()
        self.napari_layers.events.inserted.connect(self.update_model)
        self.napari_layers.events.removed.connect(self.update_model)

    def filterAcceptsRow(self, row: int, parent: QModelIndex) -> bool:
        model = self.sourceModel()
        index = model.index(row, self.filterKeyColumn(), parent)
        layer = model.data(index, Qt.UserRole)
        return self.should_accept_layer(layer)

    def should_accept_layer(self, layer: Layer) -> bool:
        return True

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if role in (Qt.DisplayRole, Qt.DecorationRole, Qt.UserRole):
            return super().data(index, role)
        return None

    def update_model(self):
        model = self.sourceModel()
        model.clear()
        for layer in self.napari_layers:
            item = QStandardItem(layer.name)
            item.setData(layer, Qt.UserRole)
            model.appendRow(item)


class ImageLayerModel(LayerModel):
    def should_accept_layer(self, layer: Layer) -> bool:
        return isinstance(layer, Image) and not isinstance(layer, Labels)


class LabelsLayerModel(LayerModel):
    def should_accept_layer(self, layer: Layer) -> bool:
        return isinstance(layer, Labels)


class PixelClassificationWidget(QWidget):
    SEG_LAYER_PARAMS = dict(name="ilastik-segmentation", opacity=1)
    PROBA_LAYER_PARAMS = dict(name="ilastik-probabilities", opacity=0.75)

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)

        self.folder_path = None
        self.prefix_name = None

        layer_model = napari_viewer.layers

        # Create a QListWidget for images instead of a QComboBox.
        self._image_list = QListWidget(clicked=self._update_widgets)
        self._image_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )  # or MultiSelection
        # Populate the list with image layers
        napari_viewer.layers.events.inserted.connect(self._update_image_list)
        napari_viewer.layers.events.removed.connect(self._update_image_list)

        labels_combo = QComboBox()
        labels_combo.setModel(LabelsLayerModel(layer_model, self))
        labels_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())

        features_state = dict.fromkeys(
            rc_pairs(len(filter_list), len(scale_list)), True
        )
        for s in range(1, len(filter_list)):
            del features_state[s, 0]
        features_dialog = CheckboxTableDialog(
            self,
            rows=list(map(filter_names.__getitem__, filter_list)),
            cols=list(map(str, scale_list)),
            state=features_state,
        )
        features_dialog.setWindowTitle("Select Features")

        # FIXME: Find a reliable way to fit dialog's size to it's contents.
        features_dialog.setMinimumSize(500, 200)

        features_button = QPushButton("&Features")
        features_button.clicked.connect(features_dialog.open)

        output_type_group = QGroupBox("Output Type")
        segmentation_button = QCheckBox("Segmentation", clicked=self._update_widgets)
        probabilities_button = QCheckBox("Probabilities", clicked=self._update_widgets)
        segmentation_button.setChecked(True)
        output_type_layout = QVBoxLayout()
        output_type_layout.addWidget(segmentation_button)
        output_type_layout.addWidget(probabilities_button)
        output_type_group.setLayout(output_type_layout)

        self.train_checkbox = QCheckBox("Train on data")
        self.train_checkbox.setChecked(True)

        run_button = QPushButton("&Run")
        run_button.setEnabled(False)
        run_button.clicked.connect(self._on_run_clicked)

        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        output_file_group = QGroupBox("Output folder")

        folder_button = QPushButton("select folder")
        folder_button.clicked.connect(self._select_folder)

        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        self.folder_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.prefix_button = QPushButton("confirm prefix")
        self.prefix_button.clicked.connect(self._select_prefix)

        self.prefix_line_edit = QLineEdit(self.prefix_name)

        self.overwrite = QCheckBox("overwrite")
        self.overwrite.setChecked(False)

        output_file_layout = QVBoxLayout()
        output_file_layout.addWidget(folder_button)
        output_file_layout.addWidget(self.folder_label)
        output_file_layout.addWidget(self.prefix_button)
        output_file_layout.addWidget(self.prefix_line_edit)
        output_file_layout.addWidget(self.overwrite)
        output_file_group.setLayout(output_file_layout)

        layout = QFormLayout()
        layout.addRow("&Image:", self._image_list)
        layout.addRow("&Labels:", labels_combo)
        layout.addRow(features_button)
        layout.addRow(output_type_group)
        layout.addRow(output_file_group)
        layout.addRow(self.train_checkbox)
        layout.addRow(run_button)
        layout.addRow(progress_bar)
        self.setLayout(layout)

        self._viewer = napari_viewer
        self._labels_combo = labels_combo
        self._features_dialog = features_dialog
        self._segmentation_button = segmentation_button
        self._probabilities_button = probabilities_button
        self._run_button = run_button
        self._progress_bar = progress_bar
        self._update_widgets()

    def _update_widgets(self):
        # For image list, check that at least one item is selected.
        output_buttons = (self._segmentation_button, self._probabilities_button)
        self._run_button.setEnabled(
            len(self._image_list.selectedItems()) > 0
            and all(c.currentData() for c in (self._labels_combo,))
            and any(b.isChecked() for b in output_buttons)
            and bool(self.folder_path)
            and bool(self.prefix_name)
        )
        self.prefix_button.setEnabled(bool(self.folder_path))
        self.prefix_line_edit.setEnabled(bool(self.folder_path))
        self.folder_label.setText(
            self.folder_path if self.folder_path else "No folder selected"
        )
        self.prefix_line_edit.setText(self.prefix_name)

    def _update_image_list(self, event=None):
        self._image_list.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                item = QListWidgetItem(layer.name)
                item.setData(Qt.UserRole, layer)
                self._image_list.addItem(item)

    def _on_run_clicked(self):
        self._set_enabled(False)

        selected_images = [
            item.data(Qt.UserRole) for item in self._image_list.selectedItems()
        ]

        if not selected_images:
            # Show a warning message and re-enable the UI
            QMessageBox.warning(self, "No Images Selected", "Please select images.")
            self._set_enabled(True)
            return

        labels_layer: Labels = self._labels_combo.currentData()

        features = FilterSet(
            filters=tuple(
                filter_list[row](scale_list[col])
                for row, col in sorted(self._features_dialog.selected)
            )
        )
        classifier = Pixel_Classifier(
            output_folder=self.folder_path,
        )

        image_data = [_item.data for _item in selected_images]
        image = da.concatenate(image_data, axis=0)

        self._labels_dtype = labels_layer.data.dtype
        self._unique_labels = numpy.unique(labels_layer.data)
        self._unique_labels = self._unique_labels[self._unique_labels != 0]

        worker = classifier._workflow(
            image,  # (c,y,x)
            labels_layer.data,  # only support labels layer with one channel dimension
            features,
            self.prefix_name,
            self.train_checkbox.isChecked(),
            self.overwrite.isChecked(),
        )

        worker.finished.connect(lambda: self._set_enabled(True))
        worker.returned.connect(self._update_output_layers)
        worker.start()

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path is not None:
            self.folder_path = folder_path

        self.prefix_name = ""

        self._update_widgets()

    def _select_prefix(self):
        self.prefix_name = self.prefix_line_edit.text()

        self._update_widgets()

    def _set_enabled(self, value):
        self._run_button.setEnabled(value)
        self._progress_bar.setVisible(not value)

    def _update_output_layers(self, proba):
        # TODO: make this pyramid, and add it as such to the napari viewer
        # maybe we should write to intermediate zarr store if arrays would become very large
        proba = proba.astype(numpy.float16).persist()

        # save results in spatialdata object.
        sdata = SpatialData()

        if self._segmentation_button.isChecked():
            # TODO: add code to write to multiscale
            labels = da.argmax(proba, axis=-1)
            # map to original labels
            labels = da.take(self._unique_labels, labels)
            labels = labels.astype(self._labels_dtype)
            sdata["labels"] = Labels2DModel.parse(
                labels,
                dims=("y", "x"),
            )

        if self._probabilities_button.isChecked():
            proba = da.max(proba, axis=-1)
            sdata["proba"] = Image2DModel.parse(
                proba[None, ...],
                dims=("c", "y", "x"),
            )

        sdata.write(
            os.path.join(self.folder_path, f"{self.prefix_name}_sdata.zarr"),
            overwrite=self.overwrite.isChecked(),
        )

        sdata = read_zarr(sdata.path)

        if self._segmentation_button.isChecked():
            self._update_seg_layer(sdata["labels"].data)
        if self._probabilities_button.isChecked():
            self._update_proba_layer(sdata["proba"].data.squeeze(0))

    def _update_seg_layer(self, data):
        try:
            layer = self._viewer.layers[self.SEG_LAYER_PARAMS["name"]]
            layer.data = data
        except KeyError:
            layer = self._viewer.add_labels(data, **self.SEG_LAYER_PARAMS)
            layer.color_mode = "AUTO"
            layer.editable = False

    def _update_proba_layer(self, proba):
        try:
            layer = self._viewer.layers[self.PROBA_LAYER_PARAMS["name"]]
            layer.data = proba
        except KeyError:
            layer = self._viewer.add_image(proba, **self.PROBA_LAYER_PARAMS)

class ObjectClassificationWidget(QWidget):
    OBJECT_LAYER_PARAMS = dict(name="ilastik-objects", opacity=1)

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)

        self.folder_path = None

        self._viewer = napari_viewer
        layer_model = napari_viewer.layers

        self._image_list = QListWidget(clicked=self._update_widgets)
        self._image_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        napari_viewer.layers.events.inserted.connect(self._update_image_list)
        napari_viewer.layers.events.removed.connect(self._update_image_list)

        annotation_combo = QComboBox()
        annotation_combo.setModel(LabelsLayerModel(layer_model, self))
        annotation_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())
        self.annotation_combo = annotation_combo

        mask_combo = QComboBox()
        mask_combo.setModel(LabelsLayerModel(layer_model, self))
        mask_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())
        self.mask_combo = mask_combo

        run_button = QPushButton("&Run")
        run_button.setEnabled(False)
        run_button.clicked.connect(self._run_object_classification)
        self.run_button = run_button

        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        self.progress_bar = progress_bar

        output_file_group = QGroupBox("Output folder")
        folder_button = QPushButton("select folder")
        folder_button.clicked.connect(self._select_folder)

        self.folder_label = QLabel()
        self.folder_label.setWordWrap(True)
        self.folder_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.overwrite = QCheckBox("overwrite")
        self.overwrite.setChecked(False)

        output_file_layout = QVBoxLayout()
        output_file_layout.addWidget(folder_button)
        output_file_layout.addWidget(self.folder_label)
        output_file_layout.addWidget(self.overwrite)
        output_file_group.setLayout(output_file_layout)

        layout = QFormLayout()
        layout.addRow("&Image:", self._image_list)
        layout.addRow("&annotation:", annotation_combo)
        layout.addRow("mask:", mask_combo)
        layout.addRow(output_file_group)
        layout.addRow(run_button)
        layout.addRow(progress_bar)
        self.setLayout(layout)

        self._update_widgets()

    def _update_widgets(self):
        # For image list, check that at least one item is selected.
        self.run_button.setEnabled(
            len(self._image_list.selectedItems()) > 0
            and all(c.currentData() for c in (self.annotation_combo, self.mask_combo))
            and bool(self.folder_path)
        )
        self.folder_label.setText(
            self.folder_path if self.folder_path else "No folder selected"
        )

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path is not None:
            self.folder_path = folder_path

        self._update_widgets()


    def _run_object_classification(self):
        self._set_enabled(False)

        selected_images = [
            item.data(Qt.UserRole).data for item in self._image_list.selectedItems()
        ]

        if not selected_images:
            # Show a warning message and re-enable the UI
            QMessageBox.warning(self, "No Images Selected", "Please select images.")
            self._set_enabled(True)
            return

        annotation_layer: Labels = self.annotation_combo.currentData()
        mask_layer: Labels = self.mask_combo.currentData()


        classifier = Object_Classifier(
            output_folder=self.folder_path,
        )

        self._annotation_dtype = annotation_layer.data.dtype
        self._unique_annotation = numpy.unique(annotation_layer.data)
        self._unique_annotation = self._unique_annotation[self._unique_annotation != 0]

        worker = classifier.object_classifier_workflow(
            mask_layer.data,
            selected_images,
            annotation_layer.data,
        )

        worker.finished.connect(lambda: self._set_enabled(True))
        worker.returned.connect(self._update_output_layers)
        worker.start()

    def _update_image_list(self, event=None):
        self._image_list.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                item = QListWidgetItem(layer.name)
                item.setData(Qt.UserRole, layer)
                self._image_list.addItem(item)
        self._update_widgets

    def _set_enabled(self, value):
        self.run_button.setEnabled(value)
        self.progress_bar.setVisible(not value)


    def _update_output_layers(self, proba):
        sdata = SpatialData()

        sdata["labels"] = Labels2DModel.parse(
                proba,
                dims=("y", "x"),
            )
        sdata.write(
            os.path.join(self.folder_path, "object_sdata.zarr"),
            overwrite=self.overwrite.isChecked(),
        )

        sdata = read_zarr(sdata.path)

        try:
            layer = self._viewer.layers[self.OBJECT_LAYER_PARAMS["name"]]
            layer.data = sdata
        except KeyError:
            layer = self._viewer.add_labels(sdata["predicted_labels"].data.squeeze(0), **self.OBJECT_LAYER_PARAMS)
            layer.color_mode = "AUTO"
            layer.editable = False
class IlastikWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()

        self.viewer = viewer  # Store Napari viewer reference
        self.setLayout(QVBoxLayout())

        # Create the tab widget
        self.tabs = QTabWidget()

        # Create first tab (Example: Image Loader)
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.tab1_layout.addWidget(PixelClassificationWidget(viewer))
        self.tab1.setLayout(self.tab1_layout)
        self.tabs.addTab(self.tab1, "pixel")

        # Create second tab (Example: Image Processing)
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        self.tab2_layout.addWidget(ObjectClassificationWidget(viewer))
        self.tab2.setLayout(self.tab2_layout)
        self.tabs.addTab(self.tab2, "object")

        # Add tabs to the main layout
        self.layout().addWidget(self.tabs)
