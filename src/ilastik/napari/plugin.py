import os
from typing import Any
import dask.array as da
import xarray as xa
import loguru
import numpy as np
from spatialdata import get_pyramid_levels
from napari.qt.threading import thread_worker
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
from ilastik.napari.filters import FilterSet, EmptyFilterListError
from ilastik.napari.gui import CheckboxTableDialog, ErrorMessageBox, CheckboxDialog, PrefixGroup
from napari import Viewer
from napari.components import LayerList
from napari.layers import Image, Labels, Layer, Shapes
from ilastik.napari.object_classification import Pixel_Classifier, Object_Classifier, InvalidPrefixError, Statistical_Functions, InvalidAnnotationsArray

logger = loguru.logger


filter_names = {
    filters.GaussianDask: "Gaussian Smoothing",
    filters.LaplacianOfGaussianDask: "Laplacian of Gaussian",
    filters.GaussianGradientMagnitudeDask: "Gaussian Gradient Magnitude",
    filters.DifferenceOfGaussiansDask: "Difference of Gaussians",
}
filter_list = (
    filters.GaussianDask,
    filters.LaplacianOfGaussianDask,
    filters.GaussianGradientMagnitudeDask,
    filters.DifferenceOfGaussiansDask,
)
scale_list = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

def infer_scales(layers, old_scale):
    scale = []
    for layer in layers:
        if layer.multiscale:
            if old_scale:
                logger.warning("Multiscaling has been inferred but already has scales. Changing or updating scales")
            else:
                logger.warning("Multiscaling has been infered adding scales")
            shapes = [i[1] for i in layer.data.shapes]
            base = shapes[0]
            scale = []
            for i in shapes[1:]:
                scale.append(base // i)
                base = i

    if len(scale)==0 and not old_scale:
        return None

    return scale

def add_layer(data:xa.DataArray|xa.DataTree, viewer:Viewer, params:dict, type:str):
    to_scale=False
    if isinstance(data, xa.DataTree):
        pyramid = [level.data for level in get_pyramid_levels(data)]
        if len(pyramid) > 1:
            data = pyramid
            to_scale = True
        else:
            data = pyramid[0]

    if type=="labels":
        layer = viewer.add_labels(data, multiscale=to_scale, **params)
        layer.color_mode = "direct"
        layer.editable = False
    elif type=="image":
        layer = viewer.add_image(data, multiscale=to_scale, **params)

def check_and_convert_multilayer(input):
    if input.multiscale:
        return input.data._data[0]
    else:
        return input.data

def set_features(defaults:list[float]=[1.0]):
    result = dict()
    for i in range(len(filter_list)):
        for j in range(len(scale_list)):
            result[(i, j)] = scale_list[j] in defaults

    return result

def thread_handler(exec:Exception):
    logger.error(exec)

    error_dir = {
        InvalidPrefixError: ErrorMessageBox("Invalid prefix."),
        FileExistsError: ErrorMessageBox("File already exists, please change the folder path or check the overwrite option."),
        NotADirectoryError: ErrorMessageBox("The given folder does not exist"),
        EmptyFilterListError: ErrorMessageBox("No filters has been passed"),
        InvalidAnnotationsArray: ErrorMessageBox("less than two annotations have been passed. You must have two or more labels to run.")
    }

    default_error_box = ErrorMessageBox("Something went wrong, check log.")

    error_dir.get(type(exec), default_error_box).exec_()

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

class ShapesLayerModel(LayerModel):
    def should_accept_layer(self, layer: Layer) -> bool:
        return isinstance(layer, Shapes) and not isinstance(layer, Labels)

class ImageViewQListWidget(QListWidget):
    def __init__(self, update_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_widgets = update_function

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self._update_widgets:
            self._update_widgets()

class ListeningQLineEdit(QLineEdit):
    def __init__(self, update_function=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_widgets = update_function

    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        if self._update_widgets and event.key() in {Qt.Key_Enter, Qt.Key_Return}:
            self._update_widgets()

class PixelClassificationWidget(QWidget):
    SEG_LAYER_PARAMS = dict(name="ilastik-segmentation", opacity=1)
    PROBA_LAYER_PARAMS = dict(name="ilastik-probabilities", opacity=0.75)

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)

        self.pixelController = PixelClassificationController()

        layer_model = napari_viewer.layers

        self._image_list = ImageViewQListWidget(self._update_widgets)
        self._image_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        napari_viewer.layers.events.inserted.connect(self._update_image_list)
        napari_viewer.layers.events.removed.connect(self._update_image_list)

        labels_combo = QComboBox()
        labels_combo.setModel(LabelsLayerModel(layer_model, self))
        labels_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())

        features_state = set_features()
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

        self.prefix_group = PrefixGroup(update_function=self._update_widgets)

        self.overwrite = QCheckBox("overwrite")
        self.overwrite.setChecked(False)

        output_file_layout = QVBoxLayout()
        output_file_layout.addWidget(folder_button)
        output_file_layout.addWidget(self.folder_label)
        output_file_layout.addWidget(self.prefix_group)
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
        self.pixelController.prefix_name = self.prefix_group.prefix_name
        output_buttons = (self._segmentation_button, self._probabilities_button)
        self._run_button.setEnabled(
            len(self._image_list.selectedItems()) > 0
            and all(c.currentData() for c in (self._labels_combo,))
            and any(b.isChecked() for b in output_buttons)
            and self.pixelController.is_runnable()
        )
        self.prefix_group.setEnabled(bool(self.pixelController.folder_path))
        self.folder_label.setText(
            self.pixelController.folder_path if self.pixelController.folder_path else "No folder selected"
        )

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

        labels_layer: Labels = self._labels_combo.currentData()

        filters=tuple(
            filter_list[row](scale_list[col])
            for row, col in sorted(self._features_dialog.selected)
        )

        worker = self.pixelController.pixel_classifier_workflow_thread(
            selected_images,  # (c,y,x)
            labels_layer,  # only support labels layer with one channel dimension
            filters,
            self.train_checkbox.isChecked(),
            self.overwrite.isChecked(),
        )

        worker.finished.connect(lambda: self._set_enabled(True))
        worker.returned.connect(self._update_output_layers)
        worker.errored.connect(thread_handler)
        worker.start()

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path is not None:
            self.pixelController.folder_path = folder_path

        self.pixelController.prefix_name = ""

        self._update_widgets()

    def _select_prefix(self):
        self.pixelController.prefix_name = self.prefix_line_edit.text()

        self._update_widgets()

    def _set_enabled(self, value):
        self._run_button.setEnabled(value)
        self._progress_bar.setVisible(not value)
        self._update_widgets()

    def _update_output_layers(self, proba):
        # TODO: make this pyramid, and add it as such to the napari viewer
        # maybe we should write to intermediate zarr store if arrays would become very large
        proba = proba.astype(np.float16).persist()

        # save results in spatialdata object.
        sdata = self.pixelController.save_data(proba, self._segmentation_button.isChecked(), self._probabilities_button.isChecked(), self.overwrite.isChecked())

        if self._segmentation_button.isChecked():
            # self._update_seg_layer([i.data for i in get_pyramid_levels(sdata["labels"])
            add_layer(sdata["labels"], self._viewer, self.SEG_LAYER_PARAMS, "labels")
        if self._probabilities_button.isChecked():
            # self._update_proba_layer([i.data for i in get_pyramid_levels(sdata["proba"])])
            add_layer(proba, self._viewer, self.PROBA_LAYER_PARAMS, "image")

class ObjectClassificationWidget(QWidget):

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)

        self._viewer = napari_viewer
        layer_model = napari_viewer.layers

        self.objectController = ObjectClassificationController()

        self._image_list = ImageViewQListWidget(self._update_widgets)
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

        shape_combo = QComboBox()
        shape_combo.setModel(ShapesLayerModel(layer_model, self))
        shape_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())
        self.shape_combo = shape_combo

        self.stat_func = CheckboxDialog([i for i in Statistical_Functions], True, parent=self)
        stat_button = QPushButton("Statistical Functions")
        stat_button.clicked.connect(self.stat_func.open)

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

        self.prefix_group = PrefixGroup(update_function=self._update_widgets)

        self.overwrite = QCheckBox("overwrite")
        self.overwrite.setChecked(False)

        output_file_layout = QVBoxLayout()
        output_file_layout.addWidget(folder_button)
        output_file_layout.addWidget(self.folder_label)
        output_file_layout.addWidget(self.prefix_group)
        output_file_layout.addWidget(self.overwrite)
        output_file_group.setLayout(output_file_layout)

        layout = QFormLayout()
        layout.addRow("&Image:", self._image_list)
        layout.addRow("&annotation:", annotation_combo)
        layout.addRow("mask:", mask_combo)
        layout.addRow("shape:", shape_combo)
        layout.addRow(stat_button)
        layout.addRow(output_file_group)
        layout.addRow(run_button)
        layout.addRow(progress_bar)
        self.setLayout(layout)

        self._update_widgets()

    def _update_widgets(self):
        # For image list, check that at least one item is selected.
        self.objectController.prefix_name = self.prefix_group.prefix_name
        self.run_button.setEnabled(
            len(self._image_list.selectedItems()) > 0
            and all(c.currentData() for c in (self.annotation_combo, self.mask_combo))
            and self.objectController.is_runnable()
        )
        self.folder_label.setText(
            self.objectController.folder_path if self.objectController.folder_path else "No folder selected"
        )
        self.prefix_group.setEnabled(bool(self.objectController.folder_path))

    def _select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder_path is not None:
            self.objectController.folder_path = folder_path

        self._update_widgets()


    def _run_object_classification(self):
        self._set_enabled(False)

        selected_images = [
            item.data(Qt.UserRole) for item in self._image_list.selectedItems()
        ]
        annotation_layer: Labels = self.annotation_combo.currentData()
        mask_layer: Labels = self.mask_combo.currentData()
        shape_layer: Labels = self.shape_combo.currentData()


        worker = self.objectController.object_classifier_workflow_thread(
            mask_layer,
            selected_images,
            annotation_layer,
            shape_layer,
            self.stat_func.get_stat_functions(),
        )

        worker.finished.connect(lambda: self._set_enabled(True))
        worker.returned.connect(self._update_output_layers)
        worker.errored.connect(thread_handler)
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

        self._update_widgets()


    def _update_output_layers(self, proba):

        sdata = self.objectController.save_data(proba, self.overwrite.isChecked())

        add_layer(sdata['labels'], self._viewer, self.objectController.object_layer_params, "labels")

class IlastikWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()

        self.viewer = viewer
        self.setLayout(QVBoxLayout())


        self.tabs = QTabWidget()

        # Add pixel classification widget
        self.pixelClassifier = PixelClassificationWidget(viewer)
        self.tab1 = QWidget()
        self.tab1_layout = QVBoxLayout()
        self.tab1_layout.addWidget(self.pixelClassifier)
        self.tab1.setLayout(self.tab1_layout)
        self.tabs.addTab(self.tab1, "pixel")

        #  Add Object classification widget
        self.objectClassifier = ObjectClassificationWidget(viewer)
        self.tab2 = QWidget()
        self.tab2_layout = QVBoxLayout()
        self.tab2_layout.addWidget(self.objectClassifier)
        self.tab2.setLayout(self.tab2_layout)
        self.tabs.addTab(self.tab2, "object")

        self.layout().addWidget(self.tabs)

class ClassificationController:

    def __init__(self):
        self.folder_path = None
        self.prefix_name = None

        self.scale = None

        self.x_offset = None
        self.y_offset = None


    def is_runnable(self)->bool:
        return bool(self.folder_path) and bool(self.prefix_name)

class ObjectClassificationController(ClassificationController):
    SDATA_NAME = "object_sdata.zarr"


    def __init__(self):
        super().__init__()
        self.object_layer_params = dict(name="ilastik-objects", opacity=1)

    @thread_worker
    def object_classifier_workflow_thread(
        self,
        mask_layer: Labels,
        selected_images:list[Image],
        annotion_layer: Labels,
        shape_layer: Shapes,
        statistical_functions:list[Statistical_Functions]
    ) -> da.Array:
        classifier = Object_Classifier(output_folder=self.folder_path)

        mask = check_and_convert_multilayer(mask_layer)

        self.scale = infer_scales(selected_images, self.scale)

        images = [check_and_convert_multilayer(i) for i in selected_images]
        image=da.concatenate(images)

        annotions = annotion_layer.data

        print(image)
        print(annotions.shape)

        if shape_layer:

            rectangle_index = [l for s, l in zip(shape_layer.shape_type, shape_layer.data) if s == 'rectangle'][0]

            array = rectangle_index.astype(int)
            a, b = array[0,-2:]
            c, d, = array[2,-2:]

            self.object_layer_params["translate"] = [a, b]

            image = image[...,a:c,b:d]
            annotions = annotions[...,a:c,b:d]
            mask = mask[...,a:c,b:d]

            print(image)
            print(annotions.shape)


        return classifier.object_classifier_workflow(mask, image, annotions, statistical_functions, self.prefix_name)

    def save_data(self, proba:da.Array, overwrite=False)->SpatialData:
        sdata = SpatialData()

        if os.path.exists(os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}")) and not overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")

        sdata["labels"] = Labels2DModel.parse(
                proba,
                dims=("y", "x"),
                scale_factors=self.scale,
                chunks=proba.chunksize,
            )
        sdata.write(
            os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}"),
            overwrite,
        )

        return read_zarr(sdata.path)

class PixelClassificationController(ClassificationController):
    SDATA_NAME = "pixel_sdata.zarr"

    def __init__(self):
        super().__init__()

    @thread_worker
    def pixel_classifier_workflow_thread(
        self,
        selected_images:list[Image],
        labels_layer:Labels,
        filters:tuple,
        to_train:bool,
        overwrite:bool
    ):

        self.scale = infer_scales(selected_images, self.scale)

        image_data = [check_and_convert_multilayer(_item) for _item in selected_images]
        image = da.concatenate(image_data, axis=0)

        self._labels_dtype = labels_layer.data.dtype
        self._unique_labels = np.unique(labels_layer.data)
        self._unique_labels = self._unique_labels[self._unique_labels != 0]

        classifier = Pixel_Classifier(output_folder=self.folder_path)

        features = FilterSet(filters=filters)

        return classifier._workflow(
            image,
            labels_layer.data,
            features,
            self.prefix_name,
            to_train,
            overwrite,
        )

    def save_data(self, proba:da.Array, is_segmentation, is_probabilities, overwrite=False)->SpatialData:
        sdata = SpatialData()

        if os.path.exists(os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}")) and not overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")

        if is_segmentation:
            # TODO: add code to write to multiscale
            labels = da.argmax(proba, axis=-1)
            # map to original labels
            labels = da.take(self._unique_labels, labels)
            labels = labels.astype(self._labels_dtype)
            sdata["labels"] = Labels2DModel.parse(
                labels,
                dims=("y", "x"),
                scale_factors=self.scale,
            )

        if is_probabilities:
            proba = da.max(proba, axis=-1)
            sdata["proba"] = Image2DModel.parse(
                proba[None, ...],
                dims=("c", "y", "x"),
                scale_factors=self.scale,
            )

        sdata.write(
            os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}"),
            overwrite=overwrite,
        )

        return read_zarr(sdata.path)
