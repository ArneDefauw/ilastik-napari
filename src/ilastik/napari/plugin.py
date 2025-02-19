from typing import Any
import os

import dask.array as da
import joblib
import loguru
import numpy
import sparse
from dask.distributed import Client
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sklearn.ensemble import RandomForestClassifier

from ilastik.napari import filters
from ilastik.napari.classifier import NDSparseClassifier, NDSparseDaskClassifier
from ilastik.napari.filters import FilterSet
from ilastik.napari.gui import CheckboxTableDialog, rc_pairs
from napari import Viewer
from napari.components import LayerList
from napari.layers import Image, Labels, Layer
from napari.qt.threading import thread_worker

logger = loguru.logger


def preprocessing_dask(image, estimators, preprocessing_path=None):
    pass
    # features=FilterSet( filters=[GaussianDask( scale=0.3 ),GaussianDask( scale=0.7 ) ] )
    # estimators = [("features", features)]
    # pipe = Pipeline(estimators)
    # feature_map_lazy = pipe.transform(image)
    # feature_map_lazy.to_zarr( preprocessing_path / "array.zarr" ) # this could be large


def pixel_training_dask(
    image, labels, preprocessing_path=None, model_path=None, **client_kwargs
):
    # WIP
    # load features from the zarr store
    X = da.from_zarr(preprocessing_path / "array.zarr")
    clf = NDSparseDaskClassifier(RandomForestClassifier(n_jobs=-1))
    # add the classifier to the pipe, and then dump it
    client = Client(**client_kwargs)
    logger.log(f"Client dashboard link {client.dashboard_link}")

    with joblib.parallel_backend(
        "dask"
    ):  # note, NDSparseDaskClassifier with dask backend will still load data that was annotated in memory (although not the full dataset, only non-zero labels)
        clf.fit(X, labels)

    if model_path is not None:
        joblib.dump(clf, os.path.join(model_path))


def pixel_classification_dask(
    image: da.Array | None,
    preprocessing_path,
    model_path,
    tmp_path,
    **client_kwargs,
):
    # WIP
    if image is None:
        # case where we train and run inference on same image
        image = da.from_zarr(preprocessing_path / "array.zarr")
    else:
        # load the preprocessing pipe from the path, do the preprocessing on image, and then do the classification
        # this should be used if we have a new image coming in, that we want to preprocesses and classify using pretrained model.
        preprocessing_pipe = joblib.load(preprocessing_path / "pipe.pkl")
        image = preprocessing_pipe.transform(image)
        # image could be large
        image.to_zarr(tmp_path)
        image.from_zarr(tmp_path)
    clf = joblib.load(model_path)
    client = Client(**client_kwargs)

    clf_scatter = client.scatter(
        clf
    )  # scatter the model otherwise issues with large task graph

    def _predict_clf(arr, model):
        arr = model.predict(arr)
        return arr.squeeze(-1)

    # probably need to use map_overlap instead of map_blocks here
    array_result = da.map_blocks(
        _predict_clf,
        image,
        dtype=image.dtype,
        drop_axis=-1,
        chunks=image.chunks[:-1],
        model=clf_scatter,
        # TODO output dtype not correct, need to fix via meta
    )

    results = array_result.compute()
    return results


@thread_worker
def _pixel_classification(image, labels, features):
    feature_map = features.transform(numpy.asarray(image.data))
    sparse_labels = sparse.COO.from_numpy(numpy.asarray(labels.data))
    clf = NDSparseClassifier(RandomForestClassifier())
    clf.fit(feature_map, sparse_labels)
    return numpy.moveaxis(clf.predict_proba(feature_map), -1, 0)


filter_names = {
    filters.Gaussian: "Gaussian Smoothing",
    filters.LaplacianOfGaussian: "Laplacian of Gaussian",
    filters.GaussianGradientMagnitude: "Gaussian Gradient Magnitude",
    filters.DifferenceOfGaussians: "Difference of Gaussians",
    filters.StructureTensorEigenvalues: "Structure Tensor Eigenvalues",
    filters.HessianOfGaussianEigenvalues: "Hessian of Gaussian Eigenvalues",
}
filter_list = (
    filters.Gaussian,
    filters.LaplacianOfGaussian,
    filters.GaussianGradientMagnitude,
    filters.DifferenceOfGaussians,
    filters.StructureTensorEigenvalues,
    filters.HessianOfGaussianEigenvalues,
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

        layer_model = napari_viewer.layers

        image_combo = QComboBox()
        image_combo.setModel(ImageLayerModel(layer_model, self))
        image_combo.currentIndexChanged.connect(lambda _index: self._update_widgets())

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

        run_button = QPushButton("&Run")
        run_button.setEnabled(False)
        run_button.clicked.connect(self._on_run_clicked)

        progress_bar = QProgressBar()
        progress_bar.setVisible(False)
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(0)

        layout = QFormLayout()
        layout.addRow("&Image:", image_combo)
        layout.addRow("&Labels:", labels_combo)
        layout.addRow(features_button)
        layout.addRow(output_type_group)
        layout.addRow(run_button)
        layout.addRow(progress_bar)
        self.setLayout(layout)

        self._viewer = napari_viewer
        self._image_combo = image_combo
        self._labels_combo = labels_combo
        self._features_dialog = features_dialog
        self._segmentation_button = segmentation_button
        self._probabilities_button = probabilities_button
        self._run_button = run_button
        self._progress_bar = progress_bar
        self._update_widgets()

    def _update_widgets(self):
        layer_combos = self._image_combo, self._labels_combo
        output_buttons = self._segmentation_button, self._probabilities_button
        self._run_button.setEnabled(
            all(c.currentData() for c in layer_combos)
            and any(b.isChecked() for b in output_buttons)
        )

    def _on_run_clicked(self):
        self._set_enabled(False)

        image_layer: Image = self._image_combo.currentData()
        labels_layer: Labels = self._labels_combo.currentData()

        features = FilterSet(
            filters=tuple(
                filter_list[row](scale_list[col])
                for row, col in sorted(self._features_dialog.selected)
            )
        )

        worker = _pixel_classification(
            image_layer.data.squeeze(), labels_layer.data.squeeze(), features
        )
        worker.finished.connect(lambda: self._set_enabled(True))
        worker.returned.connect(self._update_output_layers)
        worker.start()

    def _set_enabled(self, value):
        self._run_button.setEnabled(value)
        self._progress_bar.setVisible(not value)

    def _update_output_layers(self, proba):
        if self._segmentation_button.isChecked():
            self._update_seg_layer(proba)
        if self._probabilities_button.isChecked():
            self._update_proba_layer(proba)

    def _update_seg_layer(self, proba):
        data = numpy.argmax(proba, axis=0).astype(numpy.uint8) + 1
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
