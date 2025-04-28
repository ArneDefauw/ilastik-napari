import os
import dask.array as da
import loguru
import numpy as np
import re
import joblib

from napari.qt.threading import thread_worker
from spatialdata import read_zarr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel

from ilastik.napari import filters
from ilastik.napari.filters import FilterSet
from napari.layers import Image, Labels, Shapes
from ilastik.napari.object_classification import Pixel_Classifier, Object_Classifier, Statistical_Functions
from ilastik.napari.ilastik_exceptions import TooManyRectangles, SameLayerException, BoxOutOfBoundsException

logger = loguru.logger

class ClassificationController:


    def __init__(self, project_path:str):

        if not isinstance(project_path, str):
            raise TypeError("The argument [project_path] has the wrong data type. Please pass a [str] instead.")
        if not os.path.isdir(project_path):
            raise NotADirectoryError(f"The given path: {project_path} for argument [project_path] is not a directory. Please pass a valid one.")

        self.project_path = project_path
        self.scale = None

        self.x_offset = None
        self.y_offset = None

        self.overwrite = False

    def infer_scales(self, layers):

        scale = []
        for layer in layers:
            if layer.multiscale:
                if self.scale:
                    logger.warning("Multiscaling has been inferred but already has scales. Changing or updating scales")
                else:
                    logger.warning("Multiscaling has been infered adding scales")
                shapes = [i[1] for i in layer.data.shapes]
                base = shapes[0]
                scale = []
                for i in shapes[1:]:
                    scale.append(base // i)
                    base = i

        if len(scale)==0 and not self.scale:
            return None

        self.scale = scale

    @staticmethod
    def check_and_convert_multilayer(input):

        if input.multiscale:
            return input.data._data[0]
        else:
            return input.data

class ObjectClassificationController(ClassificationController):

    # The default name of the object data zarr store
    SDATA_NAME = "object_sdata.zarr"
    # The string name of a rectangle in the shapes layer in napari
    RECTANGLE_STRING = 'rectangle'

    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(self.project_path, Object_Classifier.MODEL_NAME)
        stats_im = self.get_statistical_functions_and_images()
        self.statistical_functions = stats_im[0] if stats_im else [i for i in Statistical_Functions]
        self.object_layer_params = dict(name="ilastik-objects", opacity=1, translate=None)

    @thread_worker
    def object_classifier_workflow_thread(
        self,
        mask_layer: Labels,
        selected_images:list[Image],
        annotion_layer: Labels,
        shape_layer: Shapes,
        depth:int,
        to_train=True,
    ) -> da.Array:

        if mask_layer.name==annotion_layer.name:
            raise SameLayerException("mask layer and annotation layers are the same")

        classifier = Object_Classifier(output_folder=self.folder_path)

        mask = self.check_and_convert_multilayer(mask_layer)

        self.infer_scales(selected_images)

        images = [(i.name, self.check_and_convert_multilayer(i)) for i in selected_images]

        annotions = annotion_layer.data

        if shape_layer and len(shape_layer.shape_type)!=0:

            rectangle_indices = [layer for shape, layer in zip(shape_layer.shape_type, shape_layer.data) if shape == self.RECTANGLE_STRING]

            num_rects = len(rectangle_indices)

            if num_rects!=0:
                if num_rects>1:
                    raise TooManyRectangles("Too many rectangles has been passed in the shapes layer.")
                array = rectangle_indices[0].astype(int)
                a, b = array[0,-2:]
                c, d, = array[2,-2:]

                # check if rectangle is in the image
                if a<0:
                    a = 0

                if b<0:
                    b = 0

                if c<0 or d<0:
                    raise BoxOutOfBoundsException("Invalid Rectangle.")

                width, height = mask.shape

                if c>width:
                    c = width

                if d>height:
                    d = height

                if a>width or b>height:
                    raise BoxOutOfBoundsException("Invalid Rectangle.")

                self.object_layer_params["translate"] = [a, b]

                images = [(i[0], i[1][...,a:c,b:d]) for i in images]
                annotions = annotions[...,a:c,b:d]
                mask = mask[...,a:c,b:d]
            else:
                logger.warning("No rectangles found in shapes layer. Continuing without it")
                self.object_layer_params["translate"] = None

        else:
            self.object_layer_params["translate"] = None


        return classifier.object_classifier_workflow(mask, images, annotions, self.statistical_functions, depth, to_train)

    def save_data(self, proba:da.Array)->SpatialData:

        # TODO: A wierd error occures when you try to train the model on a small part of the dataset (passing a rectangle), remove the shapes layer and then try to train on the whole image
        # Can be because of memory problems.
        sdata = SpatialData()

        if os.path.exists(os.path.join(self.project_path, self.SDATA_NAME)) and not self.overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")
        proba = proba.rechunk(proba.chunksize)
        sdata["labels"] = Labels2DModel.parse(
                proba,
                dims=("y", "x"),
                scale_factors=self.scale,
                chunks=proba.chunksize,
            )
        sdata.write(
            os.path.join(self.project_path, self.SDATA_NAME),
            self.overwrite,
        )

        return read_zarr(sdata.path)

    def get_statistical_functions_and_images(self):
        if not os.path.isfile(self.model_path):
            return None

        model = joblib.load(self.model_path)
        features_names = model.feature_names_in_

        stats = set()
        image_names = set()

        for f in features_names:
            stats.add(Statistical_Functions(re.search(r'^[A-Za-z]+_?[A-Za-z]+', f).group()))

            tup = f.split(" ")

            if len(tup)==2:
                image_names.add(tup[1])

        return stats, image_names

    # def check_extracted_features(self, image_layers:list[Image]):

    #     stat_func = self.get_statistical_functions_and_images()

    #     if stat_func:
    #         stats, image_names = stat_func
    #         layer_image_names = {i.name for i in image_layers}

    #         return image_names == layer_image_names and stats == self.statistical_functions

    #     return False

    def set_statistical_function(self, statistical_functions):
        self.statistical_functions = statistical_functions

class PixelClassificationController(ClassificationController):

    FILTER_NAMES = {
        filters.GaussianDask: "Gaussian Smoothing",
        filters.LaplacianOfGaussianDask: "Laplacian of Gaussian",
        filters.GaussianGradientMagnitudeDask: "Gaussian Gradient Magnitude",
        filters.DifferenceOfGaussiansDask: "Difference of Gaussians",
    }

    FILTER_LIST = (
        filters.GaussianDask,
        filters.LaplacianOfGaussianDask,
        filters.GaussianGradientMagnitudeDask,
        filters.DifferenceOfGaussiansDask,
    )

    SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

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
    ):

        self.infer_scales(selected_images)

        image_data = [self.check_and_convert_multilayer(_item) for _item in selected_images]
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
            self.overwrite,
        )

    def save_data(self, proba:da.Array, is_segmentation:bool, is_probabilities:bool)->SpatialData:
        sdata = SpatialData()

        if os.path.exists(os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}")) and not self.overwrite:
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
            self.overwrite,
        )

        return read_zarr(sdata.path)

    @staticmethod
    def set_features(defaults:list[float]=[1.0]):
        result = dict()
        for i in range(len(PixelClassificationController.FILTER_LIST)):
            for j in range(len(PixelClassificationController.SCALE_LIST)):
                result[(i, j)] = PixelClassificationController.SCALE_LIST[j] in defaults

        return result
