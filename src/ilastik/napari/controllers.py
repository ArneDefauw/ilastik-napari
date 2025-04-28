import os
import dask.array as da
import loguru
import numpy as np

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
    """
        The controller that manages the interactions between the logic and UI.

        Attributes
        ----------
        folder_path : str, Default = None
            The path to the output folder

        prefix_name : str, Default = None
            The prefix that is going to be used for all the files

        scale : list[int], Default = None
            The list of scales that the image has if it was pyramidal

        x_offset : int, Default = None
            The x offset of the image if a sub image was passed. This makes sure that the x axis is known when showing it in the viewer

        y_offset : int, Default = None
            The y offset of the image if a sub image was passed. This makes sure that the x axis is known when showing it in the viewer

        overwrite : bool, Default = False
            If true it overwrites the files at the destination.
    """

    def __init__(self):
        """
            Innitializes an ClassificationController
        """
        self.folder_path = None
        self.prefix_name = None

        self.scale = None

        self.x_offset = None
        self.y_offset = None

        self.overwrite = False


    def is_runnable(self)->bool:
        """
            Checks if the workflow is able to be run.

            Return
            ------
            bool :
                returns if the workflow is able to be run
        """
        return bool(self.folder_path) and bool(self.prefix_name)

    def infer_scales(self, layers):
        """
            Checks if the layers is multiscale and infers the scales

            Parameters
            ----------
            layers : list[Layer]
                List of layers that has a multiscale image or not

            Return
            ------
                None
        """
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
        """
        Checks if a layer is pyramidal and returns the top

        Parameters
        ----------
        input: Layer
            A napari image
        """
        if input.multiscale:
            return input.data._data[0]
        else:
            return input.data

class ObjectClassificationController(ClassificationController):
    """
        The Classification Controller that is in charge of managing the interactions of the UI with the rest of the logic.

        Attributes
        ----------
        object_layer_params :
            the parameters that are passed when adding a new layer to the napari viewer
    """
    # The default name of the object data zarr store
    SDATA_NAME = "object_sdata.zarr"
    # The string name of a rectangle in the shapes layer in napari
    RECTANGLE_STRING = 'rectangle'

    def __init__(self):
        """
            Initialize ObjectClassificationController object
        """
        super().__init__()
        self.object_layer_params = dict(name="ilastik-objects", opacity=1, translate=None)

    @thread_worker
    def object_classifier_workflow_thread(
        self,
        mask_layer: Labels,
        selected_images:list[Image],
        annotion_layer: Labels,
        shape_layer: Shapes,
        statistical_functions:list[Statistical_Functions],
        depth:int,
        to_train=True,
    ) -> da.Array:
        """
        The napari thread worker that executes the object classification workflow. It extracts all the needed data to pass to the object classification workflow.

        Parameters
        ----------
        mask_layer : Labels
            The label layer that contains the unique ids of the cells in the image
        selected_images : List[Image]
            An list of image layers that contain the data that you want to pass through to the model
        annotation_layer : Labels
            The label layer that contains the annotations that you want to pass the the model
        shape_layer : Shapes
            The part of the image that you want to train and classify on in the form of a shapes layer. Can only contain one layer and must be an rectangle
        depth : int
            The depth parameter that is needed to pass for the statistical functions. See :func:`~ilastik.napari.object_classification.Object_Classifier.feature_extractor`
        to_train : bool, default=True
            The to_train parameter that is used if you want to train the model first. See :func:`~ilastik.napari.object_classification.Object_Classifier.object_classifier_workflow`

        Return
        ------
        da.Array :
            The result of the classification in the form of an array

        Raise
        -----
        SameLayerException :
            mask_layer and annotation_layer parameters are the same layer.

        TooManyRectangles :
            More than one rectangle has been passed

        BoxOutOfBoundsException :
            The rectangle in the shapes layer is out of bounds
        """
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


        return classifier.object_classifier_workflow(mask, images, annotions, statistical_functions, depth, self.prefix_name, to_train)

    def save_data(self, proba:da.Array)->SpatialData:
        """
            makes a spatial data object of the result and saves it as an .zarr

            Parameters
            ----------
            proba : da.Array
                The result of the classification

            Return
            ------
            SpatialData :
                the spatialdata object that contains the result of the classification

            Raise
            -----
            FileExistsError
                If the file already exists and the overwrite option is false thusly can not save the object at the file destination
        """
        # TODO: A wierd error occures when you try to train the model on a small part of the dataset (passing a rectangle), remove the shapes layer and then try to train on the whole image
        # Can be because of memory problems.
        sdata = SpatialData()

        if os.path.exists(os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}")) and not self.overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")
        proba = proba.rechunk(proba.chunksize)
        sdata["labels"] = Labels2DModel.parse(
                proba,
                dims=("y", "x"),
                scale_factors=self.scale,
                chunks=proba.chunksize,
            )
        sdata.write(
            os.path.join(self.folder_path, f"{self.prefix_name}_{self.SDATA_NAME}"),
            self.overwrite,
        )

        return read_zarr(sdata.path)

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
