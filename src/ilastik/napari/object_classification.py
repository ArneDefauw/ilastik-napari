import dask
import dask.dataframe as dd
import dask.array as da
import joblib
import os
import loguru
import xarray as xa
import numpy as np

from enum import StrEnum
from dask.distributed import Client
from functools import reduce
from harpy.utils._aggregate import RasterAggregator
from sklearn.ensemble import RandomForestClassifier
from ilastik.napari.filters import FilterSet
from sklearn.pipeline import Pipeline
from ilastik.napari.classifier import NDSparseDaskClassifier
from ilastik.napari.utils import get_annotation
from napari.qt.threading import thread_worker
from ilastik.napari.ilastik_exceptions import InvalidAnnotationsArray, InvalidDepth, IncompatibleFeatures, NoModelFound
from typing import List, Tuple, Any

dask.config.set({'dataframe.query-planning': False})

logger = loguru.logger

def list_tuple_splitter(
    x:List[Tuple[Any, Any]]
    )->Tuple[List[Any],List[Any]]:
    """
    takes a list of tuples of length of 2 and splits them up in two lists.

    Parameters
    ----------
    x : List[Tuple[Any, Any]]
        The list of tuples which has two elements.

    Return
    ------
    Tuple[List[Any],List[Any]]:
        A tuple of two lists that contain the first and second elements respectively.

    Note
    ----
    If you pass a list of tuples with more than two items it will only take the first two.
    """
    a = []
    b = []

    for tup in x:
        a.append(tup[0])
        b.append(tup[1])

    return a, b

def check_and_convert_arrays_to_dask(
        argument: Any
    ) -> da.Array:
    """
        Checks the data type of an array and converts it to a dask.array.Array if possible.

        Parameters
        ----------
        argument: Any
            argument that you want to check and convert to a dask array

        Return
        ------
        dask.array.Array :
            a dask array of the input.

        Raise
        -----
        TypeError :
            if the input is a invalid datatype then raise.
    """
    if isinstance(argument, np.ndarray):
        return da.from_array(argument)
    elif isinstance(argument, xa.DataArray):
        return argument.data
    elif not isinstance(argument, da.Array):
        raise TypeError("The argument [argument] has the wrong data type. Please pass a value of the following data types [dask.array.Array, numpy.ndarray, xarray.DataArray].")
    return argument

def check_folder_argument(
        output_folder:Any
    ) -> None:
    """
        Checks if the given parameter is a string and if it is a valid path to a directory. If not the an error is raised

        Parameters
        ----------
        output_folder : Any
            the argument that you want to check if it is a valid path

        Raise
        -----
        TypeError :
            If the argument is not a string

        NotADirectoryError :
            If the argument is not a valid directory
    """
    if not isinstance(output_folder, str):
        raise TypeError("The argument [output_folder] has the wrong data type. Please pass a [str] instead.")
    if not os.path.isdir(output_folder):
        raise NotADirectoryError(f"The given path: {output_folder} for argument [output_folder] is not a directory. Please pass a valid one.")

class Pixel_Classifier:
    """
        The class that manages the pixel classification.

        Attributes
        ----------
        output_folder : str
            Path to the output folder

        Methodes
        --------
        preprocessing(
            image: da.Array,
            estimators:list[tuple[str, FilterSet]],
            prefix: str,
            overwrite: bool = False,
        )
            makes the preprocessing images for the input of the model and saves them.

        pixel_training(self,
            X: da.Array,
            labels: np.ndarray,
            model_path: str,
            **client_kwargs
        )
            trains the model and saves it.

        def pixel_classification(
            image: da.Array,
            clf: NDSparseDaskClassifier,
            predict_proba: bool = True,
            **client_kwargs,
        )
            Classifies the given input with the given model.

        def _workflow(
            images: da.Array | np.ndarray | xa.DataArray,
            labels: xa.DataArray | np.ndarray,
            features:FilterSet,
            prefix: str,
            to_train: bool=True,
            overwrite: bool=False,
        )
            Executes the entier pixel classification workflow.

        def _workflow_thread(
            image: da.Array,
            labels: xa.DataArray | np.ndarray,
            features:FilterSet,
            prefix: str,
            to_train: bool=True,
            overwrite: bool=False,
        )
            Executes the entier pixel classification workflow as an napari worker thread.

        Raises
        ------
        TypeError :
            If the argument is not a string

        NotADirectoryError :
            If the argument is not a valid directory
    """
    #: base name of the saved preprocessings .zarr file
    PREPROCESSED_ARRAY_NAME = "preprocessed_array.zarr"
    #: base name of the saved preprocessing pipline file
    PREPROCESSING_PIPE_NAME = "preprocessing_pipe.pkl"
    #: base name of the saved model file
    MODEL_NAME = "pixel_model.pkl"

    def __init__(
        self,
        output_folder:str,
    ):
        # check arguments
        check_folder_argument(output_folder)
        self.output_folder = output_folder

    def preprocessing(
        self,
        image: da.Array,
        estimators:list[tuple[str, FilterSet]],
        overwrite: bool = False,
    ) -> da.Array:
        """
            Executes the preprocessing of the pixel classifier

            Parameters
            ----------
            image : da.Array
                A dask array of the images that needs to be preprocessed
            estimators : List[Tuple[str, FilterSet]]
                The list of filters that are going to be used on the images
            prefix : str
                The prefix that is going to used when the pipeline and features are going to be saved
            overwrite : bool = False
                If you want to overwrite the original pipeline and features

            Returns
            -------
            da.Array :
                The dask array of the features that has been extraxted from the filters
        """
        pipe = Pipeline(estimators)

        arrays = []
        for i in image:
            arrays.append(pipe.transform(i))

        feature_map_lazy = da.concatenate(arrays, axis=2)
        feature_map_lazy = feature_map_lazy.rechunk(feature_map_lazy.chunksize[:-1] + ((feature_map_lazy.shape[-1]),))

        feature_map_lazy.to_zarr(
            os.path.join(
                self.output_folder, self.PREPROCESSED_ARRAY_NAME
            ),
            overwrite=overwrite,
        )  # this could be large
        joblib.dump(
            pipe, os.path.join(self.output_folder, self.PREPROCESSING_PIPE_NAME)
        )
        return da.from_zarr(
            os.path.join(self.output_folder, self.PREPROCESSED_ARRAY_NAME)
        )

    def pixel_training(
        self,
        X: da.Array,
        labels: np.ndarray,
        model_path: str,
        **client_kwargs
    ) -> None:
        """
            executes the training of the passed model with the given features using dask. This model is saved on the given path.

            Parameters
            ----------
            X : da.Array
                a dask array if the features that is going to be trained on

            labels : np.ndarray
                an array which contains the annotations that is going to classify the image

            model_path : str
                The path to the model to train on.

            client_kwargs : dict
                The arguments that you want to pass to the dask client

            Return
            ------
            None
        """
        # load features from the zarr store
        clf = NDSparseDaskClassifier(RandomForestClassifier(n_jobs=-1))
        # add the classifier to the pipe, and then dump it

        client = Client(**client_kwargs)

        logger.info(f"Client dashboard link {client.dashboard_link}")

        logger.info(X)
        with joblib.parallel_backend(
            "dask"
        ):  # note, NDSparseDaskClassifier with dask backend will still load data that was annotated in memory (although not the full dataset, only non-zero labels)
            clf.fit(X, labels)

        joblib.dump(clf, model_path)

    def pixel_classification(
        self,
        image: da.Array,
        clf: NDSparseDaskClassifier,
        predict_proba: bool = True,
        **client_kwargs,
    ) -> da.Array:
        """
            Makes a classification with the passed image.

            Parameter
            ---------
            image : da.Array
                dask array of the image that you want to classify

            clf : NDSparseDaskClassifier
                The NDSparseDaskClassifier model that you want to classify on.

            predict_proba: bool = True
                Makes it that it returns the probabilities of each class too. Default true.

            Returns
            -------
            da.Array :
                A dask array of the classified image. Contains the probabilities if `predict_proba` was true.
        """

        client = Client(**client_kwargs)

        clf_scatter = client.scatter(
            clf
        )  # scatter the model otherwise issues with large task graph

        def _predict_proba_clf(arr, model:NDSparseDaskClassifier):
            arr = model.predict_proba(arr)
            return arr

        def _predict_clf(arr, model:NDSparseDaskClassifier):
            arr = model.predict(arr)
            return arr.squeeze(-1)

        if not predict_proba:
            array_result = da.map_blocks(
                _predict_clf,
                image,
                dtype=image.dtype,
                drop_axis=-1,
                chunks=image.chunks[:-1],
                model=clf_scatter,
                # TODO output dtype not correct, need to fix via meta
            )
        else:
            try:
                nr_of_labels = len(clf.estimator._classes)
            except AttributeError:
                # run classifier on dummy set to get the number of labels
                nr_of_labels = clf.predict_proba(
                    np.zeros((1, 1, image.shape[-1]))
                ).shape[-1]

            array_result = da.map_blocks(
                _predict_proba_clf,
                image,
                dtype=image.dtype,
                drop_axis=2,
                new_axis=2,
                chunks=image.chunks[:-1]
                + ((nr_of_labels,),),  # how can we guess this dimension
                model=clf_scatter,
                # TODO output dtype not correct, need to fix via meta
            )

        return array_result

    def _workflow(
        self,
        images: da.Array | np.ndarray | xa.DataArray,
        labels: xa.DataArray | np.ndarray,
        features:FilterSet,
        to_train: bool=True,
        overwrite: bool=False,
    ):
        """
            The workflow which executes the entier workflow to classify an image.

            Parameters
            ----------
            images :
                the images that you want to classify in array form

            labels :
                The annotations that classifies the image


        """
        # check arguments if they have the write datatype and converts if possible
        images = check_and_convert_arrays_to_dask(images)

        if isinstance(labels, xa.DataArray):
            labels = labels.values
        elif not isinstance(labels, np.ndarray):
            raise TypeError("The argument [labels] has the wrong data type. Please pass a value of the following data types [numpy.ndarray, xarray.DataArray].")

        check_folder_argument(self.output_folder)

        if len(labels.shape)>2:
            labels = labels.squeeze(0)

        if not isinstance(features, FilterSet):
            raise TypeError("The argument [features] has the wrong data type. Please pass a FilterSet")

        if os.path.exists(os.path.join(self.output_folder, self.PREPROCESSED_ARRAY_NAME)) and not overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")

        # Start of workflow
        estimators = [("features", features)]
        logger.info("PIXEL CLASSIFICATION: starting preprocessing")
        data = self.preprocessing(
            image=images,
            estimators=estimators,
            overwrite=overwrite,
        )

        if to_train:
            logger.info("PIXEL CLASSIFICATION: starting training")
            self.pixel_training(
                X=data,
                labels=labels,
                model_path=os.path.join(
                    self.output_folder, self.MODEL_NAME
                ),
                processes=False,
                n_workers=1,
                threads_per_worker=10,
            )

        model_path = os.path.join(self.output_folder, self.MODEL_NAME)

        if not os.path.isfile(model_path):
            raise NoModelFound("No model is found please make or train on.")
        clf = joblib.load(model_path)

        logger.info("PIXEL CLASSIFICATION: starting classification")
        results = self.pixel_classification(
            image=data,
            clf=clf,
            processes=False,
            n_workers=1,
            threads_per_worker=10,
        )

        return results

    @thread_worker
    def _workflow_thread(
        self,
        image: da.Array,
        labels: xa.DataArray | np.ndarray,
        features:FilterSet,
        to_train: bool=True,
        overwrite: bool=False,
    ):
        return self._workflow(image, labels, features, to_train, overwrite)

class Object_Classifier:
    MODEL_NAME = "object_model.pkl"
    ID_COLUMN_NAME = 'cell_ID'

    def __init__(
        self,
        output_folder:str,
    ):
        # argument checks
        check_folder_argument(output_folder)

        self.output_folder = output_folder

    @staticmethod
    def feature_extractor(
        mask: da.Array,
        images: list[tuple[str, da.Array]],
        stats:tuple["Statistical_Functions"],
        depth:int,
    ) -> dd.DataFrame:
        # feature extraction
        mask = mask[None, ...]

        names, images = list_tuple_splitter(images)

        image = da.concatenate(images)
        image=image[ :, None, ... ]

        if mask.chunksize != image.chunksize[1:]:
            mask = mask.rechunk(image.chunksize[1:])

        features = []

        aggregator=RasterAggregator(mask_dask_array=mask, image_dask_array=image)
        single_stats = Statistical_Functions.get_single_stats(stats)

        if single_stats:
            single_features=aggregator.aggregate_stats(stats_funcs=single_stats)

            for key, feature in zip(single_stats, single_features):
                prefix = key
                feature.columns = [c if c==Object_Classifier.ID_COLUMN_NAME else f"{prefix} {names[c]}" for c in feature.columns]

            features.extend(single_features)

        try:
            if Statistical_Functions.QUANTILES in stats:
                quantiles = aggregator.aggregate_quantiles(depth)

                for i in range(len(quantiles)):
                    quantile = quantiles[i]
                    quantile.columns = [f"{Statistical_Functions.QUANTILES}_{i} {names[c]}" if c!=Object_Classifier.ID_COLUMN_NAME else c for c in quantile.columns]

                features.append(reduce(lambda left, right: dd.merge(left, right, on='cell_ID', how='outer'), quantiles))

            if Statistical_Functions.RADII_AND_AXES_MASK in stats:
                rna = aggregator.aggregate_radii_and_axes(depth)
                rna.columns = [f"{Statistical_Functions.RADII_AND_AXES_MASK}_{c}" if c!=Object_Classifier.ID_COLUMN_NAME else c for c in rna.columns]
                features.append(rna)
        except ValueError as e:
            logger.warning(e)
            raise InvalidDepth("Depth is too big")
        except AssertionError as e:
            logger.warning(e)
            raise InvalidDepth("Depth is too small")

        res = reduce(lambda left, right: dd.merge(left, right, on=Object_Classifier.ID_COLUMN_NAME, how='outer'), features)
        res = res.loc[res[Object_Classifier.ID_COLUMN_NAME]!=0]

        return res

    def object_training(
        self,
        X_train: dd.DataFrame,
        y_train: dd.DataFrame,
    ) -> None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        joblib.dump(clf, os.path.join(self.output_folder, self.MODEL_NAME))

    @staticmethod
    def object_classification(
        X: dd.DataFrame,
        clf,
    ) -> np.ndarray:
        if list(clf.feature_names_in_)!=list(X.columns):
            raise IncompatibleFeatures("Features does not match model")
        return clf.predict(X)

    def object_classifier_workflow(
        self,
        mask: da.Array | np.ndarray | xa.DataArray,
        images: list[tuple[str, da.Array | np.ndarray | xa.DataArray]],
        annotation: da.Array | np.ndarray | xa.DataArray,
        statistical_functions: tuple["Statistical_Functions"],
        depth:int,
        to_train=True,
    ) -> da.Array:

        # check arguments if they have the write datatype and converts if possible
        mask = check_and_convert_arrays_to_dask(mask)

        if len(annotation.shape)>2:
            annotation = annotation.squeeze()

        images = [(i[0], check_and_convert_arrays_to_dask(i[1])) for i in images]

        annotation = check_and_convert_arrays_to_dask(annotation)

        check_folder_argument(self.output_folder)

        if np.unique(annotation.compute()).size<3:
            raise InvalidAnnotationsArray("Annotations must contain more than 3 unique values")

        # start workflow
        logger.info("OBJECT CLASSIFICATION: extracting features")
        features = self.feature_extractor(mask, images, statistical_functions, depth)

        annotated_cells_id, annotation=get_annotation( array_1=annotation, array_2=mask)

        X_train=features[ features[self.ID_COLUMN_NAME].isin( annotated_cells_id )]
        X_train = X_train.drop(self.ID_COLUMN_NAME, axis=1)

        if to_train:
            logger.info("OBJECT CLASSIFICATION: starting training")
            self.object_training(X_train, annotation)

        logger.info("OBJECT CLASSIFICATION: starting classification")
        model_path = os.path.join(self.output_folder, self.MODEL_NAME)

        if not os.path.isfile(model_path):
            raise NoModelFound("No model is found please make or train on.")
        clf:RandomForestClassifier = joblib.load(model_path)
        y_pred_all = self.object_classification(features.drop( [self.ID_COLUMN_NAME], axis=1 ), clf)

        cell_ids=features[self.ID_COLUMN_NAME]

        assert cell_ids.shape == y_pred_all.shape

        max_id = cell_ids.max()

        dtype = np.int8
        if len(np.unique(y_pred_all))>255:
            dtype = np.int16
        lookup = np.zeros(max_id + 1, dtype=dtype)

        lookup[cell_ids] = y_pred_all
        return da.take(lookup, mask)

    @thread_worker
    def object_classifier_workflow_thread(
        self,
        mask: da.Array | np.ndarray | xa.DataArray,
        images: list[da.Array] | list[np.ndarray] | list[xa.DataArray],
        annotation: da.Array | np.ndarray | xa.DataArray,
        statistical_functions: tuple["Statistical_Functions"],
    ) -> da.Array:
        return self.object_classifier_workflow(mask, images, annotation, statistical_functions)


class Statistical_Functions(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    VAR = "var"
    KURTOSIS = "kurtosis"
    SKEW = "skew"
    QUANTILES = "quantiles"
    RADII_AND_AXES_MASK = "axes_mask"

    @staticmethod
    def get_values(args: list[StrEnum]) -> list[str]:
        return [stat.value for stat in args]

    @staticmethod
    def get_single_stats(stats: list[StrEnum]=None) -> list[str]:
        aggregate_stats = [Statistical_Functions.SUM,
                    Statistical_Functions.MEAN,
                    Statistical_Functions.COUNT,
                    Statistical_Functions.VAR,
                    Statistical_Functions.KURTOSIS,
                    Statistical_Functions.SKEW]

        if not stats:
            return aggregate_stats

        result = []
        for stat in stats:
            if stat in aggregate_stats:
                result.append(stat.value)

        return result

    @staticmethod
    def get_depth_stats(stats: list[StrEnum]=None) -> list[str]:
        aggregate_stats = [Statistical_Functions.QUANTILES,
                    Statistical_Functions.RADII_AND_AXES_MASK,]

        if not stats:
            return aggregate_stats

        result = []
        for stat in stats:
            if stat in aggregate_stats:
                result.append(stat.value)

        return result
