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

dask.config.set({'dataframe.query-planning': False})

logger = loguru.logger

def check_and_convert_arrays_to_dask(
        argument
    ) -> da.Array:
    if isinstance(argument, np.ndarray):
        return da.from_array(argument)
    elif isinstance(argument, xa.DataArray):
        return argument.data
    elif not isinstance(argument, da.Array):
        raise TypeError("The argument [argument] has the wrong data type. Please pass a value of the following data types [dask.array.Array, numpy.ndarray, xarray.DataArray].")
    return argument

def check_folder_argument(
        output_folder
    ) -> None:
    if not isinstance(output_folder, str):
        raise TypeError("The argument [output_folder] has the wrong data type. Please pass a [str] instead.")
    if not os.path.isdir(output_folder):
        raise NotADirectoryError(f"The given path: {output_folder} for argument [output_folder] is not a directory. Please pass a valid one.")

class Pixel_Classifier:
    PREPROCESSED_ARRAY_NAME = "preprocessed_array.zarr"
    PREPROCESSING_PIPE_NAME = "preprocessing_pipe.pkl"
    MODEL_NAME = "model.pkl"

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
        prefix: str,  # prefix for preprocessed array
        overwrite: bool = False,
    ) -> da.Array:
        pipe = Pipeline(estimators)

        arrays = []
        for i in image:
            arrays.append(pipe.transform(i))

        feature_map_lazy = da.concatenate(arrays, axis=2)
        feature_map_lazy = feature_map_lazy.rechunk(feature_map_lazy.chunksize[:-1] + ((feature_map_lazy.shape[-1]),))

        feature_map_lazy.to_zarr(
            os.path.join(
                self.output_folder, f"{prefix}_{self.PREPROCESSED_ARRAY_NAME}"
            ),
            overwrite=overwrite,
        )  # this could be large
        joblib.dump(
            pipe, os.path.join(self.output_folder, self.PREPROCESSING_PIPE_NAME)
        )
        return da.from_zarr(
            os.path.join(self.output_folder, f"{prefix}_{self.PREPROCESSED_ARRAY_NAME}")
        )

    def pixel_training(self,
        X: da.Array,
        labels: np.ndarray,
        model_path: str,
        **client_kwargs
    ) -> None:
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
        prefix: str,
        to_train: bool=True,
        overwrite: bool=False,
    ):
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

        if not isinstance(prefix, str):
            raise TypeError("The argument [prefix] has the wrong data type. Please pass a str")

        if prefix.isspace() or prefix=="":
            raise InvalidPrefixError("The argumnt [prefix] is empty or contains only whitespace. Please pass a valid prefix for a file")

        if os.path.exists(os.path.join(self.output_folder, f"{prefix}_{self.PREPROCESSED_ARRAY_NAME}")) and not overwrite:
            raise FileExistsError("File already exists, pass a new file or set overwrite to true")

        # Start of workflow
        estimators = [("features", features)]
        logger.info("PIXEL CLASSIFICATION: starting preprocessing")
        data = self.preprocessing(
            image=images,
            estimators=estimators,
            prefix=prefix,
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
        prefix: str,
        to_train: bool=True,
        overwrite: bool=False,
    ):
        return self._workflow(image, labels, features, prefix, to_train, overwrite)

class Object_Classifier:
    MODEL_NAME = "object_model.pkl"

    def __init__(
        self,
        output_folder:str,
    ):
        # argument checks
        check_folder_argument(output_folder)

        self.output_folder = output_folder

    def feature_extractor(
        self,
        mask: da.Array,
        image: da.Array,
        stats:tuple["Statistical_Functions"],
    ) -> dd.DataFrame:
        # feature extraction
        mask = mask[None, ...]

        if mask.chunksize != image.chunksize[1:]:
            logger.warning("Mask chunks and image chunks are not the same. Changing mask chunks...")
            mask = mask.rechunk(image.chunksize[1:])

        aggregator=RasterAggregator(mask_dask_array=mask, image_dask_array=image)
        single_stats = Statistical_Functions.get_single_stats(stats)
        features=dict(zip(single_stats,aggregator.aggregate_stats(stats_funcs=single_stats)))

        if Statistical_Functions.QUANTILES in stats:
            quantiles = aggregator.aggregate_quantiles(100)

            for i in range(len(quantiles)):
                quantile = quantiles[i]
                quantile.columns = [f"{i}_{c}" if c!='cell_ID' else c for c in quantile.columns]

            features[Statistical_Functions.QUANTILES.value] = reduce(lambda left, right: dd.merge(left, right, on='cell_ID', how='outer'), quantiles)

        if Statistical_Functions.RADII_AND_AXES_MASK in stats:
            features[Statistical_Functions.RADII_AND_AXES_MASK.value] = aggregator.aggregate_radii_and_axes(100)


        for key, feature in features.items():
            prefix = key+"_"
            feature.columns = [f"{prefix}{c}" if c!='cell_ID' else c for c in feature.columns]

        res = reduce(lambda left, right: dd.merge(left, right, on='cell_ID', how='outer'), list(features.values()))
        res = res.loc[res['cell_ID']!=0]

        return res

    def object_training(
        self,
        X_train: dd.DataFrame,
        y_train: dd.DataFrame,
        prefix: str,
    ) -> None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        joblib.dump(clf, os.path.join(self.output_folder, f"{prefix}_{self.MODEL_NAME}"))

    def object_classification(
        self,
        X: dd.DataFrame,
        clf,
    ) -> np.ndarray:
        return clf.predict(X)

    def object_classifier_workflow(
        self,
        mask: da.Array | np.ndarray | xa.DataArray,
        images: da.Array | np.ndarray | xa.DataArray,
        annotation: da.Array | np.ndarray | xa.DataArray,
        statistical_functions: tuple["Statistical_Functions"],
        prefix: str,
    ) -> da.Array:

        # check arguments if they have the write datatype and converts if possible
        mask = check_and_convert_arrays_to_dask(mask)

        if len(annotation.shape)>2:
            annotation = annotation.squeeze()

        images = check_and_convert_arrays_to_dask(images)

        annotation = check_and_convert_arrays_to_dask(annotation)

        check_folder_argument(self.output_folder)

        if not isinstance(prefix, str):
            raise TypeError("The argument [prefix] has the wrong data type. Please pass a str")

        if prefix.isspace() or prefix=="":
            raise InvalidPrefixError("The argumnt [prefix] is empty or contains only whitespace. Please pass a valid prefix for a file")

        if np.unique(annotation.compute()).size<3:
            raise InvalidAnnotationsArray("Annotations must contain more than 3 unique values")

        # start workflow
        images=images[ :, None, ... ]

        logger.info("OBJECT CLASSIFICATION: extracting features")
        features = self.feature_extractor(mask, images, statistical_functions)

        annotated_cells_id, annotation=get_annotation( array_1=annotation, array_2=mask)

        X_train=features[ features[ "cell_ID" ].isin( annotated_cells_id )]
        X_train = X_train.drop("cell_ID", axis=1)

        logger.info("OBJECT CLASSIFICATION: starting training")
        self.object_training(X_train, annotation, prefix)

        logger.info("OBJECT CLASSIFICATION: starting classification")
        clf:RandomForestClassifier = joblib.load(os.path.join(self.output_folder, f"{prefix}_{self.MODEL_NAME}"))
        y_pred_all = self.object_classification(features.drop( [ "cell_ID" ], axis=1 ), clf)

        cell_ids=features[ "cell_ID" ]

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
        prefix: str,
    ) -> da.Array:
        return self.object_classifier_workflow(mask, images, annotation, statistical_functions, prefix)

class InvalidPrefixError(Exception):

    def __init__(self,*args):
        super().__init__(*args)

class InvalidAnnotationsArray(Exception):

    def __init__(self, *args):
        super().__init__(*args)

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
    def get_single_stats(stats: list[StrEnum]) -> list[str]:
        aggregate_stats = {Statistical_Functions.SUM,
                    Statistical_Functions.MEAN,
                    Statistical_Functions.COUNT,
                    Statistical_Functions.VAR,
                    Statistical_Functions.KURTOSIS,
                    Statistical_Functions.SKEW}
        result = []
        for stat in stats:
            if stat in aggregate_stats:
                result.append(stat.value)

        return result
