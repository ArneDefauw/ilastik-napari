import dask.dataframe as dd
import dask.array as da
import joblib
import os
import loguru
import xarray as xa
import numpy as np
import sparse

from dask.distributed import Client
from functools import reduce
from harpy.utils._aggregate import RasterAggregator
from sklearn.ensemble import RandomForestClassifier
from ilastik.napari.filters import FilterSet
from sklearn.pipeline import Pipeline
from ilastik.napari.classifier import NDSparseDaskClassifier, NDSparseClassifier
from ilastik.napari.utils import get_annotation
from napari.qt.threading import thread_worker

logger = loguru.logger

def check_and_convert_arrays_to_dask(argument):
    if isinstance(argument, np.ndarray):
        return da.from_array(argument)
    elif isinstance(argument, xa.DataArray):
        return argument.data
    elif not isinstance(argument, da.Array):
        raise ValueError(f"The argument {argument} has data type: {type(argument)}. Please pass a value of the following data types [dask.array.Array, numpy.ndarray, xarray.DataArray].")
    return argument

def check_folder_argument(output_folder):
    if not isinstance(output_folder, str):
        raise ValueError(f"The argument [output_folder] has data type: {type(output_folder)}. Please pass a [str] instead.")
    if not os.path.isdir(output_folder):
        raise NotADirectoryError(f"The given path: {output_folder} for argument [output_folder] is not a directory. Please pass a valid one.")


@thread_worker
def _pixel_classification(image, labels, features):
    feature_map = features.transform(np.asarray(image.data))
    sparse_labels = sparse.COO.from_numpy(np.asarray(labels.data))

    clf = NDSparseClassifier(RandomForestClassifier())
    clf.fit(feature_map, sparse_labels)
    res = clf.predict_proba(feature_map)

    out = np.moveaxis(res, -1, 0)

    return out

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
        image,
        estimators,
        prefix: str,  # prefix for preprocessed array
        overwrite: bool = False,
    ) -> da.Array:
        pipe = Pipeline(estimators)

        arrays = []
        for i in image:
            arrays.append(pipe.transform(i))

        feature_map_lazy = da.concatenate(arrays, axis=2)

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

    def pixel_training(self, X, labels, model_path, **client_kwargs):
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
        predict_proba=True,
        **client_kwargs,
    ) -> da.Array:
        # check arguments
        image = check_and_convert_arrays_to_dask(image)

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
                drop_axis=-1,
                new_axis=-1,
                chunks=image.chunks[:-1]
                + ((nr_of_labels,),),  # how can we guess this dimension
                model=clf_scatter,
                # TODO output dtype not correct, need to fix via meta
            )

        return array_result

    def _workflow(
        self,
        image: da.Array,
        labels: xa.DataArray | np.ndarray,
        features:FilterSet,
        prefix: str,
        to_train: bool=True,
        overwrite: bool=False,
    ):
        # check arguments
        image = check_and_convert_arrays_to_dask(image)

        if isinstance(labels, xa.DataArray):
            labels = labels.values
        elif not isinstance(labels, np.ndarray):
            raise ValueError(f"The argument [labels] has data type: {type(labels)}. Please pass a value of the following data types [numpy.ndarray, xarray.DataArray].")

        check_folder_argument(self.output_folder)

        if len(labels.shape)>2:
            labels = labels.squeeze(0)

        # Start of workflow
        estimators = [("features", features)]

        data = self.preprocessing(
            image=image,
            estimators=estimators,
            prefix=prefix,
            overwrite=overwrite,
        )

        if to_train:
            self.pixel_training(
                X=data,
                labels=labels,
                model_path=os.path.join(
                    self.output_folder, self.MODEL_NAME
                ),  # path to trained model
                # kwargs passed to client
                processes=False,
                n_workers=1,
                threads_per_worker=10,
            )

        model_path = os.path.join(self.output_folder, self.MODEL_NAME)
        assert os.path.exists(model_path), f"{model_path} does not exist!"
        clf = joblib.load(model_path)

        results = self.pixel_classification(
            image=data,  # pass the preprocessed data
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
    MASK_NAME = "masks_whole"
    ANNOTATIONS_NAME = "annotation"
    ALL_STATISTICAL_FUNCTIONS = ("sum", "mean", "count", "var", "kurtosis", "skew")
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
        mask: da.Array | np.ndarray | xa.DataArray,
        image: da.Array | np.ndarray | xa.DataArray,
        stats:tuple[str] = ("sum", "mean", "count", "var", "kurtosis", "skew"),
    ) -> dd.DataFrame:

        # argument checks
        mask = check_and_convert_arrays_to_dask(mask)
        image = check_and_convert_arrays_to_dask(image)

        mask = mask[None, ...]

        aggregator=RasterAggregator(mask_dask_array=mask, image_dask_array=image)
        features=aggregator.aggregate_stats(stats_funcs=stats)

        for index in range(len(stats)):
            prefix = stats[index]+"_"
            feature = features[index]
            feature.columns = [f"{prefix}{c}" if f"{c}".isdigit() else c for c in feature.columns]

        res = reduce(lambda left, right: dd.merge(left, right, on='cell_ID', how='outer'), features)
        res = res.loc[res['cell_ID']!=0]

        return res

    def object_training(
        self,
        X_train: dd.DataFrame,
        y_train: dd.DataFrame,
    ) -> None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        joblib.dump(clf, os.path.join(self.output_folder, self.MODEL_NAME))

    def object_classification(
        self,
        X,
    ) -> np.ndarray:
        clf:RandomForestClassifier = joblib.load(os.path.join(self.output_folder, self.MODEL_NAME))
        return clf.predict(X)

    def object_classifier_workflow(
        self,
        mask: da.Array | np.ndarray | xa.DataArray,
        images: list[da.Array] | list[np.ndarray] | list[xa.DataArray],
        annotation: da.Array | np.ndarray | xa.DataArray,
    ) -> da.Array:
        mask = check_and_convert_arrays_to_dask(mask)
        annotation = check_and_convert_arrays_to_dask(annotation)

        if isinstance(images, list):
            images = [check_and_convert_arrays_to_dask(index) for index in images]
        else:
            raise ValueError(f"The argument [images] has data type: {type(mask)}. Please pass a [list] of the following data types [dask.array.Array, numpy.ndarray, xarray.DataArray].")

        image=da.concatenate(images)
        image=image[ :, None, ... ]

        # TODO: when extracting features few ids are missing: 79,
        features = self.feature_extractor(mask, image, self.ALL_STATISTICAL_FUNCTIONS)

        annotated_cells_id, annotation=get_annotation( array_1=annotation, array_2=mask)

        X_train=features[ features[ "cell_ID" ].isin( annotated_cells_id )]
        X_train = X_train.drop("cell_ID", axis=1)

        self.object_training(X_train, annotation)

        y_pred_all = self.object_classification(features.drop( [ "cell_ID" ], axis=1 ))

        cell_ids=features[ "cell_ID" ]

        assert cell_ids.shape == y_pred_all.shape

        max_id = cell_ids.max()
        lookup = np.zeros(max_id + 1, dtype=y_pred_all.dtype)
        lookup[cell_ids] = y_pred_all
        return da.take(lookup, mask)

    @thread_worker
    def object_classifier_workflow_thread(
        self,
        mask: da.Array | np.ndarray | xa.DataArray,
        images: list[da.Array] | list[np.ndarray] | list[xa.DataArray],
        annotation: da.Array | np.ndarray | xa.DataArray,
    ) -> da.Array:
        return self.object_classifier_workflow(mask, images, annotation)
