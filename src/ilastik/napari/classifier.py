import dask.array as da
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MetaEstimatorMixin,
    TransformerMixin,
)


def _fit_with(func, X, y, **kwargs):
    return func(X[tuple(y.coords)], y.data, **kwargs)


def _preprocessing(X, y):
    y_ravel = y.ravel()
    linear_indices = np.nonzero(y_ravel)[0]  # could also be done in dask
    y_data = np.take(y_ravel, linear_indices)

    # linear_indices.compute_chunk_sizes() # if you would use dask
    # write linear_indices to intermediate zarr store to prevent large task graph, can do the same for y_data
    # da.from_array(linear_indices).to_zarr( "/Users/arnedf/VIB/DATA/test_data_ilastik/linear_indices.zarr" )
    # linear_indices=da.from_zarr(  "/Users/arnedf/VIB/DATA/test_data_ilastik/linear_indices.zarr"  )
    # client=Client( n_workers=1, threads_per_worker=10 )

    shape = X.shape
    results = [da.take(X[..., i].ravel(), linear_indices) for i in range(shape[-1])]

    X = da.stack(results, axis=-1)

    # make sure X and y_data chunks are aligned
    y_data = da.from_array(y_data, chunks=X.chunks[0])

    return X, y_data


def _fit_with_dask(func, X, y, **kwargs):
    X, y = _preprocessing(X, y)
    assert isinstance(X, da.Array)
    assert isinstance(y, da.Array)

    return func(X, y, **kwargs)


def _predict_with(func, X):
    *image_shape, n_features = X.shape
    preds = func(X.reshape((-1, n_features)))
    if preds.size == X.size:
        return preds.reshape(image_shape)
    else:
        return preds.reshape((*image_shape, -1))


class NDSparseClassifier(
    BaseEstimator, MetaEstimatorMixin, ClassifierMixin, TransformerMixin
):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        return _fit_with(self.estimator.fit, X, y, **kwargs)

    def partial_fit(self, X, y, **kwargs):
        return _fit_with(self.estimator.partial_fit, X, y, **kwargs)

    def predict(self, X):
        return _predict_with(self.estimator.predict, X)

    def predict_proba(self, X):
        return _predict_with(self.estimator.predict_proba, X)

    def predict_log_proba(self, X):
        return _predict_with(self.estimator.predict_log_proba, X)

    def transform(self, X):
        return _predict_with(self.estimator.transform, X)

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).predict(X)


class NDSparseDaskClassifier(
    BaseEstimator, MetaEstimatorMixin, ClassifierMixin, TransformerMixin
):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, **kwargs):
        return _fit_with_dask(self.estimator.fit, X, y, **kwargs)

    def partial_fit(self, X, y, **kwargs):
        return _fit_with_dask(self.estimator.partial_fit, X, y, **kwargs)

    def predict(self, X):
        return _predict_with(self.estimator.predict, X)

    def predict_proba(self, X):
        return _predict_with(self.estimator.predict_proba, X)

    def predict_log_proba(self, X):
        return _predict_with(self.estimator.predict_log_proba, X)

    def transform(self, X):
        return _predict_with(self.estimator.transform, X)

    def fit_predict(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).predict(X)
