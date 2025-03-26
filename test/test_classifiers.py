import numpy as np
import os

from ilastik.napari.filters import GaussianDask, FilterSet, GaussianGradientMagnitudeDask, LaplacianOfGaussianDask
from ilastik.napari.object_classification import Pixel_Classifier, Object_Classifier
from spatialdata import read_zarr

FILE_PATH = r"C:\Users\matti\Documents\WERK\STAGE\VIB\data\sdata_channels.zarr"
MASK_NAME = 'mask_whole_testing'

def test_pixel_classification_workflow(tmp_path):
    pc = Pixel_Classifier(f"{tmp_path}")
    sdata = read_zarr(FILE_PATH)

    image = sdata[list(sdata.images.keys())[0]]

    labels = np.random.choice([0, 1, 2], size=image.shape[1:], p=[0.998, 0.001, 0.001])

    features=FilterSet(filters=[GaussianDask(scale=0.3), GaussianGradientMagnitudeDask(scale=0.7), LaplacianOfGaussianDask(scale=0.3)])

    assert len(features.filters)==3

    prefix = "pre"

    results = pc._workflow(
        image,
        labels,
        features,
        prefix,
    )

    assert os.path.exists(os.path.join(tmp_path, f"{prefix}_{pc.PREPROCESSED_ARRAY_NAME}"))
    assert os.path.exists(os.path.join(tmp_path, pc.PREPROCESSING_PIPE_NAME))

    result_shape = image.shape[1:] + (len(np.unique(labels))-1,)

    assert result_shape==results.shape

def test_object_classification_workflow(tmp_path):
    oc = Object_Classifier(f"{tmp_path}")

    sdata = read_zarr(FILE_PATH)

    image = sdata[list(sdata.images.keys())[0]]

    shape =  image.squeeze().shape

    annotaions = np.zeros(shape, dtype=int)


    num_blobs = 5
    blob_size = (5, 5)

    for _ in range(num_blobs):
        # Randomly choose top-left corner of the blob
        row = np.random.randint(0, shape[0] - blob_size[0])
        col = np.random.randint(0, shape[1] - blob_size[1])

        # Randomly decide whether the blob contains 1s or 2s
        value = np.random.choice([1, 2])

        # Insert the blob
        annotaions[row:row + blob_size[0], col:col + blob_size[1]] = value

    results = oc.object_classifier_workflow(
        sdata[MASK_NAME],
        image,
        annotaions
    )

    assert os.path.exists(os.path.join(tmp_path, oc.MODEL_NAME))

    assert results.shape == shape
