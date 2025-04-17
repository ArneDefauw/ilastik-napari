import os
import sys
import logging
import joblib
import argparse
import xarray as xa
import dask.array as da
import re
import numpy as np

from spatialdata.models import Labels2DModel
from spatialdata import SpatialData
from spatialdata import read_zarr
from ilastik.napari.object_classification import Statistical_Functions, Object_Classifier

CWD = os.getcwd()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
OBJECT_LABEL = "O"
PIXEL_LABEL = "P"

logger = logging.getLogger("model classifier")

def main():
    """
        TODO: documentation
    """
    parser = argparse.ArgumentParser(prog='pixel/object classifier',
                                      description=main.__doc__)

    parser.add_argument('-m', '--model', required=True, type=str, help='Path to the model')
    parser.add_argument('-i', '--images', required=True, type=str, help='Path to the images')
    parser.add_argument('-f', '--file_path', required=True, type=str, help='Path to the result of the classification')
    parser.add_argument('--overwrite', action='store_true', help='Use to overwrite the file at destinantion')

    args = parser.parse_args()

    model_path = check_and_get_path(args.model, '.pkl')
    images_path = check_and_get_path(args.images, '.zarr')
    result_path = os.path.join(CWD, args.file_path).rstrip("\\/")


    logger.info("OBJECT CLASSIFICATION")
    result = object_classification_workflow(model_path, images_path)


    sdata = SpatialData()

    proba = result.rechunk(result.chunksize)
    sdata["labels"] = Labels2DModel.parse(
            proba,
            dims=("y", "x"),
            chunks=proba.chunksize,
        )
    sdata.write(
        result_path,
        overwrite=args.overwrite
    )


def object_classification_workflow(
        model_path:str,
        images_path:str,
    )->da.Array:
    logger.info("FETCHING DATA FOR OBJECT CLASSIFICATION")
    model = joblib.load(model_path)

    stats = []
    image_names = []
    features_names = model.feature_names_in_

    for f in features_names:
        stats.append(Statistical_Functions(re.search(r'^[A-Za-z]+_?[A-Za-z]+', f).group()))

        tup = f.split(" ")

        if len(tup)==2:
            image_names.append(tup[1])

    stats = set(stats)
    image_names = set(image_names)

    sdata = read_zarr(images_path)

    images = []

    for i in image_names:
        images.append((i, check_and_convert_layer(sdata.images[i])))

    print(images)

    masks = list(sdata.labels.keys())

    if len(masks)>1:
        logger.warning("More than one mask has been passed getting top")

    mask = check_and_convert_layer(sdata.labels[masks[0]])

    features = Object_Classifier.feature_extractor(mask, images, stats, 100)

    X_features = features.drop( [Object_Classifier.ID_COLUMN_NAME], axis=1 )[model.feature_names_in_]

    y_pred_all = Object_Classifier.object_classification(X_features, model)

    cell_ids=features[Object_Classifier.ID_COLUMN_NAME]

    assert cell_ids.shape == y_pred_all.shape

    max_id = cell_ids.max()

    dtype = np.int8
    if len(np.unique(y_pred_all))>255:
        dtype = np.int16
    lookup = np.zeros(max_id + 1, dtype=dtype)

    lookup[cell_ids] = y_pred_all
    return da.take(lookup, mask)


def check_and_convert_layer(item):
    if isinstance(item, xa.DataTree):
        root = item.groups[1]
        item = item[root].image
    return item.data

def check_and_get_path(path_name:str, check_end:str=None):
    path = os.path.join(CWD, path_name).rstrip("\\/")

    if check_end:
        if not path.endswith(check_end):
            show_error(f"{path} does not end with {check_end}")

    if not os.path.exists(path):
        show_error(f"{path} does not exist.")

    return path

def show_documenation():
    print(main.__doc__)
    sys.exit()

def show_error(msg):
    # print("\033[91m")
    print("-"*len(msg))
    print(msg)
    print("-"*len(msg))
    sys.exit(1)

main()
