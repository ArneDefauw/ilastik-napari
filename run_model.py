import os
import sys
import logging
import joblib
import argparse
import xarray as xa
import dask.array as da
import re
import numpy as np
import spatialdata

from spatialdata.models import Labels2DModel
from spatialdata import SpatialData
from spatialdata import read_zarr
from ilastik.napari.object_classification import Statistical_Functions, Object_Classifier

CWD = os.getcwd()
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
# OBJECT_LABEL = "O"
# PIXEL_LABEL = "P"

logger = logging.getLogger("model classifier")

def main():
    """
        TODO: documentation
    """
    parser = argparse.ArgumentParser(prog='object classifier',
                                      description=main.__doc__)

    parser.add_argument('-mo', '--model', required=True, type=str, help='Path to the model')
    parser.add_argument('-d', '--sdata', required=True, type=str, help='Path to the spatial data')
    parser.add_argument('-n', '--name', required=True, type=str, help='name of the result of the classification')
    parser.add_argument('-ma', '--mask', required=True, type=str, help='The name of the mask in the data')
    parser.add_argument('-o', '--output', required=True, type=str, help='Path to the output zarr store')
    parser.add_argument('--overwrite', required=True, action='store_true')


    args = parser.parse_args()

    model_path = check_and_get_path(args.model, '.pkl')
    sdata_path = check_and_get_path(args.sdata, '.zarr')


    logger.info("OBJECT CLASSIFICATION")
    result = object_classification_workflow(model_path, sdata_path, args.mask)


    sdata = read_zarr(sdata_path)

    proba = result.rechunk(result.chunksize)
    new_layer = Labels2DModel.parse(
            proba,
            dims=("y", "x"),
            chunks=proba.chunksize,
        )

    sdata[args.name] = new_layer

    sdata.write(
        args.output,
        overwrite=args.overwrite
    )


def object_classification_workflow(
        model_path:str,
        images_path:str,
        mask_name:str
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

    mask = check_and_convert_layer(sdata.labels[mask_name])

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
