import dask.array as da
import numpy as np
import dask

from numpy.typing import NDArray


def get_annotation(array_1: da.Array, array_2: da.Array) -> tuple[NDArray, NDArray]:
    """
    Computes the mapping between annotation labels in `array_1` and corresponding labels in `array_2`.

    This function iterates over all unique labels in `array_1` (annotations) and determines
    which labels in `array_2` (original mask) they correspond to based on overlap. If a label
    in `array_2` is annotated by multiple labels from `array_1`, the annotation with the
    maximum overlap is retained.

    Parameters
    ----------
    array_1 : da.Array
        The annotation array where labels represent different annotated regions.
    array_2 : da.Array
        The original mask array containing labels of segmented regions.

    Returns
    -------
    Tuple of numpy arrays:
        - An array of unique labels from `array_2` that have been annotated.
        - An array of corresponding annotation labels from `array_1` that provide
          the best annotation (i.e., maximum overlap) for each label in `array_2`.

    Notes
    -----
    - Labels in `array_1` and `array_2` with a value of `0` (background) are ignored.
    - If a label in `array_2` is annotated by multiple labels from `array_1`, only the label
      with the maximum overlap is retained.
    """
    array_1_unique_labels = da.unique(array_1).compute()
    array_1_unique_labels = array_1_unique_labels[array_1_unique_labels != 0]
    array_2_unique_labels = da.unique(array_2).compute()
    array_2_unique_labels = array_2_unique_labels[array_2_unique_labels != 0]

    results = np.full(
        (array_2_unique_labels.size, array_1_unique_labels.size), fill_value=np.nan
    )  # array with shape (nr_of_labels_in_mask, nr_of_annotations)

    for i, _annotation_label in enumerate(array_1_unique_labels):
        overlap_array = array_2[(array_1 == _annotation_label)]
        overlap_array.compute_chunk_sizes()  # modifies array in place (check for materialization of array)

        label, label_count = dask.compute(*da.unique(overlap_array, return_counts=True))

        # user could have annotated some background. if so 0 is in label, and we remove it
        if 0 in label:
            label = label[1:]
            label_count = label_count[1:]

        idxs = np.searchsorted(label, array_2_unique_labels)

        idxs[idxs >= label.size] = 0
        found = label[idxs] == array_2_unique_labels
        results[found, i] = label_count

    max_indices = np.full(array_2_unique_labels.size, np.nan)
    valid_rows = ~np.all(np.isnan(results), axis=1)
    max_indices[valid_rows] = np.nanargmax(results[valid_rows], axis=1)

    isnotnan = ~np.isnan(max_indices)
    final_label = max_indices[isnotnan].astype(int)
    final_label_copy = final_label.copy()
    for _label in final_label_copy:
        final_label[final_label_copy == _label] = array_1_unique_labels[_label]

    return array_2_unique_labels[isnotnan], final_label
