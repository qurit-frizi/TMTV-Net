"""
We assume throughout that the image format is (samples, channels, height, width)
"""

import numpy as np
import numbers

import utilities

def to_value(v):
    """
    Convert where appropriate from tensors to numpy arrays

    Args:
        v: an object. If ``torch.Tensor``, the tensor will be converted to a numpy
            array. Else returns the original ``v``

    Returns:
        ``torch.Tensor`` as numpy arrays. Any other type will be left unchanged
    """
    if isinstance(v, torch.Tensor):
        return v.cpu().data.numpy()
    return v

from export import as_rgb_image
from export import as_image_ui8
from export import export_image


def export_as_image(name, samples, sample_id, export_root, txt_file):
    """
    Export a value as an image
    :param name:
    :param samples:
    :param export_root:
    :param txt_file:
    :return:
    """
    samples = to_value(samples)
    # an image MUST have a filter component, else we could confuse
    # if as a 2D array that we want to export in a text file
    if not isinstance(samples, np.ndarray) or len(samples.shape) <= 3:
        return False
    rgb = as_rgb_image(samples[sample_id])
    if rgb is None:
        return False

    # use the batch min/max to determine the range of the pixels. If the batch is
    # not too small, it should be fine.
    # TODO Can we find a more deterministic range?
    batch_min = np.min(samples)
    batch_max = np.max(samples)
    ui8 = as_image_ui8(rgb, min_value=batch_min, max_value=batch_max)
    if ui8 is None:
        return False
    path = export_root + name + '.png'
    export_image(ui8, path)
    return True


def export_as_npy(name, samples, sample_id, export_root, txt_file):
    samples = to_value(samples)

    if isinstance(samples, np.ndarray):
        if len(samples.shape) == 1 or len(samples.shape) == 0:
            return False  # if 1D, we probably want this exported as text
        if len(samples.shape) == 2 and samples.shape[1] < 10000:
            return False

        path = export_root + name + '.npy'
        np.save(path, samples[sample_id])
        return True
    return False


def export_as_string(name, samples, sample_id, export_root, txt_file, max_len=1024):
    samples = to_value(samples)

    if isinstance(samples, numbers.Number) or isinstance(samples, str):
        txt_file.write('%s=%s\n' % (name, str(samples)))
        return True

    if isinstance(samples, np.ndarray) and len(samples.shape) <= 2 and len(samples.shape) >= 1:
        txt_file.write('%s=%s\n' % (name, str(samples[sample_id])))
        return True
    
    if isinstance(samples, list) and isinstance(samples[0], str):
        txt_file.write('%s=%s\n' % (name, str(samples[sample_id])))
        return True

    # default fallback: as a string!
    value_str = str(samples)
    txt_file.write('%s=%s\n' % (name, value_str[:max_len]))
    return True


def export_functions():
    """
    Default export functions
    :return:
    """
    return [
        export_as_image,
        export_as_npy,
        export_as_string
    ]


def clean_mapping_name(name):
    return name.replace('_truth', '')


def export_sample(
        batch,
        sample_id,
        export_root,
        txt_file,
        exports=export_functions,
        features_to_discard=None,
        clean_mapping_name_fn=clean_mapping_name,
        classification_mappings=None,
        ):
    """
    Export a sample from the batch
    :param batch: a batch
    :param sample_id: the index of the sample to export
    :param export_root: where to export the data (typically including the sample id)
    :param txt_file: where to export text data. Must be an opened file for write
    :param exports: a functor returning the functions to be used for the export
    :param features_to_discard: a list of feature names to discard
    :param clean_mapping_name_fn: function to translate the name of batch feature name to class output name
    :param classification_mappings: a dictionary of mapping output to translate class ID to class name
    """
    fns = exports()
    for feature_name, feature in batch.items():
        if features_to_discard is not None and feature_name in features_to_discard:
            continue
        for fn in fns:
            exported = fn(feature_name, feature, sample_id, export_root, txt_file)
            if exported:
                break

    # check if we have some classification mapping names: it would be much easier
    # to read the actual class name than the class ID
    if classification_mappings is not None:
        for name, value in batch.items():
            value = to_value(value)
            if not isinstance(value, np.ndarray):
                continue
            mapping_name = clean_mapping_name_fn(name)
            m = classification_mappings.get(mapping_name)
            if m is not None:
                class_name = utilities.get_class_name(m, value[sample_id])
                txt_file.write('{}_str={}\n'.format(name, class_name))

