import sys
sys.path.append('src')
# Deployment env
import os
INPUT_DIR = os.environ["RAIVEN_INPUT_DIR"]
OUTPUT_DIR = os.environ["RAIVEN_OUTPUT_DIR"]

import copy
import time
from typing import Tuple, Dict, Optional, List, Any, Union, Sequence, Callable
import torch
from functools import partial
from typing_extensions import Literal
from layers.crop_or_pad import crop_or_pad_fun

#torch.set_num_threads(1)
#torch.multiprocessing.set_sharing_str.ategy('file_system')

# from corelib import read_nifti, make_sitk_image, get_sitk_image_attributes, resample_sitk_image
from preprocess import PreprocessDataV4_lung_soft_tissues_hot as Preprocessing
from model_unet_multiclass_deepsupervision_configured_v1 import SimpleMulticlassUNet_dice_ce_fov_v9_ds_lung_soft_hot_boundary as ModelLargeFov
from model_refiner_multiclass_deepsupervision_configured_v1 import Refiner_dice_ce_fov_v1_ds_lung_soft_hot_boundary as ModelRefiner
from model_stacking import ModelStacking as ModelEnsemble
# from auto_pet.projects.segmentation.preprocessing.create_dataset import create_case
# from auto_pet.projects.segmentation.callbacks import create_inference
from basic_typing import Batch
import SimpleITK as sitk
from SimpleITK import GetArrayFromImage, sitkNearestNeighbor, Image
import numpy as np
from torch import nn

import collections
from collections import namedtuple
import numbers
import scipy.ndimage
from utilities import transfer_batch_to_device, get_device
from trainer_v2 import TrainerV2

from raiven import Raiven # needs md2pdf module, pip install md2pdf, pip install pango
import DicomNiftiConversion as dnc
from rt_utils import RTStructBuilder
import pydicom

import matplotlib.pyplot as plt
# import markdown
from mdutils.mdutils import MdUtils
import csv
import shutil
import nibabel as nib
import datetime   
import pytz
# import pydicom
from compute_features import read_nii_mask_return_tmtv_dmax

#----animate
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib import animation


# # generating gif out of slices of 3-dimensional numpy array
# def generate_gif(numpy_3d_array):
#     fig = plt.figure()
#     im = plt.imshow(numpy_3d_array[0, :, :],    # display first slice
#                     animated=True,
#                     cmap='turbo',               # color mapping
#                     vmin=np.iinfo('uint8').min, # lowest value in numpy_3d_array
#                     vmax=np.iinfo('uint8').max) # highest value in numpy_3d_array
#     plt.colorbar(label='turbo', shrink=0.75)
#     plt.tight_layout()

#     def init():
#         im.set_data(numpy_3d_array[0, :, :])
#         return im,

#     def animate(i):
#         im.set_array(numpy_3d_array[i, :, :])
#         return im,

#     # calling animation function of matplotlib
#     anim = animation.FuncAnimation(fig,
#                                    animate,
#                                    init_func=init,
#                                    frames=np.shape(numpy_3d_array)[0],  # amount of frames being animated
#                                    interval=1000,                       # update every second
#                                    blit=True)
#     anim.save("test.gif")   # save as gif
#     plt.show()
#-----------
Number = Union[float, int]


"""Generic Tensor with th `N` and `C` components removed"""
TensorX = Union[np.ndarray, torch.Tensor]


"""Generic Tensor as numpy or torch. Must be shaped as [N, C, D, H, W, ...]"""
TensorNCX = Union[np.ndarray, torch.Tensor]


"""Shape expressed as [D, H, W, ...] components"""
ShapeX = Sequence[int]

"""Generic Shape"""
Shape = Sequence[int]

InferenceOutput = namedtuple('InferenceOutput', 'output_found output_truth output_input output_raw')


# import pyradise
# import pyradise.fileio as ps_io
# import pyradise.data as ps_data

# def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0)):
#     """
#     Resample itk_image to new out_spacing
#     :param itk_image: the input image
#     :param out_spacing: the desired spacing
#     :return: the resampled image
#     """
#     # get original spacing and size
#     original_spacing = itk_image.GetSpacing()
#     original_size = itk_image.GetSize()
#     # calculate new size
#     out_size = [
#         int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
#         int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
#         int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
#     ]
#     # instantiate resample filter with properties and execute it
#     resample = sitk.ResampleImageFilter()
#     resample.SetOutputSpacing(out_spacing)
#     resample.SetSize(out_size)
#     resample.SetOutputDirection(itk_image.GetDirection())
#     resample.SetOutputOrigin(itk_image.GetOrigin())
#     resample.SetTransform(sitk.Transform())
#     resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
#     resample.SetInterpolator(sitk.sitkNearestNeighbor)
#     return resample.Execute(itk_image)


# class ExampleModalityExtractor(ps_io.ModalityExtractor):

#     def extract_from_dicom(self, path: str) -> Optional[ps_data.Modality]:
#         # Extract the necessary attributes from the DICOM file
#         tags = (ps_io.Tag((0x0008, 0x0060)),  # Modality
#                 ps_io.Tag((0x0008, 0x103e)))  # Series Description
#         dataset_dict = self._load_dicom_attributes(tags, path)

#         # Identify the modality rule-based
#         modality = dataset_dict.get('Modality', {}).get('value', None)
#         series_desc = dataset_dict.get('Series Description', {}).get('value', '')
#         if modality == 'MR':
#             if 't1' in series_desc.lower():
#                 return ps_data.Modality('T1')
#             elif 't2' in series_desc.lower():
#                 return ps_data.Modality('T2')
#             else:
#                 return None
#         elif modality == 'PT':
#             return ps_data.Modality('PT')
#         else:
#             return None

#     def extract_from_path(self, path: str) -> Optional[ps_data.Modality]:
#         # We can skip the implementation of this method, because we work
#         # exclusively with DICOM files
#         return None

# class ExampleOrganExtractor(ps_io.OrganExtractor):

#     def extract(self,
#                 path: str
#                 ) -> Optional[ps_data.Organ]:
#         # Identify the discrete image file's organ rule-based
#         filename = os.path.basename(path)

#         # Check if the image contains a seg prefix
#         # (i.e., it is a segmentation)
#         if not filename.startswith('seg'):
#             return None

#         # Split the filename for extracting the organ name
#         organ_name = filename.split('_')[-1].split('.')[0]
#         return ps_data.Organ(organ_name)


# class ExampleAnnotatorExtractor(ps_io.AnnotatorExtractor):

#     def extract(self,
#                 path: str
#                 ) -> Optional[ps_data.Annotator]:
#         # Identify the discrete image file's annotator rule-based
#         filename = os.path.basename(path)

#         # Check if the image contains a seg prefix
#         # (i.e., it is a segmentation)
#         if not filename.startswith('seg'):
#             return None

#         # Split the filename for extracting the annotator name
#         annotator_name = filename.split('_')[2]
#         return ps_data.Annotator(annotator_name)

# def convert_subject_to_dicom_rtss(input_dir_path: str,
#                                   output_dir_path: str,
#                                   dicom_image_dir_path: str,
#                                   use_3d_conversion: bool = True
#                                   ) -> None:
#     # Specify a reference modalities
#     # This is the modality of the DICOM image series that will be
#     # referenced in the DICOM-RTSS.
#     reference_modality = 'PT'

#     # Create the loader
#     loader = ps_io.SubjectLoader()

#     # Create the writer and specify the output file name of the
#     # DICOM-RTSS files
#     writer = ps_io.DicomSeriesSubjectWriter()
#     rtss_filename = 'rtss.dcm'

#     # (optional)
#     # Instantiate a new selection to exclude the original DICOM-RTSS SeriesInfo
#     # Note: If this is omitted the original DICOM-RTSS will be copied to the
#     # corresponding output directory.
#     selection = ps_io.NoRTSSInfoSelector()

#     # Create the file crawler for the discrete image files and
#     # loop through the subjects
#     crawler = ps_io.DatasetFileCrawler(input_dir_path,
#                                        extension='.nii.gz',
#                                        modality_extractor=ExampleModalityExtractor(),
#                                        organ_extractor=ExampleOrganExtractor(),
#                                        annotator_extractor=ExampleAnnotatorExtractor())
#     for series_info in crawler:
#         # Load the subject
#         subject = loader.load(series_info)

#         # Print the progress
#         print(f'Converting subject {subject.get_name()}...')

#         # Construct the path to the subject's DICOM images
#         dicom_subject_path = os.path.join(dicom_image_dir_path, subject.get_name())

#         # Construct a DICOM crawler to retrieve the reference
#         # DICOM image series info
#         dcm_crawler = ps_io.SubjectDicomCrawler(dicom_subject_path,
#                                                 modality_extractor=ExampleModalityExtractor())
#         dicom_series_info = dcm_crawler.execute()

#         # (optional)
#         # Keep all SeriesInfo entries that do not describe a DICOM-RTSS for loading
#         dicom_series_info = selection.execute(dicom_series_info)

#         # (optional)
#         # Define the metadata for the DICOM-RTSS
#         # Note: For some attributes, the value must follow the value
#         # representation of the DICOM standard.
#         meta_data = ps_io.RTSSMetaData(patient_size='180',
#                                        patient_weight='80',
#                                        patient_age='050Y',
#                                        series_description='Converted from NIfTI')

#         # Convert the segmentations to a DICOM-RTSS with standard smoothing settings.
#         # For the conversion we can either use a 2D or a 3D algorithm (see API reference
#         # for details).
#         # Note: Inappropriate smoothing leads to corrupted structures if their size
#         # is too small
#         if use_3d_conversion:
#             conv_conf = ps_io.RTSSConverter3DConfiguration()
#         else:
#             conv_conf = ps_io.RTSSConverter2DConfiguration()

#         converter = ps_io.SubjectToRTSSConverter(subject,
#                                                  dicom_series_info,
#                                                  reference_modality,
#                                                  conv_conf,
#                                                  meta_data)
#         rtss = converter.convert()

#         # Combine the DICOM-RTSS with its output file name
#         rtss_combination = ((rtss_filename, rtss),)

#         # Write the DICOM-RTSS to a separate subject directory
#         # and include the DICOM files crawled before
#         # Note: If you want to output just a subset of the
#         # original DICOM files you may use additional selectors
#         print('adresssssssss:',output_dir_path)
#         writer.write(rtss_combination, output_dir_path,
#                      subject.get_name(), dicom_series_info)


def get_MIP_from_3Dnifti(path, axis_num):
    # Load the Nifti file
    img = nib.load(path)
    # Get the data from the Nifti file
    data = img.get_fdata()
    # Perform the MIP operation along the axis_num; z-axis=2
    mip = data.max(axis=axis_num)
    # return MIP image
    return mip

def read_nifti(path: str) -> sitk.Image:
    """Read a NIfTI image. Return a SimpleITK Image."""
    nifti = sitk.ReadImage(str(path))
    return nifti


def write_nifti(sitk_img: sitk.Image, path: str) -> None:
    """Save a SimpleITK Image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)


def make_sitk_image(
        image: np.ndarray,
        origin_xyz: np.ndarray,
        spacing_xyz: np.ndarray,
        direction_xyz: Tuple[float, ...] = (1, 0, 0, 0, 1, 0, 0, 0, 1)) -> sitk.Image:
    """
    Create an Simple ITK image from a numpy array

    Returns:
        a sitk image
    """
    assert len(image.shape) == 3
    assert len(origin_xyz) == 3
    assert len(spacing_xyz) == 3
    assert len(direction_xyz) == 9

    image_sitk = sitk.GetImageFromArray(image)
    image_sitk.SetOrigin(origin_xyz)
    image_sitk.SetSpacing(spacing_xyz)
    image_sitk.SetDirection(direction_xyz)

    return image_sitk


def get_sitk_image_attributes(sitk_image: sitk.Image) -> Dict:
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['pixelid'] = sitk_image.GetPixelIDValue()
    attributes['origin'] = sitk_image.GetOrigin()
    attributes['direction'] = sitk_image.GetDirection()
    attributes['spacing'] = np.array(sitk_image.GetSpacing())
    # attributes['shape'] = np.array(sitk_image.GetSize(), dtype=np.int)
    attributes['shape'] = np.array(sitk_image.GetSize(), dtype=int)
    return attributes


def resample_sitk_image(sitk_image: sitk.Image,
                        attributes=None,
                        interpolator=sitk.sitkLinear,
                        fill_value: float = 0) -> sitk.Image:
    """
    Resample a SimpleITK Image.
    Parameters
    ----------
    sitk_image : sitk.Image
        An input image.
    attributes : dict or None
        The desired output image's spatial domain (its meta-data). If None, the original image's meta-data is used.
    interpolator
        Available interpolators:
            - sitk.sitkNearestNeighbor : nearest
            - sitk.sitkLinear : linear
            - sitk.sitkGaussian : gaussian
            - sitk.sitkLabelGaussian : label_gaussian
            - sitk.sitkBSpline : bspline
            - sitk.sitkHammingWindowedSinc : hamming_sinc
            - sitk.sitkCosineWindowedSinc : cosine_windowed_sinc
            - sitk.sitkWelchWindowedSinc : welch_windowed_sinc
            - sitk.sitkLanczosWindowedSinc : lanczos_windowed_sinc
    fill_value : int or float
        A value used for padding, if the output image size is less than `new_size`.
    Returns
    -------
    sitk.Image
        The resampled image.
    Notes
    -----
    This implementation is based on https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    if attributes:
        # provided attributes:
        orig_pixelid = attributes['pixelid']
        orig_origin = attributes['origin']
        orig_direction = attributes['direction']
        orig_spacing = attributes['spacing']
        orig_size = attributes['shape']
    else:
        # use original attributes:
        orig_pixelid = sitk_image.GetPixelIDValue()
        orig_origin = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing = np.array(sitk_image.GetSpacing())
        orig_size = np.array(sitk_image.GetSize(), dtype=int)

    
    new_size = [int(s) for s in orig_size]  # SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputOrigin(orig_origin)
    resample_filter.SetOutputSpacing(orig_spacing)
    resample_filter.SetOutputDirection(orig_direction)
    resample_filter.SetDefaultPixelValue(fill_value)
    resample_filter.SetOutputPixelType(orig_pixelid)
    resample_filter.SetTransform(sitk.Transform())
    resampled_sitk_image = resample_filter.Execute(sitk_image)

    return resampled_sitk_image

def sub_tensor(tensor: torch.Tensor, min_indices: Shape, max_indices_exclusive: Shape) -> torch.Tensor:
    """
    Select a region of a tensor (without copy)

    Examples:
        >>> t = torch.randn([5, 10])
        >>> sub_t = sub_tensor(t, [2, 3], [4, 8])
        Returns the t[2:4, 3:8]

        >>> t = torch.randn([5, 10])
        >>> sub_t = sub_tensor(t, [2], [4])
        Returns the t[2:4]

    Args:
        tensor: a tensor
        min_indices: the minimum indices to select for each dimension
        max_indices_exclusive: the maximum indices (excluded) to select for each dimension

    Returns:
        torch.tensor
    """
    assert len(min_indices) == len(max_indices_exclusive)
    assert len(tensor.shape) >= len(min_indices)

    for dim, (min_index, max_index) in enumerate(zip(min_indices, max_indices_exclusive)):
        size = max_index - min_index
        tensor = tensor.narrow(dim, min_index, size)
    return tensor

def resize(v: TensorX, multiple_of: Union[int, Sequence[int]], padding_value: Number) -> Tuple[TensorNCX, ShapeX]:
    """
    Resize the volume so that its size is a multiple of `mod`.
.
    Padding is added at the end of the volume.

    Returns:
        resized volume, padding
    """
    assert len(v.shape) == 3, 'Must be DHW format!'
    padding = np.asarray(v.shape) % multiple_of
    for n in range(len(padding)):
        if padding[n] != 0:
            if isinstance(multiple_of, int):
                padding[n] = multiple_of - padding[n]
            else:
                assert len(padding) == len(multiple_of)
                padding[n] = multiple_of[n] - padding[n]

    v_sub = crop_or_pad_fun(
        v.unsqueeze(0).unsqueeze(0),  # needs NC channels!
        v.shape + padding,
        padding_default_value=padding_value
    )
    return v_sub.squeeze(0).squeeze(0), padding

def get_output_fn(outputs):
    return torch.softmax(outputs['seg'].output, dim=1)

def create_inference(configuration, fov_half_size=None, tile_step=None, test_time_augmentation_axis=None, get_output_fn=get_output_fn, nb_outputs=2, postprocessing_fn = partial(torch.argmax, dim=1), no_output_ref_collection=False, internal_type=torch.float32):
    if configuration is not None:
        sequence_model = configuration.training.get('sequence_model')
        if sequence_model is not None:
            inference_sequence = partial(inference_process_wholebody_3d,
                feature_names=('suv', 'seg', 'sequence_label', 'sequence_input', 'sequence_output'),
                output_truth_name='seg',
                multiple_of=None,
                main_input_name='suv',
                tiling_strategy='none',  # classification all at once!
                postprocessing_fn=partial(torch.argmax, dim=1),
                get_output=get_output_fn,
                nb_outputs=nb_outputs,
                no_output_ref_collection=no_output_ref_collection,
                internal_type=internal_type,
            )
            return inference_sequence


    if fov_half_size is None:
        fov_half_size = configuration.data.get('fov_half_size')

    if fov_half_size is None:
        # this is not a windowing based inference!
        # TODO To be handled (e.g., sequence model)
        return None

    if tile_step is None:
        tile_step = fov_half_size

    inference_3d = partial(inference_process_wholebody_3d,
        feature_names=('ct', 'ct_lung', 'suv', 'seg', 'ct_soft', 'suv_hot', 'cascade.inference.output_found', 'z_coords', 'y_coords', 'x_coords'),
        output_truth_name='seg',
        main_input_name='suv',

        tile_shape=fov_half_size * 2,
        tile_step=tile_step,
        tile_margin=0,
        multiple_of=fov_half_size * 2,

        tile_weight='weighted_central',
        tiling_strategy='tiled_3d',
        postprocessing_fn=partial(torch.argmax, dim=1),
        get_output=get_output_fn,
        nb_outputs=nb_outputs,
        invalid_indices_value=0.0,  # default to `no segmentation`
        no_output_ref_collection=no_output_ref_collection,
        internal_type=internal_type,
    )

    if test_time_augmentation_axis is None:
        test_time_augmentation_axis = configuration.training.get('test_time_augmentation_axis')

    if test_time_augmentation_axis:
        def flip_batch(batch, axis):
            new_batch = {}
            discard_features = ('bounding_boxes_min_max',)
            for name, value in batch.items():
                if isinstance(value, torch.Tensor) and len(value.shape) == 3:
                    # TODO: performance penalty: the full image is copied
                    # multiple times (one time per axis per augmentation)
                    new_batch[name] = torch.flip(value, [axis])
                elif isinstance(value, torch.Tensor) and len(value.shape) == 4 and name not in discard_features:
                    # this is for the probability map. Since we have
                    # an additional `C` component, the axis to be flipped
                    # is the next one
                    assert value.shape[0] == 2, f'name={name}, shape={value.shape}'
                    new_batch[name] = torch.flip(value, [axis + 1])
                else:
                    new_batch[name] = value
            return new_batch

        transforms = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        # the inverse of axis flip is the same axis flip
        transforms_inv = [
            partial(flip_batch, axis=0),
            partial(flip_batch, axis=1),
            partial(flip_batch, axis=2),
        ]

        tta_fn = partial(test_time_inference, 
            inference_fn=inference_3d, 
            transforms=transforms, 
            transforms_inv=transforms_inv
        )
        inference_3d = tta_fn
    return inference_3d


def zero_to_one_tfm(d):
    d = np.sqrt(d)
    max_value = d.max()
    min_value = d.min()
    return 1 - (d - min_value) / max_value


def central_weighting(
        block_shape: ShapeX,
        center_shape_fraction: float = 0.5,
        weight_from_distance_transform_fn=zero_to_one_tfm) -> np.ndarray:
    """
    Create a "weighting" where the center has maximum weight and weights further from the
    center have lower values.

    The purpose of block weighting is to avoid artifacts from reconstructing a large object from
    smaller blocks by weighting the boundaries in a way that interpolate between adjacent blocks.


    Args:
        block_shape: the shape of the blocks
        center_shape_fraction: fraction of the block to have its value set to maximum weight
        weight_from_distance_transform_fn: calculate the weighting from distance transform

    Returns:
        a weighting array
    """
    center_half_region = (np.asarray(block_shape) * center_shape_fraction).astype(int) // 2
    w = np.zeros(block_shape, dtype=np.float32)

    slices = [slice(d // 2 - c_h, d // 2 + c_h) for d, c_h in zip(block_shape, center_half_region)]
    w[tuple(slices)] = 1.0

    w = weight_from_distance_transform_fn(scipy.ndimage.morphology.distance_transform_cdt(w < 0.5))
    return w



def create_umap_3d_tiled(
        model,
        batch,
        tile_shape,
        tile_step,
        tile_margin,
        get_output,
        feature_names,
        nb_outputs=2,
        tile_weight=None,
        internal_type=torch.float32,
        invalid_indices_value=1):
    """
    Create a UMap for a 3D model
    """
    assert len(feature_names) > 0
    # add the N, C dimensions
    batch = {name: value.unsqueeze(0).unsqueeze(0) if name in feature_names else value for name, value in batch.items()}

    shape = batch[feature_names[0]].shape
    dim = len(shape) - 2
    assert dim == 3, 'expecting NCDHW shaped data!'
    if isinstance(tile_shape, numbers.Number):
        tile_shape = [tile_shape] * dim
    if isinstance(tile_margin, numbers.Number):
        tile_margin = [tile_margin] * dim
    if isinstance(tile_step, numbers.Number):
        tile_step = [tile_step] * dim

    shape_binary = [shape[0], nb_outputs] + list(shape[2:])

    final = torch.zeros(shape_binary, dtype=internal_type)
    final_weight = torch.zeros(shape, dtype=internal_type)
    shape_zyx = shape[2:]

    z = 0
    while z + tile_shape[0] <= shape_zyx[0]:
        y = 0
        while y + tile_shape[1] <= shape_zyx[1]:
            x = 0
            while x + tile_shape[2] <= shape_zyx[2]:
                features = {}
                for name in feature_names:
                    tiled, margin_min, margin_max = extract_sub_volume(
                        batch[name],
                        tile_shape,
                        (z, y, x),
                        margin=tile_margin
                    )
                    features[name] = tiled

                # non image features
                other_feature_names = set(batch.keys()) - set(feature_names)
                for name in other_feature_names:
                    features[name] = batch[name]

                model_device = get_device(model)
                features = transfer_batch_to_device(features, model_device)
                with torch.no_grad():
                    umap_slice_output = model(features)
                    if get_output is not None:
                        umap_slice_output = get_output(umap_slice_output)
                    sub_umap = umap_slice_output.cpu().detach()
                    assert sub_umap.shape[1] == nb_outputs, f'expecting nb_outputs={nb_outputs}, got={sub_umap.shape[1]}'

                # crop the added margin
                sub_umap = sub_tensor(sub_umap, [0, 0] + list(margin_min), list(sub_umap.shape[:2]) + list(margin_min + tile_shape))

                # copy the results to `final` tensor
                sub_final = sub_tensor(final, (0, 0, z, y, x), (final.shape[0], final.shape[1], z + tile_shape[0], y + tile_shape[1], x + tile_shape[2]))
                assert sub_final.shape == sub_umap.shape
                sub_final_weight = sub_tensor(final_weight, (0, 0, z, y, x), (final.shape[0], 1, z + tile_shape[0], y + tile_shape[1], x + tile_shape[2]))
                if tile_weight is None:
                    sub_final_weight += 1.0
                    sub_final += sub_umap
                else:
                    assert tile_weight.shape == sub_final_weight.shape
                    sub_final_weight += tile_weight
                    sub_final += sub_umap * tile_weight

                x += tile_step[2]
            y += tile_step[1]
        z += tile_step[0]

    #if final_weight.min() == 0:
    #    print('WARNING!!! The full FoV was not performed!')

    invalid_indices = torch.where(final_weight <= 0)
    final_weight[invalid_indices] = 1.0
    final[invalid_indices] = invalid_indices_value # make sure invalid indices are marked with background class!
    final /= final_weight
    return final

def extract_sub_volume(v: TensorNCX, shape: ShapeX, p: ShapeX, margin: ShapeX):
    p = np.asarray(p)
    margin = np.asarray(margin)
    shape = np.asarray(shape)
    v_shape = np.asarray(v.shape)[2:]
    dim = len(shape)

    margin_min = p - np.stack([np.asarray([0] * dim), p - margin]).max(axis=0)
    margin_max = np.stack([v_shape, p + shape + margin]).min(axis=0) - (p + shape)

    min_bb = p - margin_min
    max_bb = p + shape + margin_max
    sub_v = sub_tensor(v, [0, 0] + list(min_bb), list(v.shape[:2]) + list(max_bb))
    return sub_v, margin_min, margin_max

def uncrop(v, orig_shape):
    """
    Remove the left and right padding
    """
    assert len(orig_shape) == 3, 'Must be DHW format!'
    assert len(v.shape) == 5, 'Must be NCDHW'
    shape_difference = np.asarray(v.shape[2:]) - np.asarray(orig_shape)
    left_padding = shape_difference // 2
    right_padding = shape_difference - left_padding

    cropping_right = np.asarray(v.shape[2:]) - right_padding
    o = sub_tensor(
        v,
        min_indices=[0, 0] + list(left_padding),
        max_indices_exclusive=list(v.shape[:2]) + list(cropping_right)
    )
    return o

def inference_process_wholebody_3d(
        batch: Batch,
        model: nn.Module,
        feature_names: List[str],
        get_output: Any = None,
        output_truth_name='spect_full',
        main_input_name='spect_low3',
        tile_shape=(64, 64, 64),
        tile_step=(64, 64, 64),
        tile_margin=(32, 32, 32),
        tile_weight: Literal['none', 'weighted_central'] = 'none',
        nb_outputs=1,
        multiple_of: Optional[Union[int, Sequence[int]]] = 32,
        padding_value: Union[Number, Dict[str, Number]] = 0,
        other_feature_names: Optional[List[str]] = None,
        postprocessing_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        invalid_indices_value: float = 1,
        no_output_ref_collection=False,
        internal_type=torch.float32,
        tiling_strategy: Literal['none', 'tiled_3d'] = 'none') -> InferenceOutput:
    """
    Process fully convolutional network outputs with inputs that hare larger than
    the model field of view by tiling the input and aggregating the results.

    Args:
        invalid_indices_value: this is the value to be set for the voxels
            that have not been processed. Default to `1` to show any
            missing regions. Should not happen when all the parameters
            are set appropriately 
        no_output_ref_collection: if True, no reference data is collected
            for the output. This is mostly to save RAM...
    """

    normalized_batch = {}
    padded = None
    original_shape = batch[feature_names[0]].shape

    if isinstance(tile_shape, numbers.Number):
        tile_shape = [tile_shape] * 3

    feature_name_present = list(set(batch.keys()).intersection(set(feature_names)))

    # in certain cases (e.g., UNet), the model expects sizes
    # to be a multiple of a size
    if multiple_of is not None:
        for name in feature_name_present:
            if isinstance(padding_value, collections.Mapping):
                p = padding_value.get(name)
                assert p is not None, f'missing padding value for volume={name}'
            else:
                p = padding_value

            normalized_batch[name], padded = resize(batch[name], multiple_of, p)
    else:
        normalized_batch = batch

    if other_feature_names is not None:
        for name in other_feature_names:
            normalized_batch[name] = batch[name]

    if tile_weight == 'none':
        tile_weight = None
    elif tile_weight == 'weighted_central':
        tile_weight = central_weighting(block_shape=tile_shape)
        # must have NCX format
        tile_weight = torch.from_numpy(tile_weight).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f'value not supported={tile_weight}')

    # now do the tiling according to the strategy
    if tiling_strategy == 'none':
        model_device = get_device(model)
        normalized_batch = transfer_batch_to_device(normalized_batch, model_device)
        with torch.no_grad():
            output = model(normalized_batch)

        if get_output is not None:
            output = get_output(output)
    elif tiling_strategy == 'tiled_3d':
        output = create_umap_3d_tiled(
            model=model,
            batch=normalized_batch,
            tile_shape=tile_shape,
            tile_step=tile_step,
            tile_margin=tile_margin,
            tile_weight=tile_weight,
            get_output=get_output,
            feature_names=feature_name_present,
            nb_outputs=nb_outputs,
            invalid_indices_value=invalid_indices_value,
            internal_type=internal_type
        )
    else:
        raise ValueError(f'value={tiling_strategy} is not handled!')

    # un-crop the output
    if padded is not None and max(padded) != 0:
        output = uncrop(output, original_shape)

    # prepare the output result
    assert len(output.shape) == 5
    output_truth = None
    if output_truth_name is not None:
        output_truth = batch.get(output_truth_name)
    
    output_input = None
    if main_input_name is not None:
        output_input = batch.get(main_input_name)

    if postprocessing_fn is not None:
        output_final = postprocessing_fn(output)
    else:
        output_final = output

    # remove the `N` dimension
    output = output.squeeze(0)
    output_final = output_final.squeeze(0)
    assert output_final.shape == original_shape, f'expected shape={original_shape}, got={output.shape}'
    if no_output_ref_collection:
        output_truth=None
        output_input=None
        output_found=None
        #output=None
    return InferenceOutput(output_found=output_final, output_truth=output_truth, output_input=output_input, output_raw=output)


def create_case(uid: str, ct: Image, suv: Image, seg: Optional[Image], spacing: Tuple[float, float, float], case_name: Optional[str] = None, timepoint: Optional[str] = None, patient_name: Optional[str] = None, interpolator=sitk.sitkLinear) -> Dict:
    ct_np = GetArrayFromImage(ct)
    suv_np = GetArrayFromImage(suv)
    if seg is not None:
        seg_np = GetArrayFromImage(seg)
    # print('ct_np.shape=', ct_np.shape)
    # print('suv_np.shape=', suv_np.shape)
    # print('seg_np.shape', seg_np.shape)

    # this section has been added to resize CT to PET size in case of size mismatch
    if not ct_np.shape == suv_np.shape:
        # ctimg = sitk.ReadImage(ctpath, imageIO="NiftiImageIO")
        # ptimg = sitk.ReadImage(ptpath, imageIO="NiftiImageIO")  
        ct = sitk.Resample(ct, suv, interpolator=sitk.sitkLinear, defaultPixelValue=-1024)
        ct_np = GetArrayFromImage(ct)

    assert ct_np.shape == suv_np.shape, f'shape mismatch: {ct_np.shape}, {suv_np.shape}'
    if seg is not None:
        if not ct_np.shape == seg_np.shape:
            seg = sitk.Resample(seg, suv, interpolator=sitk.sitkNearestNeighbor, defaultPixelValue=0)
            seg_np = GetArrayFromImage(seg)
            # print('ct_np.shape=', ct_np.shape)
            # print('seg_np.shape', seg_np.shape)
            # print('suv_np.shape=', suv_np.shape)
        assert ct_np.shape == seg_np.shape

    attributes = get_sitk_image_attributes(suv)
    target_attributes = copy.copy(attributes)

    print(f'spacing={attributes["spacing"]}')

    # resampled to a specific spacing
    if spacing is not None:
        spacing_target = np.asarray(spacing, dtype=float)
        target_attributes['spacing'] = spacing_target
        target_attributes['shape'] = (np.asarray(attributes['shape']) * np.asarray(attributes['spacing']) / spacing_target).round().astype(np.int32)

        resampled_ct = resample_sitk_image(ct, target_attributes, fill_value=-1024.0, interpolator=interpolator)
        ct_np = GetArrayFromImage(resampled_ct)
        resampled_suv = resample_sitk_image(suv, target_attributes, fill_value=0, interpolator=interpolator)
        suv_np = GetArrayFromImage(resampled_suv)
        if seg is not None:
            resampled_seg = resample_sitk_image(seg, target_attributes, fill_value=0, interpolator=sitkNearestNeighbor) 
            seg_np = GetArrayFromImage(resampled_seg)
    else:
        # if we don't resample, the voxels MUST be aligned!
        assert ct_np.shape == suv_np.shape
        if seg is not None:
            assert ct_np.shape == seg_np.shape

    case_data = {
        'uid': uid,
        'patient_name': patient_name,
        'timepoint': timepoint,
        'ct': ct_np,
        'suv': suv_np,

        # we need this tag to do the inverse pre-processing
        'original_spacing': attributes['spacing'],
        'original_origin': attributes['origin'],
        'original_shape': attributes['shape'],
        'original_direction': attributes['direction'],
    }

    if seg is not None:
        case_data['seg'] = seg_np
        case_data['mtv'] = float(seg_np.sum())

    if spacing is not None:
        case_data['target_spacing'] = target_attributes['spacing']
        case_data['target_origin'] = target_attributes['origin']
        case_data['target_shape'] =  target_attributes['shape']

    return case_data

def load_case(ct_path: str, suv_path: str, spacing: Tuple[float, float, float] = (6, 6, 6)) -> Batch:
    def load_volume(path: str) -> Tuple[sitk.Image, str]:
        uuid = None
        if os.path.isdir(path):
            # if directory, assume there is a single file
            # in it to be loaded
            files = os.listdir(path)
            assert len(files) == 1, f'unexpected number of files! Path={path}'
            path = os.path.join(path, files[0])
            print(f'Loading file={path}')
            uuid = os.path.splitext(files[0])[0]
        else:
            print(f'path={path} is not a directory!')

        if uuid is None:
            uuid = os.path.splitext(os.path.basename(path))[0]

        if '.nii.gz' in path or '.nii' in path: 
            return read_nifti(path), uuid

        if '.mha' in path:
            return sitk.ReadImage(path), uuid

        raise RuntimeError(f'volume type is not handled! path={path}')
    
    ct, _ = load_volume(ct_path)
    suv, suv_uid = load_volume(suv_path)

    case_data = create_case(
        uid='test_case',
        ct=ct,
        suv=suv,
        seg=None,
        spacing=spacing
    )
    case_data['ct'] = torch.from_numpy(case_data['ct'])
    case_data['suv'] = torch.from_numpy(case_data['suv'])
    case_data['uid'] = suv_uid

    return case_data


def save_case(output: sitk.Image, path: str):
    if '.mha' in path or '.nii' in path:
        sitk.WriteImage(output, path, True)
        return
        
    if '.npy' in path:
        np.save(path, sitk.GetArrayFromImage(output))
        return

    raise RuntimeError(f'volume type is not handled! path={path}')


def debug_input(batch: Batch) -> None:
    suv = batch['suv']
    ct = batch['ct']
    print('PET shape=', suv.shape[::-1])
    print('PET max_value=', suv.max())
    print('PET spacing=', batch['original_spacing'])
    print('PET origin=', batch['original_origin'])
    print('PET direction=', batch['original_direction'])
    print('CT shape=', ct.shape[::-1])


def debug_output(seg: sitk.Image, seg_path: str, uid:str, dicom_path: str):
    mask_from_sitkImage_zyx = np.transpose(sitk.GetArrayFromImage(seg), (2, 1, 0))
    mask_from_sitkImage_xzy = np.transpose(mask_from_sitkImage_zyx, axes=(2, 0, 1))
    mask_from_sitkImage_xyz = np.transpose(mask_from_sitkImage_xzy, (2, 1, 0))
    mask_from_sitkImage_int64 = mask_from_sitkImage_xyz
    mask_from_sitkImage_bool = mask_from_sitkImage_int64.astype(bool)
    # Create new RT Struct. Requires the DICOM series path for the RT Struct.
    rtstruct = RTStructBuilder.create_new(dicom_series_path = dicom_path)
    # Add the 3D mask as an ROI setting the color, description, and name
    rtstruct.add_roi(
    mask=mask_from_sitkImage_bool, 
    color=[255, 0, 255], 
    name="TMTV ROI!"
    )
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    # rtstruct.save(os.path.join(OUTPUT_DIR, 'tmtv-rt-struct_'+timestamp))
    rtstruct.save(os.path.join(OUTPUT_DIR, uid+'_tmtv-rt-struct'))
    print('output spacing=', seg.GetSpacing())
    print('output origin=', seg.GetOrigin())
    print('output shape=', seg.GetSize())
    print('output direction=', seg.GetDirection())
    seg_np = sitk.GetArrayFromImage(seg)
    nb_voxels_c1 = seg_np.sum()
    mtv_ml = nb_voxels_c1 * np.prod(seg.GetSpacing()) / 1000
    print('nb_voxels_c1:',nb_voxels_c1)
    
    output_dict = read_nii_mask_return_tmtv_dmax(seg_path)
    return output_dict

def inference_fn(
        ct_path: str, 
        suv_path: str,
        uid: str,
        output_path: str, 
        model_large_fov: nn.Module, 
        model_refiner: nn.Module) -> None:

    print('inference setup!!')
    preprocessor_f32 = Preprocessing(internal_type=torch.float32)
    preprocessor_f16 = Preprocessing(internal_type=torch.float16)

    case_load_large_fov_time_start = time.perf_counter()
    case_data = load_case(ct_path=ct_path, suv_path=suv_path, spacing=(6, 6, 6))
    case_load_large_fov_time_end = time.perf_counter()
    # uuid = case_data['uid']
    
    case_preprocess_large_fov_time_start = time.perf_counter()
    case_data = preprocessor_f32(case_data)
    case_preprocess_large_fov_time_end = time.perf_counter()
    
    inference_large_fov_fn = create_inference(
        configuration=None,
        fov_half_size=np.asarray((64, 48, 48)),
        tile_step=np.asarray((64, 48, 48)),
        test_time_augmentation_axis=False,
        no_output_ref_collection=True,
        internal_type=torch.float16,
    )
    # print(inference_large_fov_fn)
    print('starting inference!!')
    inference_large_fov_time_start = time.perf_counter()
    output_large_fov = inference_large_fov_fn(case_data, model_large_fov)
    inference_large_fov_time_end = time.perf_counter()

    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/inference0.npy', output_large_fov.output_raw[0])
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/inference1.npy', output_large_fov.output_raw[1])
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/inference1_seg.npy', output_large_fov.output_found)
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv.npy', case_data['suv'])
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/ct.npy', case_data['ct'])

    target_origin_xyz = case_data['target_origin']
    target_spacing_xyz = case_data['target_spacing']

    del case_data
    case_data = None

    #
    # Refiner stage
    #
    case_data_original = load_case(ct_path=ct_path, suv_path=suv_path, spacing=None)
    debug_input(case_data_original)
    case_data_original = preprocessor_f16(case_data_original)

    # resample the probability map to original geometry
    target_image_stik = make_sitk_image(
        output_large_fov.output_raw[1].type(torch.float32).numpy(), 
        origin_xyz=target_origin_xyz, 
        spacing_xyz=target_spacing_xyz
    )

    original_origin_xyz = case_data_original['original_origin']
    original_spacing_xyz = case_data_original['original_spacing']
    original_direction_xyz = case_data_original['original_direction']
    original_shape_xyz = case_data_original['suv'].shape[::-1]
    original_attributes = get_sitk_image_attributes(target_image_stik)
    original_attributes['spacing'] = original_spacing_xyz
    original_attributes['shape'] = original_shape_xyz
    original_attributes['origin'] = original_origin_xyz

    image_resampled_itk = resample_sitk_image(target_image_stik, attributes=original_attributes, fill_value=0.0)
    image_resampled_npy = GetArrayFromImage(image_resampled_itk)
    case_data_original['cascade.inference.output_found'] = torch.from_numpy(image_resampled_npy)

    # 9.2 Gb here
    del(target_image_stik)
    target_image_stik = None
    del(image_resampled_itk)
    image_resampled_itk = None
    del(image_resampled_npy)
    image_resampled_npy = None

    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/suv_orig.npy', case_data_original['suv'])
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/ct_orig.npy', case_data_original['ct'])
    #np.save('/mnt/datasets/ludovic/AutoPET/tmp/cascade.inference.npy', case_data_original['cascade.inference.output_found'])

    inference_refiner_fn = create_inference(
        configuration=None,
        fov_half_size=np.asarray((48, 48, 48)),
        tile_step=np.asarray((48, 48, 48)),
        test_time_augmentation_axis=False,
        no_output_ref_collection=True,
        internal_type=torch.float16,
    )

    print('starting inference!!')
    inference_refiner_time_start = time.perf_counter()
    output_refiner_fov = inference_refiner_fn(case_data_original, model_refiner)
    inference_refiner_time_end = time.perf_counter()

    del(case_data_original)
    case_data_original = None

    output_found_sitk = make_sitk_image(
        output_refiner_fov.output_found.numpy(), 
        origin_xyz=original_origin_xyz, 
        spacing_xyz=original_spacing_xyz,
        direction_xyz=original_direction_xyz
    )
    # don't `use os.path.isdir`, the folder doesn't exist yet!
    is_output_dir = len(os.path.basename(os.path.join(output_path,'/'))) == 0
    if is_output_dir:
        # use the UUID as output filename
        # output_path = os.path.join(output_path, uuid + 'nii.gz')
        root_output_folder = output_path
        output_path = os.path.join(output_path, uid + '_SEG.nii.gz')
    else:
        root_output_folder = os.path.dirname(output_path)

    print('root_output_folder=', root_output_folder)
    if not os.path.exists(root_output_folder):
        print('Creating directories=', root_output_folder)
        os.makedirs(root_output_folder)

    save_case(output_found_sitk, output_path)

    print(f'Loading time={case_load_large_fov_time_end - case_load_large_fov_time_start}')
    print(f'Preprocessing time={case_preprocess_large_fov_time_end - case_preprocess_large_fov_time_start}')
    print(f'Inference time={inference_large_fov_time_end - inference_large_fov_time_start}')
    print(f'Inference refiner time={inference_refiner_time_end - inference_refiner_time_start}')
    return output_found_sitk, output_path


    

if __name__ == '__main__':
    # import sys
    
    """
    base_path = '/mnt/datasets/ludovic/AutoPET/dataset/raw/FDG-PET-CT-Lesions/PETCT_2e44706eaf/05-06-2005-NA-PET-CT Ganzkoerper  primaer mit KM-03974'
    #base_path = '/mnt/datasets/ludovic/AutoPET/dataset/raw/FDG-PET-CT-Lesions/PETCT_1f65acff65/05-06-2007-NA-PET-CT Ganzkoerper nativ u. mit KM-95034'
    sys.argv += [
        '--ct_path', f'{base_path}/CTres.nii.gz',
        '--pt_path', f'{base_path}/SUV.nii.gz',
        #'--ct_path', '/mnt/datasets/ludovic/AutoPET/submissions/test2/images/ct/1f65acff65.mha',
        #'--pt_path', '/mnt/datasets/ludovic/AutoPET/submissions/test2/images/pet/1f65acff65.mha',
        '--model_large_fov_path', '/mnt/datasets/ludovic/AutoPET/submissions/AutoPET_cnn_v22/ensemble/best_dice_e500_0.5048483769139531.model',
        '--model_refiner_path', '/mnt/datasets/ludovic/AutoPET/submissions/AutoPET_cnn_v22/model_refiner/best_dice_foreground_e6000_0.6803068066681786.model',
        #'--output_path', '/mnt/datasets/ludovic/AutoPET/tmp/inference1_refiner_seg.npy',
        '--output_path', '/mnt/datasets/ludovic/AutoPET/tmp/',
        '--device', 'cuda:0'
    ]
    """
    
    
    
    import argparse
    parser = argparse.ArgumentParser(description='Process PET and CT wholebody images and gives suspicious segments')
    # parser.add_argument('--ct_path', help='path to CT volume. Can be a directory with a single volume or full pathname', default='./TMP_INPUT/CT')
    # parser.add_argument('--pt_path', help='path to 18F-FDG PET volume expressed in SUVbw unit. Can be a directory with a single volume or full pathname', default='./TMP_INPUT/PT')
    # parser.add_argument('--model_large_fov_path', help='', default='./src/models/e_latest.model')
    # parser.add_argument('--model_refiner_path', help='', default='./src/models/r_latest.model')
    parser.add_argument('--output_path', help='folder or full path where the segmentation will be exported', default=OUTPUT_DIR)
    # parser.add_argument('--device', help='one of (`cpu`, `cuda:0`, ...)', default='cpu')
    parser.add_argument('--device', help='one of (`cpu`, `cuda:0`, ...)', default='cuda:0')
    args = parser.parse_args()

    
    modality_dirs = dnc.list_of_modality_dirs(INPUT_DIR)

    slices_pt = dnc.read_slices_from_dir(modality_dirs['PT'])
    # get reference header for metadata
    ref_dcm_pt = slices_pt[0]



    # print(modality_dirs)
    if not os.path.exists('./TMP_INPUT/PT/'):
        os.makedirs('./TMP_INPUT/PT/')
    if not os.path.exists('./TMP_INPUT/CT/'):
        os.makedirs('./TMP_INPUT/CT/')
    # if not os.path.exists('./TMP_OUTPUT/'):
    #     os.makedirs('./TMP_OUTPUT/')

    # loop through all the files and subdirectories in the folder
    for root, dirs, files in os.walk('./TMP_INPUT/'):
        for file in files:
            # create the full path to the file
            file_path = os.path.join(root, file)
            # delete the file
            os.remove(file_path)



    pt_nifti_path = dnc.dicomToNifti(modality_dirs['PT'], './TMP_INPUT/PT')
    ct_nifti_path = dnc.dicomToNifti(modality_dirs['CT'], './TMP_INPUT/CT')

    ct_path = './TMP_INPUT/CT'
    pt_path = './TMP_INPUT/PT'
    model_large_fov_path = './src/models/e_latest.model'
    model_refiner_path = './src/models/r_latest.model'
    output_path = OUTPUT_DIR

    print(f'starting...\nPID={os.getpid()}')
    is_available = torch.cuda.is_available()
    if is_available and len(args.device) > 0:
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    print('Device=', device)

    print('Model loading...')
    # hmm seems like there is some shared references between the
    # models corrupting the results. Make a deep copy
    models = [copy.deepcopy(ModelLargeFov()) for c in range(4)]
    model_ensemble = ModelEnsemble(models)
    TrainerV2.load_state(
        model_ensemble, 
        model_large_fov_path, 
        device=device, 
        strict=True
    )
    model_ensemble = model_ensemble.to(device)

    model_refiner = ModelRefiner()
    TrainerV2.load_state(
        model_refiner, 
        model_refiner_path, 
        device=device, 
        strict=True
    )
    model_refiner = model_refiner.to(device)

    print('Inference...')
    seg_sitk, seg_output_path = inference_fn(
        ct_path, 
        pt_path,
        modality_dirs['ID'],
        output_path, 
        model_ensemble, 
        model_refiner
    )

    o_dict = debug_output(seg_sitk, seg_output_path, modality_dirs['ID'], modality_dirs['PT'])

    # Markdown   
    raiven = Raiven()

    # nii_file_pet = os.path.join(preprocessing_data_dir, patientid, 'study1','SUV.nii.gz')
    nii_file_pet = pt_nifti_path
    nii_pet_Coronal = get_MIP_from_3Dnifti(nii_file_pet, 0)
    nii_pet_Sagittal = get_MIP_from_3Dnifti(nii_file_pet, 1)
    

    # # Load the DICOM data from the NIFTI image file
    # dicom_data = pydicom.dcmread(nii_file_pet, force=True)

    # # Get the patient name from the DICOM data
    # patient_name = dicom_data.PatientName

    # print(patient_name)


    # nii_file_seg = os.path.join(preprocessing_data_dir, patientid, 'study1', 'SEG.nii.gz')
    # nii_file_seg = '/home/azureuser/AutoPETGIT/ToDocker_v3/TMP_INPUT/CT/15-20335_CT_20150508.nii.gz'
    # nii_seg_Coronal= get_MIP_from_3Dnifti(nii_file_seg, 0)
    # nii_seg_Sagittal= get_MIP_from_3Dnifti(nii_file_seg, 1)


    # nii_file_predict = os.path.join(preprocessing_data_dir, patientid, 'study1', patientid+'.nii')
    nii_file_predict = seg_output_path
    nii_predict_Coronal= get_MIP_from_3Dnifti(nii_file_predict, 0)
    nii_predict_Sagittal= get_MIP_from_3Dnifti(nii_file_predict, 1)
    

    # for identifier, data_path in zip([patientid,], [os.path.join(preprocessing_data_dir, patientid),]):
    #         print('identifier',identifier)
    #         print('data_path',data_path)


    # csv_file = ComputesTMTVsDmaxFromNii(data_path=data_path, get_identifier=identifier)
    # csv_file = ComputesTMTVsDmaxFromNii(data_path='./TMP_INPUT', get_identifier='identifierggg')
    # print('csv_file:',csv_file)
    # data_dict = csv_file.compute_and_save_surrogate_features()

   # Build the report

    patient_name = str(ref_dcm_pt.PatientName)
    patient_id = str(ref_dcm_pt.PatientID)
    patient_sex = ref_dcm_pt.PatientSex
    # patient_identity_removed = ref_dcm_pt.PatientIdentityRemoved
    patient_weight = float(ref_dcm_pt.PatientWeight)
    patient_dob = ref_dcm_pt.PatientBirthDate[0:4]+'/'+ref_dcm_pt.PatientBirthDate[4:6]+'/'+ref_dcm_pt.PatientBirthDate[6:]

    scan_date = ref_dcm_pt.AcquisitionDate[0:4]+'/'+ref_dcm_pt.AcquisitionDate[4:6]+'/'+ref_dcm_pt.AcquisitionDate[6:] #'04/12/2006'
    scan_injected_dose = float(ref_dcm_pt.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
    

    # image_file_name = "ensambled_"+patientid+".png"
    image_file_name = "ensambled_.png"
    # filename_figure = os.path.join(OUTPUT_DIR, image_file_name)
    filename_figure = os.path.join(OUTPUT_DIR, image_file_name)


    plt.rcParams.update({'font.size': 5})
    hf = plt.figure(dpi=150)
    a = plt.subplot(2,2,1)
    plt.imshow(np.rot90(nii_pet_Coronal, 1),vmin=0,vmax=16)
    plt.title('Input PET image (SUV) \n Coronal projection')
    plt.colorbar(fraction=0.046, pad=0.04)
    a.set_aspect('equal')

    a = plt.subplot(2,2,2)
    plt.imshow(np.rot90(nii_pet_Sagittal, 1),vmin=0,vmax=16)
    plt.title('Input PET image (SUV) \n Sagittal projection')
    plt.colorbar(fraction=0.046, pad=0.04)
    a.set_aspect('equal')
    a.axis('off')

    a = plt.subplot(2,2,3)
    plt.imshow(np.rot90(nii_predict_Coronal, 1),vmin=0,vmax=1,cmap='gray')
    plt.title('Lesion segmentation \n Coronal projection')
    plt.colorbar(fraction=0.046, pad=0.04)
    a.set_aspect('equal')

    a = plt.subplot(2,2,4)
    plt.imshow(np.rot90(nii_predict_Sagittal, 1),vmin=0,vmax=1,cmap='gray')
    plt.title('Lesion segmentation \n Sagittal projection')
    plt.colorbar(fraction=0.046, pad=0.04)
    a.set_aspect('equal')
    a.axis('off')



    # a = plt.subplot(3,2,5)
    # plt.imshow(np.rot90(nii_seg_Coronal, 1),vmin=0,vmax=1,cmap='gray')
    # plt.title('Grand Truth \n Coronal projection')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # a.set_aspect('equal')

    # a = plt.subplot(3,2,6)
    # plt.imshow(np.rot90(nii_seg_Sagittal, 1),vmin=0,vmax=1,cmap='gray')
    # plt.title('Grand Truth \n Sagittal projection')
    # plt.colorbar(fraction=0.046, pad=0.04)
    # a.set_aspect('equal')


    # hf.tight_layout()
    plt.subplots_adjust(wspace=-0.5, hspace=0.4)
    plt.savefig(filename_figure,bbox_inches='tight')


    # Set header properties

    mdFile = MdUtils(file_name="output")

    tz = pytz.timezone('America/Los_Angeles')
    now = datetime.datetime.now(tz)

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    mdFile.new_line("___________________________")
    mdFile.new_line("RAIVEN AUTOMATED REPORT")
    mdFile.new_line("Generated on (d/m/y): " + dt_string)
    mdFile.new_line("___________________________")
    mdFile.new_header(level=1, title="Module: TMTVNet")
    mdFile.new_paragraph(
        " "
        "Total metabolic tumor volume quantification in Lymphoma PET/CT Images"
        " "
    )
    mdFile.new_paragraph(
        "**Module version**: 0.1.0"
        " "
    )
    mdFile.new_paragraph(
        "**Module Author**: Fereshteh Yousefirizi"
        " "
    )
    mdFile.new_line("___________________________")
    mdFile.new_header(level=2, title="Patient Data")
    mdFile.new_line("**Patient name:** " + patient_name)
    mdFile.new_line("**Patient ID:** " + patient_id)
    mdFile.new_line("**Date of birth:** " + patient_dob)
    mdFile.new_line("**Patient sex:** " + patient_sex)
    mdFile.new_line("**Weight:** " + str(patient_weight))
    # mdFile.new_line("**Anonymized:** " + patient_identity_removed)
    mdFile.new_line("")
    mdFile.new_header(level=2, title="Scan Data")
    mdFile.new_line("**Scan date:** " + scan_date)
    mdFile.new_line("**Injected dose (MBq):** " + str(scan_injected_dose/pow(10,6)))
    mdFile.new_line("**Patient weight (kg):** " + str(patient_weight))
    mdFile.new_line("")
    mdFile.new_header(level=2, title="Autosegmentation results")
    mdFile.new_line("")
    mdFile.new_line("![explicit image ref](" + filename_figure + " 'Projections')")
    mdFile.new_line("___________________________")
    mdFile.new_header(level=2, title="Lesion measurements")
    # mdFile.new_line("**sTMTV sagittal**: " + data_dict['sTMTV_sagittal'])
    # mdFile.new_line("**sTMTV coronal**: " + data_dict['sTMTV_coronal'])
    mdFile.new_line("**TMTV (ml)**: " + str(round(o_dict['TMTV']  / 1000, 3)))
    # mdFile.new_line("**Sagittal xy**: " + data_dict['Sagittal_xy'])
    # mdFile.new_line("**Sagittal z**: " + data_dict['Sagittal_z'])
    # mdFile.new_line("**Coronal xy**: " + data_dict['Coronal_xy'])
    # mdFile.new_line("**Coronal z**: " + data_dict['Coronal_z'])
    
    # mdFile.new_line("**Number of regions**: " + str(o_dict['Seg_regions']))
    mdFile.new_line("**Dmax (mm)**: " + str(round(o_dict['Dmax'], 3)))
    # mdFile.new_line("**sDmax (mm) euclidean**: " + data_dict['sDmax_(mm)_euclidean'])
    mdFile.new_line("")
    # mdFile.create_md_file()

    raiven.savemd(mdFile, "output.md")






    print('Done!')


    
