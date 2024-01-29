from typing import Dict, Sequence, Tuple
import SimpleITK as sitk
import numpy as np


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


def get_sitk_image_attributes(sitk_image: sitk.Image) -> Dict:
    """Get physical space attributes (meta-data) of the image."""
    attributes = {}
    attributes['pixelid'] = sitk_image.GetPixelIDValue()
    attributes['origin'] = sitk_image.GetOrigin()
    attributes['direction'] = sitk_image.GetDirection()
    attributes['spacing'] = np.array(sitk_image.GetSpacing())
    attributes['shape'] = np.array(sitk_image.GetSize(), dtype=np.int)
    return attributes


def make_sitk_image_attributes(
        shape_xyz: Sequence[int],
        spacing_xyz: Sequence[float], 
        origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        direction_xyz=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        pixel_id: int = sitk.sitkFloat32) -> Dict:
    """Create image attributes"""
    attributes = {}
    attributes['pixelid'] = pixel_id
    attributes['origin'] = tuple(np.array(origin_xyz, dtype=float))
    attributes['direction'] = tuple(np.array(direction_xyz, dtype=float))
    attributes['spacing'] = tuple(np.array(spacing_xyz, dtype=float))
    attributes['shape'] = tuple(np.array(shape_xyz, dtype=int))
    return attributes


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