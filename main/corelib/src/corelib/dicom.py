from typing import Any, Dict, Tuple
import pydicom
import SimpleITK as sitk
import numpy as np
import os


def reorder_slices_single_series(dicom_names):
    if len(dicom_names) == 1:
        return dicom_names
    # reorder the slices as this will influence the orientation
    # of the image :(
    dcms = [pydicom.read_file(f) for f in dicom_names]
    dcms_instance_number = [s.InstanceNumber for s in dcms]
    sorted_instance_number = np.argsort(dcms_instance_number)[::-1]
    return np.asarray(dicom_names)[sorted_instance_number]
    

def read_dicom(path: str) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Read folder containing a single DICOM series and reconstruct it as a 3D volume

    If path is a filepath, reconstruct a single DICOM file (e.g., 3D SPECT may be 
    stored as a single DICOM file)
    """
    reader = sitk.ImageSeriesReader()
    if os.path.isdir(path):
        # VERY VERY IMPORTANT: the slice order MATTERS!!!!!
        # https://stackoverflow.com/questions/41037407/itk-simpleitk-dicom-series-loaded-in-wrong-order-slice-spacing-incorrect
        # so we need to reorder them :(
        series_ids = reader.GetGDCMSeriesIDs(path)
        assert len(series_ids) == 1, f'Expected single DICOMs series in a folder! Got={len(series_ids)}'
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        assert len(dicom_names) > 0, 'foler does not contain DICOM files!'
        dicom_names = reorder_slices_single_series(dicom_names)
    else:
        # single DICOM (e.g., SPECT)
        dicom_names = [path]
        
    reader.SetFileNames(dicom_names)
    #reader.SetFileNames(dicom_names)
    dicom_tags = {'TODO', 'RETURN THE TAGS!'}
    image = reader.Execute()
    return image, dicom_tags
