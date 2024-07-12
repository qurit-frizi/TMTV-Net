import os
import pydicom
import pandas as pd
import gzip
import shutil
import json
import csv
# from rt_utils import RTStructBuilder
import numpy as np
import SimpleITK as sitk
# from helper import winapi_path, bqml_to_suv
from datetime import datetime


import platform
import dateutil

def winapi_path(dos_path, encoding=None):
    path = os.path.abspath(dos_path)
    if platform.system() == 'Windows':
        if path.startswith("\\\\"):
            path = "\\\\?\\UNC\\" + path[2:]
        else:
            path = "\\\\?\\" + path

    return path


def bqml_to_suv(dcm_file: pydicom.FileDataset) -> float:
    '''
    Calculates the conversion factor from Bq/mL to SUV bw [g/mL] using 
    the dicom header information in one of the images from a dicom series
    '''
    # TODO: You can access these attributes in a more user friendly way rather
    # than using the codes...change this at some point
    nuclide_dose = dcm_file[0x054, 0x0016][0][0x0018, 0x1074].value  # Total injected dose (Bq)
    weight = dcm_file[0x0010, 0x1030].value  # Patient weight (Kg)
    half_life = float(dcm_file[0x054, 0x0016][0][0x0018, 0x1075].value)  # Radionuclide half life (s)

    parse = lambda x: dateutil.parser.parse(x)

    series_time = str(dcm_file[0x0008, 0x0031].value)  # Series start time (hh:mm:ss)
    series_date = str(dcm_file[0x0008, 0x0021].value)  # Series start date (yyy:mm:dd)
    series_datetime_str = series_date + ' ' + series_time
    series_dt = parse(series_datetime_str)

    nuclide_time = str(dcm_file[0x054, 0x0016][0][0x0018, 0x1072].value)  # Radionuclide time of injection (hh:mm:ss)
    nuclide_datetime_str = series_date + ' ' + nuclide_time
    nuclide_dt = parse(nuclide_datetime_str)

    delta_time = (series_dt - nuclide_dt).total_seconds()
    decay_correction = 2 ** (-1 * delta_time / half_life)
    suv_factor = (weight * 1000) / (decay_correction * nuclide_dose)

    return suv_factor


def list_of_modality_dirs(input_dir):
    # scan the input dir for directories containing DICOM series
    # determine which directory corresponds to which modality
    dir_list = next(os.walk(input_dir))[1]
    for direc in dir_list:
        file_list = os.listdir(os.path.join(input_dir, direc))
        for file in file_list:
            filename = os.path.join(input_dir, direc, file)
            if filename.endswith(".dcm"):
                ds = pydicom.read_file(filename)
                Patient_ID = ds.PatientID
                if ds.Modality == 'PT':
                    print('Found PET DIR')
                    pt_dir = os.path.join(input_dir, direc)
                    break
                elif ds.Modality == 'CT':
                    print('Found CT DIR')
                    ct_dir = os.path.join(input_dir, direc)
                    break
                else:
                    print('Found dir without PET or CT')
                    break
    modality_dirs = {'PT': pt_dir, 'CT': ct_dir, 'ID': Patient_ID}
    return modality_dirs


def read_slices_from_dir(input_dir):
    # read and sort .dcm slices from a directory
    # first, read all dicom files
    dicom_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".dcm"):
                filename_full = os.path.join(root, file)
                ds = pydicom.read_file(filename_full)
                dicom_files.append(ds)
    # second, only choose files that have 'location' attribure, and sort
    slices = []
    skipcount = 0
    # only include dicom files that represent image slices
    for f in dicom_files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount += 1
    # print('Skipped {} files'.format(skipcount))
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    return slices


def dicomToNifti(seriesDir, savePath):
    # converts DICOM series in the seriesDir to NIFTI image in the savePath specified
    dicom_files = read_slices_from_dir(seriesDir)

    if not dicom_files:
        raise RuntimeError("No DICOM files found in the specified directory.")

    traits = {
        "Patient ID": getattr(dicom_files[0], 'PatientID', None),
        "Patient's Sex": getattr(dicom_files[0], 'PatientSex', None),
        "Patient's Age": getattr(dicom_files[0], 'PatientAge', None),
        "Patient's Birth Date": getattr(dicom_files[0], 'PatientBirthDate', None),
        "Patient's Weight": getattr(dicom_files[0], 'PatientWeight', None),
        "Institution Name": getattr(dicom_files[0], 'InstitutionName', None),
        "Referring Physician's Name": getattr(dicom_files[0], 'ReferringPhysicianName', None),
        "Operator's Name": getattr(dicom_files[0], 'OperatorsName', None),
        "Study Date": getattr(dicom_files[0], 'StudyDate', None),
        "Study Time": getattr(dicom_files[0], 'StudyTime', None),
        "Modality": getattr(dicom_files[0], 'Modality', None),
        "Series Description": getattr(dicom_files[0], 'SeriesDescription', None),
        "Dimensions": np.array(getattr(dicom_files[0], 'pixel_array', None)).shape,
    }

    reader = sitk.ImageSeriesReader()
    seriesNames = [ds.filename for ds in dicom_files]
    reader.SetFileNames(seriesNames)
    image = reader.Execute()

    if traits["Modality"] == 'PT':
        suv_factor = bqml_to_suv(dicom_files[0])  # Calculate SUV factor using the first DICOM file

        # Apply Rescale Slope and Intercept for each slice
        for slice_file in dicom_files:
            rescale_slope = slice_file[0x0028, 0x1053].value
            rescale_intercept = slice_file[0x0028, 0x1052].value

            slice_index = dicom_files.index(slice_file)
            image_slice = sitk.VectorIndexSelectionCast(image, slice_index)

            image_slice = sitk.Multiply(image_slice, rescale_slope)
            image_slice = sitk.Add(image_slice, rescale_intercept)
            image_slice = sitk.Multiply(image_slice, suv_factor)

            image = sitk.Paste(image, image_slice, image_slice.GetSize(), image_slice.GetIndex(), image_slice.GetIndex())

    sitk.WriteImage(image, os.path.join(savePath, f'{traits["Patient ID"]}_{traits["Modality"]}_{traits["Study Date"]}.nii.gz'), imageIO='NiftiImageIO')
    return os.path.join(savePath, f'{traits["Patient ID"]}_{traits["Modality"]}_{traits["Study Date"]}.nii.gz')
