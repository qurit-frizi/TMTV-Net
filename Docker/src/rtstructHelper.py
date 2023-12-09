import random
import nibabel as nib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import measure

import pydicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

def concatenate_coordinates(coordinates_x, coordinates_y, coordinates_z):

    vector = np.zeros((len(coordinates_x)*3,1))

    for i in range(len(coordinates_x)):
        vector[i*3+0] = coordinates_x[i] 
        vector[i*3+1] = coordinates_y[i]
        vector[i*3+2] = coordinates_z[i]

    return vector

def find_first_slice_position(dcms):
    
    patientStartingZ = 0 
    for idx, dcm in enumerate(dcms):
        ds = pydicom.dcmread(dcm, stop_before_pixels=True)
        if not 'ImagePositionPatient' in ds or ds.ImagePositionPatient is None:
            continue
        if ds.ImagePositionPatient[2] <= patientStartingZ or idx==0:
            patientStartingZ = ds.ImagePositionPatient[2]
    
    return patientStartingZ

def convert(input_nifti_path: str, input_dicom_path: str, output_dicom_path: str):
    
    #---------------
    # First DICOM part
    #---------------

    # Get number of DICOM files in DICOM path
    dicomFiles = next(os.walk(input_dicom_path))[2] 
    numberOfDicomImages = len(dicomFiles)
    numberOfROIs = 1   # The whole volume is 1 ROI, assuming 1 tumour per patient
    
    # Load template DICOM file header (first file)
    ds = pydicom.dcmread(os.path.join(input_dicom_path, "%s"%dicomFiles[0]),stop_before_pixels=True) 

    xPixelSize = ds.PixelSpacing[0]
    yPixelSize = ds.PixelSpacing[1]
    zPixelSize = ds.SliceThickness
    
    print("Each voxel is ",xPixelSize," x ",yPixelSize," x ",zPixelSize)

    # Find position of first slice
    patientPosition = ds.ImagePositionPatient
    patientStartingZ = find_first_slice_position([os.path.join(input_dicom_path, '%s'%_) for _ in dicomFiles])
    
    print('Patient position is ', patientPosition[:2])
    print('First slice at ', patientStartingZ)

    #---------------
    # NIFTI part
    #---------------

    # Load nifti volume
    nii = nib.load(input_nifti_path)
    volume = nii.get_fdata()
    volume = volume.astype(float)

    AllCoordinates = []

    if len(volume.shape)==4: 
        volume = volume[...,0]
        print('Assuming the first channel of the input nifti is the seg mask.')
    elif len(volume.shape)==3:
        print('Segmentation mask is same size of the patient image volume.')
    else:
        print('Dimension not supported.')
            
    # Loop over slices in volume, get contours for each slice
    for slice in range(volume.shape[2]):
        
        AllCoordinatesThisSlice = []

        image = volume[:,:,slice] 
        
        # Get contours in this slice using scikit-image
        contours = measure.find_contours(image, 0.5)
        
        # Save contours for later use
        for n, contour in enumerate(contours):
            #print("n is ",n,"for slice ",slice)
            nCoordinates = len(contour[:,0])
            #print("number of coordinates is ",len(contour[:,0])*3," for contour ",n," for slice ",slice)
            zcoordinates = slice * np.ones((nCoordinates,1)) 
            
            # Add patient position offset
            reg_contour = np.append(contour, zcoordinates, -1)
            # Assume no other orientations for simplicity
            reg_contour[:,0] = reg_contour[:,0] * xPixelSize + patientPosition[0]
            reg_contour[:,1] = reg_contour[:,1] * yPixelSize + patientPosition[1]
            reg_contour[:,2] = reg_contour[:,2] * zPixelSize + patientStartingZ

            # Storing coordinates as mm instead of as voxels
            #coordinates = concatenate_coordinates(contour[:,0] * xPixelSize, contour[:,1] * yPixelSize, zcoordinates * zPixelSize)
            coordinates = concatenate_coordinates(*reg_contour.T)
            coordinates = np.squeeze(coordinates)
            
            AllCoordinatesThisSlice.append(coordinates)

        AllCoordinates.append(AllCoordinatesThisSlice)
    
    #print("All coordinates has length ",len(AllCoordinates))
    #print("All coordinates slice 0 has length ",len(AllCoordinates[0]))
    #print("All coordinates slice 1 has length ",len(AllCoordinates[1]))
    #print("All coordinates slice 1 contour 1 has length ",len(AllCoordinates[1][1]))
    #print("Coordinates are ",AllCoordinates[1][1])

    #---------------
    # Second DICOM part (RTstruct)
    #---------------

    # Referenced Frame of Reference Sequence
    refd_frame_of_ref_sequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence = refd_frame_of_ref_sequence

    # Referenced Frame of Reference Sequence: Referenced Frame of Reference 1
    refd_frame_of_ref1 = Dataset()
    refd_frame_of_ref1.FrameOfReferenceUID = ds.FrameOfReferenceUID # '1.3.6.1.4.1.9590.100.1.2.138467792711241923028335441031194506417'

    # RT Referenced Study Sequence
    rt_refd_study_sequence = Sequence()
    refd_frame_of_ref1.RTReferencedStudySequence = rt_refd_study_sequence

    # RT Referenced Study Sequence: RT Referenced Study 1
    rt_refd_study1 = Dataset()
    rt_refd_study1.ReferencedSOPClassUID = ds.SOPClassUID # '1.2.840.10008.5.1.4.1.1.481.3'
    rt_refd_study1.ReferencedSOPInstanceUID = ds.SOPInstanceUID # '1.3.6.1.4.1.9590.100.1.2.201285932711485367426568006803977990318'

    # RT Referenced Series Sequence
    rt_refd_series_sequence = Sequence()
    rt_refd_study1.RTReferencedSeriesSequence = rt_refd_series_sequence

    # RT Referenced Series Sequence: RT Referenced Series 1
    rt_refd_series1 = Dataset()
    rt_refd_series1.SeriesInstanceUID = ds.SeriesInstanceUID  # '1.3.6.1.4.1.9590.100.1.2.170217758912108379426621313680109428629'

    # Contour Image Sequence
    contour_image_sequence = Sequence()
    rt_refd_series1.ContourImageSequence = contour_image_sequence

    # Loop over all DICOM images
    for image in range(1,numberOfDicomImages+1):
        dstemp = pydicom.dcmread(os.path.join(input_dicom_path, "%s"%dicomFiles[image-1]),stop_before_pixels=True) 
        # Contour Image Sequence: Contour Image
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = dstemp.SOPClassUID      # '1.2.840.10008.5.1.4.1.1.2'
        contour_image.ReferencedSOPInstanceUID = dstemp.SOPInstanceUID # '1.3.6.1.4.1.9590.100.1.2.257233736012685791123157667031991108836'
        contour_image_sequence.append(contour_image)

    rt_refd_series_sequence.append(rt_refd_series1)
    rt_refd_study_sequence.append(rt_refd_study1)
    refd_frame_of_ref_sequence.append(refd_frame_of_ref1)

    # Structure Set ROI Sequence
    structure_set_roi_sequence = Sequence()
    ds.StructureSetROISequence = structure_set_roi_sequence

    # Loop over ROIs
    for ROI in range(1,numberOfROIs+1):
        # Structure Set ROI Sequence: Structure Set ROI
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = str(ROI)
        structure_set_roi.ReferencedFrameOfReferenceUID = ds.FrameOfReferenceUID # '1.3.6.1.4.1.9590.100.1.2.138467792711241923028335441031194506417'
        structure_set_roi.ROIName = 'ROI_' + str(ROI)
        structure_set_roi.ROIGenerationAlgorithm = 'AndersNicePythonScript'
        structure_set_roi_sequence.append(structure_set_roi)

    # ROI Contour Sequence
    roi_contour_sequence = Sequence()
    ds.ROIContourSequence = roi_contour_sequence

    # Loop over ROI contour sequences
    for ROI in range(1,numberOfROIs+1):

        # ROI Contour Sequence: ROI Contour 1
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = [0, 230, 0]

        # Contour Sequence
        contour_sequence = Sequence()
        roi_contour.ContourSequence = contour_sequence

        # Loop over slices in volume (ROI)
        for slice in range(volume.shape[2]):

            # Should Contour Sequence be inside this loop?
            #contour_sequence = Sequence()
            #roi_contour.ContourSequence = contour_sequence

            # Loop over contour sequences in this slice
            numberOfContoursInThisSlice = len(AllCoordinates[slice])
            for c in range(numberOfContoursInThisSlice):

                currentCoordinates = AllCoordinates[slice][c]

                # Contour Sequence: Contour 1
                contour = Dataset()

                # Contour Image Sequence
                contour_image_sequence = Sequence()
                contour.ContourImageSequence = contour_image_sequence

                # Load the corresponding dicom file to get the SOPInstanceUID
                dstemp = pydicom.dcmread(os.path.join(input_dicom_path, "%s"%dicomFiles[slice]),stop_before_pixels=True) 

                # Contour Image Sequence: Contour Image 1
                contour_image = Dataset()
                contour_image.ReferencedSOPClassUID = dstemp.SOPClassUID  # '1.2.840.10008.5.1.4.1.1.2'
                contour_image.ReferencedSOPInstanceUID = dstemp.SOPInstanceUID # '1.3.6.1.4.1.9590.100.1.2.76071554513024464020636223132290799275'
                contour_image_sequence.append(contour_image)

                contour.ContourGeometricType = 'CLOSED_PLANAR'
                contour.NumberOfContourPoints = len(currentCoordinates)
                contour.ContourData = currentCoordinates.tolist()
                contour_sequence.append(contour)

        roi_contour.ReferencedROINumber = ROI
        roi_contour_sequence.append(roi_contour)


    # RT ROI Observations Sequence
    rtroi_observations_sequence = Sequence()
    ds.RTROIObservationsSequence = rtroi_observations_sequence

    # Loop over ROI observations
    for ROI in range(1,numberOfROIs+1):
        # RT ROI Observations Sequence: RT ROI Observations 1
        rtroi_observations = Dataset()
        rtroi_observations.ObservationNumber = str(ROI)
        rtroi_observations.ReferencedROINumber = str(ROI)
        rtroi_observations.ROIObservationLabel = ''
        rtroi_observations.RTROIInterpretedType = ''
        rtroi_observations.ROIInterpreter = ''
        rtroi_observations_sequence.append(rtroi_observations)
   
    # Add RTSTRUCT specifics
    ds.Modality = 'RTSTRUCT' # So the software can recognize RTSTRUCT
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' # So the software can recognize RTSTRUCT
    
    random_str_1 = "%0.8d" % random.randint(0,99999999)
    random_str_2 = "%0.8d" % random.randint(0,99999999)
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.2.1125."+random_str_1+".1"+random_str_2 # Just some random UID
    
    RTDCM_name = os.path.join(output_dicom_path, "segmentationRTSTRUCT.dcm")
    ds.save_as(RTDCM_name)
    print('RTSTRUCT saved as %s'%RTDCM_name)
    
def get_parser():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Convert nifti images to RTSTRUCT file')

    # Positional arguments.
    parser.add_argument("input_nifti", help="Path to input NIFTI image")
    parser.add_argument("input_dicom", help="Path to input DICOM images")
    parser.add_argument("output_dicom", help="Path to output DICOM image")
    return parser.parse_args()

if __name__ == "__main__":
    p = get_parser()
    print(p.input_nifti)
    convert(p.input_nifti, p.input_dicom, p.output_dicom)