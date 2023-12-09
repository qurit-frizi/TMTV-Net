# Collection of functions for applying PSMA-Hornet

import os
import numpy as np
from rt_utils import RTStructBuilder # this also needs "pip install opencv-python"
#from scipy.ndimage.measurements import label
import scipy.io as sio
import pydicom 
from pydicom import dcmread
import scipy.io as sio
from scipy.ndimage import zoom
import math
import SimpleITK as sitk
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dropout, concatenate, Softmax, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.layers import AveragePooling2D

import tensorflow.keras.backend as K
import nibabel as nib

activ = tf.keras.layers.LeakyReLU(alpha=0.1)

INPUT1_SHAPE_2p5D = (192,192,3)
INPUT2_SHAPE_2p5D = (192,192,3)
OUTPUT_SHAPE_2p5D = (192,192,1)
OPTIMIZER = tf.keras.optimizers.Adam(lr=1e-5,epsilon=1e-7)

def write_nifti(sitk_img, voxel_spacing, path):
    """Save a SimpleITK Image to disk in NIfTI format."""
    image = sitk.GetImageFromArray(sitk_img)
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(image)
    # re-save to update the header info
    img = nib.load(str(path))
    header = img.header
    header['pixdim'][1:4] = voxel_spacing
    affine = np.diag([-1, -1, 1, 1])
    new_img = nib.Nifti1Image(sitk_img, affine, header)
    nib.save(new_img, path)

def read_nifti(path):
    """Read nifti image into a numpy array"""
    # re-save to update the header info
    img = nib.load(str(path))
    print(type(img))
    #header = img.header
    img_data = np.array(img.get_fdata())
    return img_data

def imageReaderMat(filenameFull):
    # reading n-d data from matlab file
    varName = 'image'  # name of struct/dict field
    FA_org = sio.loadmat(filenameFull)
    img_data = FA_org[varName]
    return img_data

def MIP(img3D,axisNum=1):
    # plot maximum intensity projection of a 3D image
    img2D = np.max(img3D, axis=axisNum)
    if axisNum==1:
        img2D = np.rot90(img2D,k=-1)
    return img2D

def read_mat_images_from_dir(mat_dir,shape,suffix):
    # run predictions on all files in the list and return list of 3D images
    file_list = next(os.walk(mat_dir))[2]
    file_list = [file for file in file_list if file.endswith(suffix)]
    print('Number of slices to be loaded:'+ str(len(file_list)))
    file_list.sort()
    # allocate 3D volume
    num_planes = len(file_list)
    img_3D = np.zeros((shape[0], shape[1], num_planes),dtype='float32')
    plane_count = 0
    for file in file_list:
        plane_count += 1
        fullFileName = os.path.join(mat_dir, file)
        img = imageReaderMat(fullFileName)
        img_3D[:,:,plane_count-1] = img
    return img_3D

def volume_from_slices_with_scaling(slices, apply_scaling):
    # build volume image from slices, and apply correct scaling
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)

    for i, s in enumerate(slices):
        img2d = s.pixel_array # pixel value data (unscaled)
        if apply_scaling == True:
            img2d = s.RescaleIntercept + img2d * s.RescaleSlope # collect scaling
        img3d[:,:,i] = img2d
    return img3d

def convert_activity_to_suv(img_act, hdr):
    '''Returns an image in units of SUV based on body weight.This function is based on the calculation described in the Quantitative Imaging Biomarkers Alliance for the Vendor-neutral pseudo-code for SUV Calculation - extracted "happy path only". http://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)
    INPUT: 
            img_act: (numpy ndarray)
                voxel array in activity values
            hdr: (object)
                the header of the DICOM slice
    OUTPUT: suvbw_img: (numpy.ndarray) 
            matrix that contains the pixel information in "SUVbw" units (i.e. (Bq/ml)/(Bq/g)
    '''
    bw = float(hdr.PatientWeight) * 1000 # weight in grams
    print("weight:"+str(bw))
    if ('ATTN' in hdr.CorrectedImage and 'DECY' in hdr.CorrectedImage) and hdr.DecayCorrection == 'START':
        if hdr.Units == 'BQML':
            # seconds (0018,1075)
            half_life = hdr.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            scan_time = hdr.SeriesTime  # (0008,0031)
            # (0018,1072)
            start_time = hdr.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            # convert tref and injtime from strings to datetime
            scan_time = datetime.strptime(scan_time.split('.')[0], '%H%M%S')
            start_time = datetime.strptime(start_time.split('.')[0], '%H%M%S')
            decay_time = scan_time - start_time
            # (18,1074)
            inj_act = hdr.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            decayed_act = inj_act * 2**(-decay_time.total_seconds()/half_life)
            SUVbw_scale_factor = bw/decayed_act
            img_suv = img_act*SUVbw_scale_factor
            return img_suv

def labelmap_to_rtstruct(lab3d, dicom_series_path, label_dict):
    # convert label map into RTSTRUCT
    rtstruct = RTStructBuilder.create_new(dicom_series_path=dicom_series_path)
    rtstruct.add_roi(
            mask=mask3d_sel,
            color=[255, 140, 0],
            name=ROI_NAME
        )
    return rtstruct

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
                if ds.Modality == 'PT':
                    print('Found PET DIR')
                    pt_dir = os.path.join(input_dir,direc)
                    break
                elif ds.Modality == 'CT':
                    print('Found CT DIR')
                    ct_dir = os.path.join(input_dir,direc)
                    break
                else:
                    print('Found dir without PET or CT')
                    break
    modality_dirs = {'PT':pt_dir,'CT':ct_dir}
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
    #print('Skipped {} files'.format(skipcount))
    slices = sorted(slices,key=lambda s: s.SliceLocation)
    return slices

def orig_to_model_space(img3d_pt, img3d_ct, dcm_pt, dcm_ct):
    # native model pixel size: 3.6458332538605 mm x 3.6458332538605 mm
    # 192x192 matrix size
    pixel_size_ref = [3.6458332538605, 3.6458332538605, 3.6458332538605] # in mm
    pixel_size_pt = [dcm_pt.PixelSpacing[0], dcm_pt.PixelSpacing[1], dcm_pt.SliceThickness]
    pixel_size_ct = [dcm_ct.PixelSpacing[0], dcm_ct.PixelSpacing[1], dcm_ct.SliceThickness]
    ct_scale_x = pixel_size_ct[0]/pixel_size_ref[0]
    ct_scale_y = pixel_size_ct[1]/pixel_size_ref[1]
    print('X scale for CT:' + str(ct_scale_x))

    img3d_ct_resamp = zoom(img3d_ct, (ct_scale_x, ct_scale_y, 1), order=2)
    ct_new_size = img3d_ct_resamp.shape
    print('New size for CT:' + str(ct_new_size))
    pt_new_size = [192, 192]

    pad_d1_post = math.floor((pt_new_size[0] - ct_new_size[0])/2);
    pad_d1_pre = pt_new_size[0] - ct_new_size[0] - pad_d1_post;

    pad_d2_post = math.floor((pt_new_size[1] - ct_new_size[1])/2);
    pad_d2_pre = pt_new_size[1] - ct_new_size[1] - pad_d2_post;

    pad_size = ((pad_d1_pre,pad_d1_post),(pad_d2_pre,pad_d2_post),(0,0))
    img3d_ct_new = np.pad(img3d_ct_resamp,pad_size,mode='constant')

    print('Old CT dims:' + str(ct_new_size))
    print('New CT dims:' + str(img3d_ct_new.shape))

    #img3d_ct_new = np.flip(img3d_ct_new,axis=2)
    img3d_ct_new = np.flip(img3d_ct_new,axis=1)
    img3d_ct_new = np.rot90(img3d_ct_new,1)

    #img3d_pt_new = np.flip(img3d_pt,axis=2)
    img3d_pt_new = np.flip(img3d_pt,axis=1)
    img3d_pt_new = np.rot90(img3d_pt_new,1)
    return img3d_pt_new, img3d_ct_new
    #new_array = zoom(array, (0.5, 0.5, 2))

def preprocess_input_pt(img3d_pt_suv):
    # do pre-processing of PET voxel values
    # input must be in SUV units
    img3d_pt_suv[img3d_pt_suv<0] = 0
    img3d_pt_suv = img3d_pt_suv/10
    return img3d_pt_suv

def preprocess_input_ct(img3d_ct_hu):
    # do pre-processing of CT voxel values
    # input must be in HU
    img3d_ct_hu[img3d_ct_hu>1200] = 1200
    img3d_ct_hu = img3d_ct_hu-800
    img3d_ct_hu[img3d_ct_hu<0] = 0
    img3d_ct_hu = img3d_ct_hu/400
    return img3d_ct_hu

def makeModelPredictions_multislice(model, img3d_input1, img3d_input2):
    # run predictions on the provided numpy volumes
    input_blob = dataBlobGenerator()
    input_blob.loadData(img3d_input1, img3d_input2)
    input_blob.create_datasets()
    # when training in parallel, the batch size should be equal to the number of GPUs
    num_devices = 1
    # this may create predictions of larger size than input
    input_dataset = input_blob.dataset_main.batch(num_devices, drop_remainder=True)
    steps = np.math.ceil(img3d_input1.shape[2]/num_devices)
    img_pred = model.predict(input_dataset, verbose=0, steps=steps)
    if img_pred.shape[3]==1:
        img_pred = img_pred[:,:,:,0]
    # make slice axis last
    img_pred = np.moveaxis(img_pred,0,2)
    return img_pred

class dataBlobGenerator:
    # class to load and iterate over image data
    # uploads entire dataset to RAM
    # the iterator method can be wrapped in a tensorflow dataset
    def __init__(self):
        self.data_input1 = []
        self.data_input2 = []
        self.data_output = []
        self.num_planes_total = 0
        
    def loadData(self, img3d_input1, img3d_input2):
        assert img3d_input1.shape[0] == img3d_input2.shape[0], "Input lists should have equal shape (dim 0)"
        assert img3d_input1.shape[1] == img3d_input2.shape[1], "Input lists should have equal shape (dim 1)"
        assert img3d_input1.shape[2] == img3d_input2.shape[2], "Input lists should have equal shape (dim 2)"

        self.num_planes_total = img3d_input1.shape[2] # defines when the iterator should reset
        # create placeholder 3d output
        img3d_output = np.zeros(img3d_input1.shape,dtype='float32')

        # read files into internal dict
        print('Loading image data to RAM...', end=' ')
        for imNumber in range(self.num_planes_total):
            # read+preprocess inputs
            img_data_input1 = img3d_input1[:,:,imNumber]
            img_data_input1 = img_data_input1.astype('float32')
            img_data_input2 = img3d_input2[:,:,imNumber]
            img_data_input2 = img_data_input2.astype('float32')
            img_data_output = img3d_output[:,:,imNumber]
            img_data_output = img_data_output.astype('float32')

            # expand to create channels
            img_data_input1 = img_data_input1[..., np.newaxis]
            img_data_input2 = img_data_input2[..., np.newaxis]
            img_data_output = img_data_output[..., np.newaxis]

            # write into dict
            self.data_input1.append(img_data_input1)
            self.data_input2.append(img_data_input2)
            self.data_output.append(img_data_output)

        self.data_input1 = np.array(self.data_input1)
        self.data_input2 = np.array(self.data_input2)
        self.data_output = np.array(self.data_output)
        print('Done.')

    def imageGeneratorMultislice(self):
        # put together neighboring slices as different channels
        imNumber = 0
        while imNumber < self.num_planes_total:

            if imNumber > 0:
                img_data_input1_pre = self.data_input1[imNumber-1]
                img_data_input2_pre = self.data_input2[imNumber-1]
            else:
                img_data_input1_pre = self.data_input1[imNumber]
                img_data_input2_pre = self.data_input2[imNumber]
            
            img_data_input1_mid = self.data_input1[imNumber]
            img_data_input2_mid = self.data_input2[imNumber]
            
            if imNumber < self.num_planes_total-1:
                img_data_input1_post = self.data_input1[imNumber+1]
                img_data_input2_post = self.data_input2[imNumber+1]
            else:
                img_data_input1_post = self.data_input1[imNumber]
                img_data_input2_post = self.data_input2[imNumber]

            img_data_input1 = np.concatenate([img_data_input1_pre,
                                            img_data_input1_mid,
                                            img_data_input1_post],-1)

            img_data_input2 = np.concatenate([img_data_input2_pre,
                                            img_data_input2_mid,
                                            img_data_input2_post],-1)
            # output
            img_data_output = self.data_output[imNumber]
            yield {'input1':img_data_input1, 'input2':img_data_input2}, img_data_output
            imNumber += 1
    
    def create_datasets(self):
        # Create datasets from stored arrays
        self.dataset_main = tf.data.Dataset.from_generator(self.imageGeneratorMultislice,
                                              args=[],
                                              output_types=({'input1':tf.float32,'input2':tf.float32}, tf.float32),
                                              output_shapes = ({'input1':INPUT1_SHAPE_2p5D,'input2':INPUT2_SHAPE_2p5D},OUTPUT_SHAPE_2p5D))
        self.dataset_main = self.dataset_main.repeat()


def initialize_model(model_name, modelWeightsFile):
    num_classes = 11
    input1 = tf.keras.Input(shape=INPUT1_SHAPE_2p5D, name='input1')
    input2 = tf.keras.Input(shape=INPUT2_SHAPE_2p5D, name='input2')
    args = [input1,input2,num_classes]
    model = get_model_by_name(model_name, args)
    predictions = Conv2D(num_classes, 1, activation = 'sigmoid')(model.layers[-2].output) # layer previous to last
    model = Model(inputs=[input1,input2], outputs=predictions)
    model.load_weights(modelWeightsFile)
    model.compile(optimizer=OPTIMIZER, loss=loss_DICE_multilabel)
    return model

def get_model_by_name(model_name, args):
    # define model by name
    if model_name == 'unet2D_inputfusion_vlarge_singleoutput':
        model = unet2D_inputfusion_vlarge_singleoutput(args[0], args[1])
    else:
        raise ValueError('Model name unknown.')
    return model
    # Do the other thing

def unet2D_inputfusion_vlarge_singleoutput(input1, input2):
    conv1_1 = inception_block(input1,32)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = inception_block(input2,32)
    conv1_2 = BatchNormalization()(conv1_2)
    conv1 = concatenate([conv1_1,conv1_2],axis = -1)
    # down-convolutions
    conv1 = Conv2D(128, 3, activation = activ, padding = 'same',data_format = "channels_last")(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = activ, padding = 'same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation = activ, padding = 'same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = activ, padding = 'same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation = activ, padding = 'same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = activ, padding = 'same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation = activ, padding = 'same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(512, 3, activation = activ, padding = 'same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation = activ, padding = 'same')(conv5)
    conv5 = BatchNormalization()(conv5)
    # up-convolutions plus skip connections
    upsamp6 = Conv2D(256, 2, activation = activ, padding = 'same')(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,upsamp6],axis=-1)
    conv6 = Conv2D(256, 3, activation = activ, padding = 'same')(merge6)
    conv6 = Conv2D(256, 3, activation = activ, padding = 'same')(conv6)
    upsamp7 = Conv2D(128, 2, activation = activ, padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,upsamp7],axis=-1)
    conv7 = Conv2D(128, 3, activation = activ, padding = 'same')(merge7)
    conv7 = Conv2D(128, 3, activation = activ, padding = 'same')(conv7)

    upsamp8 = Conv2D(128, 2, activation = activ, padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,upsamp8],axis=-1)
    conv8 = Conv2D(128, 3, activation = activ, padding = 'same')(merge8)
    conv8 = Conv2D(128, 3, activation = activ, padding = 'same')(conv8)

    upsamp9 = Conv2D(128, 2, activation = activ, padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,upsamp9],axis=-1)
    conv9 = Conv2D(128, 3, activation = activ, padding = 'same')(merge9)
    conv9 = Conv2D(128, 3, activation = activ, padding = 'same')(conv9)
    # 1x1 convolution instead of the dance layer
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    # conv10_softmax = Conv2D(numOutputChannels, 1, activation = 'softmax')(conv10)
    
    model = Model(inputs=[input1, input2], outputs = conv10)
    return model

def inception_block(input, num_filters):
    tower_1 = Conv2D(num_filters, (1,1), padding='same', activation = activ, data_format = "channels_last")(input)
    tower_1 = Conv2D(num_filters, (3,3), padding='same', activation = activ)(tower_1)

    tower_2 = Conv2D(num_filters, (1,1), padding='same', activation = activ, data_format = "channels_last")(input)
    tower_2 = Conv2D(num_filters, (5,5), padding='same', activation = activ)(tower_2)

    tower_3 = Conv2D(num_filters, (1,1), padding='same', activation = activ, data_format = "channels_last")(input)
    tower_3 = Conv2D(num_filters, (7,7), padding='same', activation = activ)(tower_3)

    tower_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input)
    tower_4 = Conv2D(num_filters, (1,1), padding='same', activation = activ,data_format = "channels_last")(tower_4)

    output = concatenate([tower_1, tower_2, tower_3, tower_4], axis = -1)
    return output

def loss_DICE_multilabel(y_true, y_pred, num_channels=11):
    dice=0
    for index in range(0,num_channels):
        dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice

def dice_coef(y_true, y_pred):
    smooth = 0.01
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)