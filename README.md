# TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmentation in Lymphoma PET/CT Images

***Overview***
Total metabolic tumor volume (TMTV) segmentation has significant value enabling quantitative imaging biomarkers for lymphoma management. In this work, we tackle the challenging task of automated tumor delineation in lymphoma from PET/CT scans using a cascaded approach.TMTV-Net focuses on automating Total Metabolic Tumor Volume (TMTV) segmentation from PET/CT scans, offering valuable quantitative imaging biomarkers for effective lymphoma management. 
TMTV-Net showcases robust performance and versatility in TMTV segmentation across diverse multi-site external datasets, covering a spectrum of lymphoma subtypes. Our model is containerized and available in this repository for further multi-site evaluation and generalizability analysis. 
*We welcome and encourage you to share your results with us.*


***Results:***
Our cascaded soft-voting guided approach resulted in performance with an average DSC of 0.68±0.12 for the internal test data from developmental dataset, and an average DSC of 0.66±0.18 on the multi-site external data (n=518), significantly outperforming (p<0.001) state-of-the-art (SOTA) approaches including nnU-Net and SWIN UNETR. While TTA yielded en-hanced performance gains for some of the comparator methods, its impact on our cascaded ap-proach was found to be negligible (DSC: 0.66±0.16). Our approach reliably quantified TMTV, with a correlation of 0.89 with the ground truth (p<0.001). Furthermore, in terms of visual as-sessment, concordance between quantitative evaluations and clinician feedback was observed in the majority of cases. The average relative error (ARE) and the absolute error (AE) in TMTV prediction on external multi-centric dataset are ARE=0.43±0.54 and AE=157.32±378.12 (mL) for all the external test data (n=518) and ARE=0.30±0.22 and AE=82.05±99.78 (mL) when the 10 % outliers (n=53) were excluded. 

  ![image](https://github.com/qurit-frizi/TMTV-Net/assets/84542058/3b7a51f8-8b6c-4dc7-a3f4-711efc30995d)

*Figure 1: TMTV-Net, was utilized at the University of Wisconsin on their dataset for the segmentation of TMTV. (a) Hodgkin Case, DSC=0.83, TMTV relative error=0.18, (b) DLBCL case, DSC=0.66, TMTV relative error=0.10, (c) Hodgkins case, DSC=0.76, (d) DLBCL, DSC=0.67.*

## ⚙️  Usage <a name="installation"> </a>


#📁 Required folder structure for training
# Data Preprocessing

For training you need to convert the nifti files to HDF5:

    cd <root>/auto_pet/src/auto_pet/projects/segmentation/preprocessing/
    python create_dataset.py

The preprocessed data will be stored here: `<root>/dataset/preprocessed/`

# 📁 Required folder structure for testing



### [Easy use: testing mode](#virtual) <a name="easy-use-testing-mode"> </a> 
## 🐳 Docker for Inference

This repository includes a Docker setup for running inference using TMTV-Net. The Docker image encapsulates the necessary environment, dependencies, and the trained model for seamless and reproducible inference on new data.

### 📂 Directory Structure

- **src:** Python files serving as libraries and helpers.
- **models:** Folder containing the trained model weights (`.model` files).
- **dockerfile:** Configuration file for building the Docker image.
- **main.py:** Entry point for running inference.
- **requirements.txt:** List of Python dependencies required for the Docker image.

### 🚀 Usage

#### 1. Build Docker Image


Before running inference, build the Docker image:

``bash
docker build -t tmtv-net-inference .

#### 1. Run Inference

Run the Docker container for inference:

docker run -it tmtv-net-inference python main.py


📦 Git LFS (Large File Storage)
Due to the large size of the model files, we use Git LFS (Large File Storage) to efficiently handle and version these files. Make sure you have Git LFS installed to fetch the model weights properly.

Installing Git LFS:
#### On Linux
sudo apt-get install git-lfs
#### On macOS
brew install git-lfs

Clone Repository with LFS:
git lfs install
git clone https://github.com/qurit-frizi/TMTV-Net.git


Feel free to explore and adapt the provided commands based on your specific folder structure or naming conventions. This Docker setup ensures a consistent and reproducible environment for running TMTV-Net inference.

## Usage
TMTV-Net is shared for research-use only. COMMERCIAL USE IS PROHIBITED for the time being. For further information please email frizi@bccrc.ca 

## 📖 Citations (to be updated based on the revision result)
Please cite the following paper if you use TMTV-Net for your research:
Yousefirizi, F. et. al. TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmenta-tion in Lymphoma PET/CT Images – a Multi-Center Generalizability Analysis 

## 🙏 Acknowledgments
*We welcome and encourage you to share your results with us.*
