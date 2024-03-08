# TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmentation in Lymphoma PET/CT Images

***Overview***
Segmentation of Total Metabolic Tumor Volume (TMTV) holds significant value in facilitating quantitative imaging biomarkers for lymphoma management. We addressed the challenging task of automated tumor delineation in lymphoma from PET/CT scans. Our model, TMTV-Net, is specifically designed to automate TMTV segmentation from PET/CT scans. TMTV-Net demonstrates robust performance and adaptability in TMTV segmentation across diverse multi-site external datasets, encompassing various lymphoma subtypes. Furthermore, we have containerized our model and made it available in this repository for additional multi-site evaluations and generalizability analyses.

*We welcome and encourage you to share your results with us.* (Email: yousefi.f@gmail.com)

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
``



#### 1. Run Inference

Run the Docker container for inference:


``bash
docker run -it -v [/absolute/local/data/folder]:/input -v [/absolute/local/output/folder]:/output tmtv-net-inference python main.py
``


## Models

This repository includes large models that are hosted on Google Drive due to their size. To download the models, follow these steps:

1. Click on the following link to access the model files:
   - [Models](https://drive.google.com/file/d/1zfGIV_1k6YgijsEJUO9jVccN9Z67eJgi/view?usp=drive_link)

2. Once the Google Drive page opens, click on the "Download" button to download the model file to your local machine.

3. After downloading, you can place the model files in the appropriate directories within your Docker container or project folder.



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
[Yousefirizi, F. et. al. TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmenta-tion in Lymphoma PET/CT Images – a Multi-Center Generalizability Analysis, European Journal of Nuclear Medicine and Molecular Imaging. 2024 Feb 8. doi: 10.1007/s00259-024-06616-x.](https://pubmed.ncbi.nlm.nih.gov/38326655/)

## 🙏 Acknowledgments
[Torch Research Workflows](https://trw.readthedocs.io/en/latest/)

*We welcome and encourage you to share your results with us.*
