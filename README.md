# TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmentation in Lymphoma PET/CT Images

***Overview***
Segmentation of Total Metabolic Tumor Volume (TMTV) holds significant value in facilitating quantitative imaging biomarkers for lymphoma management. We addressed the challenging task of automated tumor delineation in lymphoma from PET/CT scans. Our model, TMTV-Net, is specifically designed to automate TMTV segmentation from PET/CT scans. TMTV-Net demonstrates robust performance and adaptability in TMTV segmentation across diverse multi-site external datasets, encompassing various lymphoma subtypes. Furthermore, we have containerized our model and made it available in this repository for additional multi-site evaluations and generalizability analyses.

*We welcome and encourage you to share your results with us.* (Email: yousefi.f@gmail.com)

![image](https://github.com/qurit-frizi/TMTV-Net/assets/84542058/e624af9e-0389-4237-885b-4b2ebbd8d3fe)


*Figure 1: TMTV-Net, was utilized at the University of Wisconsin on their dataset for the segmentation of TMTV. (a) Hodgkin Case, DSC=0.83, TMTV relative error=0.18, (b) DLBCL case, DSC=0.66, TMTV relative error=0.10, (c) Hodgkins case, DSC=0.76, (d) DLBCL, DSC=0.67.*

## ⚙️  Usage <a name="installation"> </a>


# Data Preprocessing

For training you need to convert the nifti files to HDF5:

    cd <root>/auto_pet/src/auto_pet/projects/segmentation/preprocessing/
    python create_dataset.py

The preprocessed data will be stored here: `<root>/dataset/preprocessed/`




### [Easy use: testing mode](#virtual) <a name="easy-use-testing-mode"> </a> 
## 🐳 Docker for Inference

This repository includes a Docker setup for running inference using TMTV-Net. The Docker image encapsulates the necessary environment, dependencies, and the trained model for seamless and reproducible inference on new data.


### 📂 Directory Structure

- **src:** This directory contains Python files serving as libraries and helpers.
- ***models:*** This folder is designated for storing trained model weights (`.model` files). Please ensure to create a folder named "models" within the "src" directory. To access the model files that need to be included in the "models" folder, click the following link:
   - [Models](https://drive.google.com/file/d/1zfGIV_1k6YgijsEJUO9jVccN9Z67eJgi/view?usp=drive_link)
- **dockerfile:** Configuration file used for building the Docker image.
- **main.py:** Entry point for executing inference.
- **requirements.txt:** A list of Python dependencies required for the Docker image.



### 🚀 Usage

### 0. Donwload (Clone) this repo

Clone this repository to your local machine using the following command:

```bash
git clone https://github.com/qurit-frizi/TMTV-Net.git
```

### 1. Ensure to include model files

This repository includes large models that are hosted on Google Drive due to their size. To download the models, follow these steps:

1. Click on the following link to access the model files:
   - [Models](https://drive.google.com/file/d/1zfGIV_1k6YgijsEJUO9jVccN9Z67eJgi/view?usp=drive_link)

2. Once the Google Drive page opens, click on the "Download" button to download the model file to your local machine.

3. After downloading, please place the model files in a folder named "models" within the "main/src" directory.



### 2. Build Docker Image

Before running inference, build the Docker image. Navigate to the `main/` folder where the Dockerfile is located, then execute the following command:

```bash
docker build -t tmtv-net-inference .
```


### 3. Run Inference

Execute the Docker container to perform inference:

```bash
docker run -it -v [/absolute/local/data/folder]:/input -v [/absolute/local/output/folder]:/output tmtv-net-inference
```

Ensure to replace [/absolute/local/data/folder] and [/absolute/local/output/folder] with the absolute paths to your local data and output folders respectively. This command will initiate the inference process within the Docker container.

![image](https://github.com/qurit-frizi/TMTV-Net/assets/84542058/830bc633-7f56-4cdd-b4e5-7d1cf34d39f2)
Please ensure that the input/output mapping is correctly configured for Windows systems.

📁 Required folder structure for testing: The input folder path in your machine should contain a folder of CT and a folder of PET DICOM scans. 


Feel free to explore and adapt the provided commands based on your specific folder structure or naming conventions. This Docker setup ensures a consistent and reproducible environment for running TMTV-Net inference.


## License
TMTV-Net is shared for research-use only. COMMERCIAL USE IS PROHIBITED for the time being. For further information please email frizi@bccrc.ca 

## 📖 Citations (to be updated based on the revision result)
Please cite the following paper if you use TMTV-Net for your research:
[Yousefirizi, F. et. al. TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmenta-tion in Lymphoma PET/CT Images – a Multi-Center Generalizability Analysis, European Journal of Nuclear Medicine and Molecular Imaging. 2024 Feb 8. doi: 10.1007/s00259-024-06616-x.](https://pubmed.ncbi.nlm.nih.gov/38326655/)

## 🙏 Acknowledgments
[Torch Research Workflows](https://trw.readthedocs.io/en/latest/)

*We welcome and encourage you to share your results with us.*
