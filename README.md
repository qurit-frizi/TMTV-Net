# TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmenta-tion in Lymphoma PET/CT Images – a Multi-Center Generalizability Analysis 

***Overview***
Total metabolic tumor volume (TMTV) segmentation has significant value enabling quantitative imaging biomarkers for lymphoma management. In this work, we tackle the challenging task of automated tumor delineation in lymphoma from PET/CT scans using a cascaded approach.TMTV-Net focuses on automating Total Metabolic Tumor Volume (TMTV) segmentation from PET/CT scans, offering valuable quantitative imaging biomarkers for effective lymphoma management. This repository details our approach and results.
TMTV-Net showcases robust performance and versatility in TMTV segmentation across diverse multi-site external datasets, covering a spectrum of lymphoma subtypes. Our model is containerized and available in this repository for further multi-site evaluation and generalizability analysis. *We welcome and encourage you to share your results with us.*

![image](https://github.com/qurit-frizi/TMTV-Net/assets/84542058/ba5d838e-f2ac-4d6a-b029-511e1de2853e)
*Figure 1: TMTV-Net.*
***Results:***
Our cascaded soft-voting guided approach resulted in performance with an average DSC of 0.68±0.12 for the internal test data from developmental dataset, and an average DSC of 0.66±0.18 on the multi-site external data (n=518), significantly outperforming (p<0.001) state-of-the-art (SOTA) approaches including nnU-Net and SWIN UNETR. While TTA yielded en-hanced performance gains for some of the comparator methods, its impact on our cascaded ap-proach was found to be negligible (DSC: 0.66±0.16). Our approach reliably quantified TMTV, with a correlation of 0.89 with the ground truth (p<0.001). Furthermore, in terms of visual as-sessment, concordance between quantitative evaluations and clinician feedback was observed in the majority of cases. The average relative error (ARE) and the absolute error (AE) in TMTV prediction on external multi-centric dataset are ARE=0.43±0.54 and AE=157.32±378.12 (mL) for all the external test data (n=518) and ARE=0.30±0.22 and AE=82.05±99.78 (mL) when the 10 % outliers (n=53) were excluded. 

## 📁 Required folder structure

## ⚙️  Installation <a name="installation"> </a>

### [Easy use: testing mode](#virtual) <a name="easy-use-testing-mode"> </a> 

## 📖 Citations (to be updated based on the revision result)
Please cite the following paper if you use TMTV-Net for your research:
Yousefirizi, F. TMTV-Net: Fully Automated Total Metabolic Tumor Volume Segmenta-tion in Lymphoma PET/CT Images – a Multi-Center Generalizability Analysis 

## 🙏 Acknowledgments
*We welcome and encourage you to share your results with us.*
