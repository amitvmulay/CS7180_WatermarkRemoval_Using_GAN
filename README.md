# CS7180 WatermarkRemoval Using cGAN

## 1. Please follow the below steps for watermark removal from colored natural images
### Software Dependencies for train.py
Python 3.7  
CUDA 10.2  
Tensoflow-gpu  

### Dataset
CLWD <a href="https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing" target="_top">Download Link</a>

#### Additional instructions for training the model for Colored Natural Images from CLWD 
For speeding up the training, use ImgToNpArray.py to generate NumPy array data structure of images from the dataset and save it as .npy file.
We already have these files created for subset of CLWD dataset (10k watermarked and 10k watermark free images) and stored at following directory "/scratch/mulay.am/datasets/CLWD_1/" and these are accessible from all other repositories. So train.py with provided path for training and testing data shoudl be able to access the data from mentioned folders without any access issue when executed from Discovery Cluster Node by any Discovery account.

In case of issues accessing these files from mentioned path n train.py, kindly use above download link of the CLWD dataset and then use ImgToNpArray.py to generate your own files.

ImgToNpArray.py Script can be used to Generate both training and testing data array files which are used for inference after every 10 epochs. Make appropriate changes to get the dataset of size you want.

GroundTruth.py can be used to plot test ground truth images. Make appropriate changes to plot either watermark or watermark free images of the selected set.

## 2. Please follow the below steps for watermark removal from grey-scale text documents
### Software Dependencies for DE_GAN_watermark_removal.ipynb
Python 3.7  
Google Colab  
Pytorch-gpu 

### Dataset
Custom dataset <a href="https://docs.google.com/u/0/uc?export=download&confirm=2lHb&id=0B9eZ-svj9om8Yjc1YWYwZTUtYTNhOS00ZWE2LTliOGItM2UzMGM2M2VkNWRj" target="_top">Download Link</a> (no need to download data on local for code to run).

#### Additional instructions for training the model for grey-scaled images of text documents
The DE_GAN_watermark_removal.ipynb file can be run directly. Data need not be downloaded to the local machine as the code accesses is directly from the google drive. Please provide google drive authentication for code to access the dataset while running cell number 5. Please ensure to follow the comments/instructions written above the cells for smooth execution of the codes and saving the trained model to google drive. 
