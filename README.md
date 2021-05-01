# CS7180 WatermarkRemoval Using cGAN

## Datasets used
CLWD <a href="https://drive.google.com/file/d/17y1gkUhIV6rZJg1gMG-gzVMnH27fm4Ij/view?usp=sharing" target="_top">Download Link</a>

### Additional instructions
For speeding up the training, use ImgToNpArray.py to generate NumPy array data structure of images from the dataset and save it as .npy file.
We already have these files created for subset of CLWD dataset (10k watermarked and 10k watermark free images) and stored at following directory "/scratch/mulay.am/datasets/CLWD_1/" and these are accessible from all other repositories. So train.py with provided path for training and testing data shoudl be able to access the data from mentioned folders without any access issue when executed from Discovery Cluster Node by any Discovery account.

In case of issues accessing these files from mentioned path n train.py, kindly use above download link of the CLWD dataset and then use ImgToNpArray.py to generate your own files.

ImgToNpArray.py Script can be used to Generate both training and testing data array files which are used for inference after every 10 epochs. Make appropriate changes to get the dataset of size you want.

GroundTruth.py can be used to plot test ground truth images. Make appropriate changes to plot either watermark or watermark free images of the selected set.
