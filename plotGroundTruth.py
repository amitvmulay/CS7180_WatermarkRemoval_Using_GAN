
import numpy as np
import matplotlib.pyplot as plt


X_test_w  = np.load("/scratch/mulay.am/datasets/CLWD_1/X_test_w_5.npy")
X_test_g  = np.load("/scratch/mulay.am/datasets/CLWD_1/X_test_g_5.npy")
dir_result = "/scratch/mulay.am/1watermark/Results/"

def plot_images(path_save=None, titleadd=""):
    fig = plt.figure(figsize=(40, 10))
    nsample = X_test_w.shape[0]
    for i, img in enumerate(X_test_w):
        print(i)
        ax = fig.add_subplot(1, nsample, i + 1)
        ax.imshow(img)

    fig.suptitle("Original images " + titleadd, fontsize=30)

    if path_save is not None:
        plt.savefig(path_save,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()
    else:
        plt.close()


plot_images(path_save=dir_result + "OriginalImg",titleadd="")
