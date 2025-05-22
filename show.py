import cv2
import numpy as np
import matplotlib.pylab as plt
import torch

path = r'D:\Desktop\new6\deepcrack_net\compared\deepcrack'
ipath = r'D:\Desktop\new6\deepcrack_net\comnpy\deepcrack'
# for i in range(40):
i = 52
img_path = path + '\pred' + str(i + 1) + '.npy'
mask_path = ipath + '\img' + str(i + 1) + '.npy'
img = np.load(img_path)
msk = np.load(mask_path)
# print(img.shape)
B= img.shape[0]
for j in range(B):
    image = img[j]
    image[image>0] = 1
    image[image<0] = 0
    lab = msk[j]
    image = torch.from_numpy(image)
    lab = torch.from_numpy(lab)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(image.permute(1, 2, 0), cmap='gray')
    ax1.set_title('pred')
    ax2.imshow(lab.permute(1, 2, 0))#, cmap='gray')
    ax2.set_title('label')
    plt.pause(0.005)
    plt.show()