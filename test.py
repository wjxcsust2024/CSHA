import torch
import numpy as np
from torch.utils.data import DataLoader
from Net2 import Net
import os
from dataloader2 import Datases_loader as dataloader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1
model = Net().to(device)
#savedir = r'D:\Desktop\new6\weights\midweights\crack100_1_40.pth'
savedir = r'C:\Users\15504\Desktop\new4\weights\Net776_1.pth'

#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\test_lab'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\CrackTree260\test_img'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_lab'
imgdir = r'C:\Users\15504\Desktop\new6\crack776\crack776\test\image'
labdir = r'C:\Users\15504\Desktop\new6\crack776\crack776\test\label'
#imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_img'
#labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Deepcrack\test_lab'
imgsz = 512
resultsdir = r'C:\Users\15504\Desktop\new4\results\Net776_1'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)
    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img)
        # print(pred.shape)
        #for i in pred:
            #plt.imshow(i[0].cpu().detach().numpy())
            #plt.show()
        #print(pred)
        #CE
        # B = pred.shape[0]
        # pred = pred.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
        # pred = pred.argmax(1)
        # pred = pred.reshape(B, 1, imgsz, imgsz)
        #
        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy())

if __name__ == '__main__':
    test()