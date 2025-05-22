import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np

# 输入图片路径、掩码路径，高、宽，加载图片
class Datases_loader(Dataset):
    def __init__(self, root_images, root_masks, h, w):
        super().__init__()
        self.root_images = root_images
        self.root_masks = root_masks
        self.h = h
        self.w = w
        self.images = []
        self.labels = []

        #获得img和msk文件
        files = sorted(os.listdir(self.root_images))
        sfiles = sorted(os.listdir(self.root_masks))
        for i in range(len(sfiles)):
            img_file = os.path.join(self.root_images, files[i])
            mask_file = os.path.join(self.root_masks, sfiles[i])
            # print(img_file, mask_file)
            self.images.append(img_file)
            self.labels.append(mask_file)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]
        image = Image.open(image)
        mask = Image.open(mask)
        tf = transforms.Compose([
            transforms.Resize((int(self.h * 1.25), int(self.w * 1.25))),
            #transforms.RandomHorizontalFlip(0.5),
            #transforms.RandomVerticalFlip(0.5),
            #transforms.RandomRotation(16),
            transforms.CenterCrop((self.h, self.w)),
            transforms.ToTensor()
        ])

        # image = image.convert('L')
        image = image.convert('RGB')
        # image = image.filter(ImageFilter.SHARPEN)
        # mask = mask.convert('L')
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        seed = np.random.randint(1459343089)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        img = tf(image)
        img = norm(img)

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # print(np.max(mask))
        mask = tf(mask)
        mask[mask>0] = 1.
        # mask = (mask==13).float()  #kouyifeigai,duobiaoqianweidanbiaoqian
        sample = {'image': img, 'mask': mask, } #kouyifeigai,duobiaoqianweidanbiaoqian
        # sample = {'image': torch.Tensor(img), 'mask': torch.Tensor(mask)}

        return sample

# 可以设置batchsize为8，观察加载的图片
def imshow_image(mydata_loader):
    plt.figure()
    for (cnt, i) in enumerate(mydata_loader):
        image = i['image']
        label = i['mask']
        # print(image.shape, label.shape)

        for j in range(8):  # 一个批次设为：8
            # ax = plt.subplot(2, 4, j + 1)
            # ax.axis('off')
            ax1 = plt.subplot(121)
            ax2 = plt.subplot(122)

            # permute函数：可以同时多次交换tensor的维度
            # print(image[j].permute(1, 2, 0).shape)
            # print(image.shape)
            # print(label.shape)
            ax1.imshow(image[j].permute(1, 2, 0), cmap='gray')
            ax1.set_title('image')
            ax2.imshow(label[j].permute(1, 2, 0), cmap='gray')
            ax2.set_title('mask')
            # plt.pause(0.005)
            plt.show()
        if cnt == 6:
            break
    plt.pause(0.005)

def save_imgae(mydata_loader):
    for (cnt, i) in enumerate(mydata_loader):
        image = i['image']
        # label = i['mask']
        # print(image.shape, label.shape)

        for j in range(image.shape[0]):  # 一个批次设为：8
            # img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imsave(r'E:\my code\two_path_network\resimg\260\img\\' + str(cnt) + '_' + str(j) + ".png",
                       image[j].permute(1, 2, 0).numpy())

        # if cnt == 6:
        #     break

if __name__ == '__main__':
    # imgdir = r'/root/autodl-tmp/crack_transformer1/data/Deepcrack/test_img'
    # labdir = r'/root/autodl-tmp/crack_transformer1/data/Deepcrack/test_lab'
    d = Datases_loader(r'D:\Desktop\new\Deepcrack\Deepcrack\train_img',
                       r'D:\Desktop\new\Deepcrack\Deepcrack\train_lab',
                       512, 512)
    # d = Datases_loader(imgdir, labdir, 512, 512)
    mydata_loader = DataLoader(d, batch_size=8, shuffle=False)
    imshow_image(mydata_loader)
    # save_imgae(mydata_loader)

    # for sample in mydata_loader:
    #     x = sample['image']
    #     y = sample['mask']
        # print(x, y)