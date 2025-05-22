import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt

from Net2 import Net
from dataloader1 import Datases_loader as dataloader
from dataloader2 import Datases_loader as test_dataloader
from Dice_BCEwithLogits import SoftDiceLoss as bcedice

def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.9, max_iter=80):
    return initial_lr * ((1 - step / float(max_iter)) ** power)


def adjust_lr(optimizer, step, initial_lr):
    lr = inverseDecayScheduler(step, initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
batchsz = 4
lr = 0.0005
items = 100
model = Net().to(device)
criterion = bcedice()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=2e-5)


imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\train_img'
labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\train_lab'
savedir = r'C:\Users\15504\Desktop\new3\weights\Net315_1.pth'

test_imgdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_img'
test_labdir = r'C:\Users\15504\Desktop\new6\Deepcrack\Crack315\Crack315\test_lab'

resultsdir = r'C:\Users\15504\Desktop\new3\results\Net315_1'
csv_path = r'C:\Users\15504\Desktop\new3\results\Dtrc_315_100.csv'

imgsz = 512
dataset = dataloader(imgdir, labdir, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)

ls_loss = []

metrics = []


def train():
    for epoch in range(items):
        lossx = 0
        model.train()
        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred = model(img)

            loss = criterion(pred, lab)
            loss.backward()
            optimizer.step()

            lossx += loss
            # .item()
            # print(f'epoch: {epoch + 1} --- id: {idx + 1}')

        adjust_lr(optimizer, epoch, optimizer.param_groups[0]['lr'])
        lossx = lossx / dataset.num_of_samples()
        ls_loss.append(lossx)
        print(f'epoch {epoch + 1} --- loss: {lossx}')
        # 保存模型权重
        torch.save(model.state_dict(), savedir)

        # 测试并保存结果
        with torch.no_grad():
            test(epoch + 1)
        # 评估当前模型并保存指标
        metrics_dict = evaluate(epoch + 1)
        metrics_dict['loss'] = lossx.item()  # 在指标中添加损失值
        metrics.append(metrics_dict)

        # 保存指标到 CSV
        save_metrics_to_csv(metrics)


def test(epoch):
    test_dataset = test_dataloader(test_imgdir, test_labdir, imgsz, imgsz)
    testsets = DataLoader(test_dataset, batch_size=4, shuffle=False)
    model.load_state_dict(torch.load(savedir))
    model.eval()
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)
        pred = model(img)
        print(f'epoch: {epoch} --- id: {idx + 1}')
        # for na in name:
        #     print(na)
        #
        # for i, na in zip(pred, name):
        #     binary_img = np.where(i[0].cpu().detach().numpy() < 0.5, 255, 0).astype(np.uint8)
        #     plt.close()

        np.save(os.path.join(resultsdir, f'pred{idx + 1}.npy'), pred.detach().cpu().numpy())
        np.save(os.path.join(resultsdir, f'label{idx + 1}.npy'), lab.detach().cpu().numpy())


def evaluate(epoch):
    name = resultsdir
    tp, tn, fp, fn = 0.0001, 0.0001, 0.0001, 0.0001
    tp1, fp1, fn1 = 0.0001, 0.0001, 0.0001
    tn1, fp1, fn1 = 0.0001, 0.0001, 0.0001
    iou1, iou2, miou = 0., 0., 0.
    epoachs = 16  # npy 文件数量
    total = 63  # 总共图片数量

    for i in range(epoachs):
        img_path = name + f'/pred{i + 1}.npy'
        mask_path = name + f'/label{i + 1}.npy'
        img = np.load(img_path)
        msk = np.load(mask_path)
        img[img > 0] = 1.
        img[img <= 0] = 0.
        msk[msk > 0] = 1.
        msk[msk <= 0] = 0.
        B = img.shape[0]
        ax1, ax2 = img.shape[2], img.shape[3]

        for num in range(B):
            for j in range(ax1):
                for k in range(ax2):
                    if msk[num][0][j][k] == 0:
                        if img[num][0][j][k] == 0:
                            tn += 1
                            tn1 += 1
                        else:
                            fp += 1
                            fp1 += 1
                    else:
                        if img[num][0][j][k] != 0:
                            tp += 1
                            tp1 += 1
                        else:
                            fn += 1
                            fn1 += 1
            iou1 += tp1 / (tp1 + fp1 + fn1)
            iou2 += tn1 / (tn1 + fp1 + fn1)
            miou += (iou1 + iou2)
            tp1, tn1, fp1, fn1 = 0.0001, 0.0001, 0.0001, 0.0001
            iou1, iou2 = 0.0001, 0.0001

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * recall * precision) / (recall + precision)
    noise = fp / (tp + fp)
    accurancy = (tp + tn) / (tp + tn + fp + fn)
    miou_score = miou / (total * 2)

    print(f'--------epoch {epoch}--------')
    print('recall:', recall)
    print('f1:', f1)
    print('precision:', precision)
    print('noise:', noise)
    print('accurancy:', accurancy)
    print('mIOU:', miou_score)

    return {
        'epoch': epoch,
        'recall': recall,
        'f1': f1,
        'precision': precision,
        'noise': noise,
        'accurancy': accurancy,
        'mIOU': miou_score
    }


def save_metrics_to_csv(metrics):
    df = pd.DataFrame(metrics)
    df.to_csv(csv_path, index=False, mode='w')  # 指定写入模式为覆盖


if __name__ == "__main__":
    train()
