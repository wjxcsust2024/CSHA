import numpy as np

name = r'C:\Users\15504\Desktop\new4\results\Net776_1' #npy文件目录名
tp, tn, fp, fn = 0.0001, 0.0001, 0.0001, 0.0001
tp1, fp1, fn1 = 0.0001, 0.0001, 0.0001
tn1, fp1, fn1 = 0.0001, 0.0001, 0.0001
iou1, iou2, miou = 0., 0., 0.
epoachs = 156 #得到的npy文件数量
total = 156 #总共图片数量
for i in range(epoachs):
    img_path = name + '\pred' + str(i+1) + '.npy'
    mask_path = name + '\label' + str(i+1) + '.npy'
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
                    if msk[num][0][j][k]==0:
                        if img[num][0][j][k]==0:
                            tn += 1
                            tn1 += 1
                        else:
                            fp += 1
                            fp1 += 1
                    else:
                        if img[num][0][j][k]!=0:
                            tp += 1
                            tp1 += 1
                        else:
                            fn += 1
                            fn1 += 1
            iou1 += tp1 / (tp1+fp1+fn1)
            iou2 += tn1 / (tn1+fp1+fn1)
            miou += (iou1+iou2)
            tp1 = 0.0001
            tn1 = 0.0001
            fp1 = 0.0001
            fn1 = 0.0001
            iou1 = 0.0001
            iou2 = 0.0001

print('--------'+name+'--------')
recall = tp / (tp+fn)
precision = tp / (tp+fp)
f1 = (2*recall*precision) / (recall+precision)
noise = fp / (tp+fp)
accurancy = (tp+tn) / (tp+tn+fp+fn)

print('recall:', recall)
print('f1:', f1)
print('precision:', precision)
print('noise:', noise)
print('accurancy:', accurancy)
print('mIOU:', miou/(total*2))