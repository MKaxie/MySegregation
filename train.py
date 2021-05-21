import os
import glob
import torch
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.BaseNet import CPFNet
from utils.Breast import Breast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt



class Multi_DiceLoss(nn.Module):
    # 混合Dice损失计算
    def __init__(self, class_num=2, smooth=0.00001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):

        input = torch.exp(input)

        Dice = torch.Tensor([0.]).cuda()

        for i in range(0, self.class_num):

            input_i = input[:, i, :, :]
            target_i = (target == i).float()

            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)

            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice = Dice + dice

        dice_loss = 1 - Dice / (self.class_num)
        return dice_loss


def validation(pred,label):
    #计算分类正确的像素点数和所有的像素点数的比例
    pred=torch.argmax(torch.exp(pred),dim=1)
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    f = label.any()
    if not f:
        return 100., 100.

    pred[pred>0] = 1.
    pred[pred<=0] = 0. 

    p = np.sum(pred==1)
    l = np.sum(label==1)

    intersection = np.sum((pred==label)*(label==1))
    iou = 1.0*intersection/(p+l-intersection)

    label_class = np.sum(label==1)
    correct_pixel = np.sum((pred == label)*(label==1))
    pixel_accuracy = 1.0*correct_pixel/label_class

    return iou ,pixel_accuracy


def train(EPOCHS, model, optimizer, trainloader, validloader,scheduler,Batch_Size):

    print('\n')
    best_loss = float('inf')
    criterion = Multi_DiceLoss()

    for epoch in range(EPOCHS):

        tq = tqdm(total=len(trainloader) * Batch_Size)
        tq.set_description('epoch %d' % (epoch))
        
        loss_record = []
        trian_loss = 0.0
        model.train()
        
        for idx,(img,label) in enumerate(trainloader):

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda().long()

            aux_out, main_out = model(img)

            optimizer.zero_grad()
            loss_aux = F.nll_loss(aux_out, label)
            loss_main = criterion(main_out, label)

            loss = loss_aux + loss_main

            loss.backward()
            optimizer.step()
            trian_loss += loss.item()

            tq.update(Batch_Size)
            tq.set_postfix(loss='%.6f' % (trian_loss/(idx+1)))

            loss_record.append(loss.item())

            if loss.item()<best_loss:
                best_loss = loss.item()
                if best_loss<0.05:
                    name = time.strftime("%Y-%m-%d %X", time.localtime()).split(" ")[0]
                    path = f'trainedModel/{str(best_loss)[:4]}_{name}_{img.shape[-1]}_model.pth'
                    torch.save(model.state_dict(), path)

        tq.close()
        AverageLoss = np.mean(loss_record)
        print('Average Loss:',AverageLoss,'\n')

        valid(model,validloader,Batch_Size)

        scheduler.step(trian_loss)


def valid(model, dataloader,Batch_Size):

    with torch.no_grad():
        model.eval()

        tq = tqdm(total=len(dataloader)*Batch_Size)

        AveIOU = []
        AveAcc = []

        for idx,(img,label) in enumerate(dataloader):

            tq.set_description('Validation')

            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            aux_predict, predict = model(img)

            acc,iou = validation(predict, label)
            AveIOU.append(iou)
            AveAcc.append(acc)

            tq.update(Batch_Size)
            tq.set_postfix({'Accuracy':acc,'IOU':iou,})
        tq.close()
        print('Average IOU:{:.2f}%'.format(np.mean(AveIOU)*100))
        print('Avearge Accuracy:{:.2f}%\n'.format(np.mean(AveAcc)*100))


def test(model, dataloader, testPath):

    imgList = glob.glob(os.path.join(testPath,'imgs/*.jpg'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        model.to(device)
        model.load_state_dict(torch.load('best_model.pth',map_location=device))
        model.eval()

        iou = 0.
        t = 0
        for idx,(img,label)in enumerate(dataloader):

            savePath = imgList[idx].replace('imgs','labels')

            if torch.cuda.is_available():
                img = img.to(device,dtype=torch.float32)

            pred_,pred =model(img)
            predict = torch.argmax(torch.exp(pred),dim=1)
            predict = predict.data.cpu().numpy()[0]

            ou = validation(pred_,label)
            if ou!=0:
                iou += ou
                t += 1

            if idx%5==0:
                print('intersection over union:',ou,'%')

            predict[predict>0] = 255
            predict[predict<=0] = 0

            img = Image.fromarray(predict.astype(np.uint8))
            img.save(savePath)

        print('Average intersection over union:',iou/t,'%')


def main(mode='tr',scale=(512,512)):

    # 训练次数
    EPOCHS = 40
    Batch_Size = 4

    # 数据路径
    trianPath = 'data/train'
    validPath = 'data/valid'
    testPath = 'data/test'

    # 准备数据
    dataTrian = Breast(trianPath,scale,mode='tr')
    dataValid = Breast(validPath,scale,mode='val')
    dataTest = Breast(testPath,scale,mode='ts')

    # 训练数据
    trianLoader = DataLoader(
        dataTrian,
        batch_size=Batch_Size,
        shuffle=True,
        pin_memory=True,
        drop_last=True ,
        num_workers=8)

    # 验证数据
    validLoader = DataLoader(
        dataValid,
        batch_size=Batch_Size,
        shuffle=True,
        pin_memory=True,
        drop_last=True ,
        num_workers=8)

    # 测试数据
    testLoader = DataLoader(
        dataTest,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=True )

    # 创建模型
    model = CPFNet(out_planes=2)

    if torch.cuda.is_available():
        model = model.cuda()

    # 构造优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    # 选择执行模式
    if mode == 'tr':
        train(EPOCHS, model, optimizer, trianLoader, validLoader,scheduler,Batch_Size)
    if mode == 'ts':
        test(model, testLoader, testPath)


if __name__ == '__main__':
    main(mode='tr')





#Average Loss:0.0025
#Average IOU:94%
#Avearge Accuracy:89%

























