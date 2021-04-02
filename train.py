import os
import glob
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from model.BaseNet import CPFNet
from utils.Breast import Breast
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


def pixel_accuracy(pred,label):
    #计算分类正确的像素点数和所有的像素点数的比例
    pred=torch.argmax(torch.exp(pred),dim=1)
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    pred[pred>0] = 1.
    pred[pred<=0] = 0.

    label_class = np.sum(label>0)
    correct_pixel = np.sum((pred == label)*(label>0))
    pixel_accuracy = 1.0*correct_pixel/label_class

    return pixel_accuracy


def adjust_learning_rate(lr, optimizer, epoch, epochs):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if epoch%3==0:
        lr = lr * (1 - epoch / epochs) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr


def train(EPOCHS, model, optimizer, trainloader, validloader):

    best_loss = float('inf')
    criterion = Multi_DiceLoss()
    for epoch in range(EPOCHS):

        trian_loss = 0.0
        model.train()
        print('\n训练：...')
        #lr = adjust_learning_rate(0.001, optimizer, epoch, EPOCHS)

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

            if loss<best_loss:
                best_loss = loss
                torch.save(model.state_dict(), 'best_model.pth')

            if (idx)%10==0:
                print('idx:{}\tloss:{}'.format(idx,loss.item()))

        print('Epoch:{}\ttrainLoss:{}\n'.format(epoch,trian_loss/idx))

        valid(model,validloader)
    #     break
    # plt.figure(figsize=(15,12))
    # predict = main_out[0][0].cpu().detach().numpy()
    #
    # plt.imshow(predict,'gray')
    # plt.show()


def valid(model, dataloader):

    print('验证：...')

    with torch.no_grad():
        model.eval()
        corrct = 0

        for idx,(img,label) in enumerate(dataloader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()

            aux_predict, predict = model(img)
            acc = pixel_accuracy(predict, label)

            corrct += acc
            if (idx+1)%4==0:
                print('pixel_accuracy:\t',acc)

        print('Average Accuracy:{}%'.format(corrct/len(dataloader)*100))


def test(model, dataloader, testPath):

    imgList = glob.glob(os.path.join(testPath,'imgs/*.jpg'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        model.to(device)
        model.load_state_dict(torch.load('best_model.pth',map_location=device))
        model.eval()

        for idx,img in enumerate(dataloader):

            savePath = imgList[idx].replace('imgs','labels')

            if torch.cuda.is_available():
                img = img.to(device,dtype=torch.float32)

            pred_,pred =model(img)
            predict = torch.argmax(torch.exp(pred),dim=1)
            predict = predict.data.cpu().numpy()[0]

            predict[predict>0]=255
            predict[predict<=0]=0

            img = Image.fromarray(predict.astype(np.uint8))
            img.save(savePath)


def main(mode='tr'):

    # 训练次数
    EPOCHS = 10

    # 数据路径
    trianPath = r'D:\PyProfile\06-MyTrain\data\train'
    validPath = r'D:\PyProfile\06-MyTrain\data\valid'
    testPath = r'D:\PyProfile\06-MyTrain\data\test'

    # 准备数据
    dataTrian = Breast(trianPath,(256,256),mode='tr')
    dataValid = Breast(validPath,(256,256),mode='val')
    dataTest = Breast(testPath,(256,256),mode='ts')

    # 训练数据
    trianLoader = DataLoader(
        dataTrian,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True )

    # 验证数据
    validLoader = DataLoader(
        dataValid,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True )

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-2)

    # 选择执行模式
    if mode == 'tr':
        train(EPOCHS, model, optimizer, trianLoader, validLoader)
    if mode == 'ts':
        test(model, testLoader, testPath)


if __name__ == '__main__':
    main(mode='ts')


    

























