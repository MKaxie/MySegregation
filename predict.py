import os
import cv2
from glob import glob
import torch
import pydicom as pdc
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import scale
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.BaseNet import CPFNet
from utils.Breast import Breast
import matplotlib.pyplot as plt
from torchvision import transforms




def predictImgs(model,imgPath,scale): # 分割图片数据

    imgList = glob.glob(imgPath + '/*')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelPath = findmodel(scale)

    with torch.no_grad():

        model.to(device)
        model.load_state_dict(torch.load(modelPath,map_location=device)) # 预训练模型读取
        model.eval()
        
        tq = tqdm(total=len(imgList)) # 进度展示

        resize = transforms.Resize(scale)
        to_tensor = transforms.ToTensor()

        for idx,img in enumerate(imgList):

            tq.set_description('Predicting imgs')

            savePath = imgList[idx].replace('imgs','masks')

            img = Image.open(img).convert('RGB') # 将单通道图像转换成RGB图像
             # 更改图像大小

            img = resize(img)
            img = to_tensor(img) #

            img = img.view(1,3,scale[0],scale[1]) # batch化处理

            if torch.cuda.is_available():
                img = img.to(device,dtype=torch.float32)

            pred_,pred =model(img)
            prediction = torch.argmax(torch.exp(pred),dim=1)
            prediction = prediction.data.cpu().numpy()[0]

            prediction[prediction>0] = 255 
            prediction[prediction<=0] = 0

            prediction = prediction.astype('uint8')

            prediction = imfill(prediction)

            mask = Image.fromarray(prediction)
            mask.save(savePath)

            tq.update(1)
        tq.close()


def predictMats(model,matPath): # 分割.mat文件内的影像数据

    # 建议先测试原有.mat文件的格式，仅支持读取原图.mat文件
    
    def predict(model,modelPath,data,matPath,name): # 预测函数

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mask = np.zeros((len(data),data.shape[-1],data.shape[-1]))

        
        with torch.no_grad():

            model.to(device)
            model.load_state_dict(torch.load(modelPath,map_location=device)) # 预训练模型读取
            model.eval()

            tq = tqdm(total=len(data))

            for i in range(len(data)):

                tq.set_description(f'Predicting {name}')

                img = data[i].T

                # out = np.zeros(img.shape,dtype='uint8')
                # out = cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX) # 将非uint8数据类型的归一化到0-255区间内

                img = Image.fromarray(img,'RGB') # 将单通道改为三通道
                img = np.array(img) #.astype('uint8')
                img = torch.from_numpy(img)
                img = img.view(1,3,scale[0],scale[1]) # batch化处理
                img = transforms.ToTensor(img.copy()).float()

                if torch.cuda.is_available(): # GPU加速
                    img = img.to(device,dtype=torch.float32)

                pred_,pred =model(img) # 预测结果

                prediction = torch.argmax(torch.exp(pred),dim=1) # 预测结果转化
                prediction = prediction.data.cpu().numpy()[0]

                prediction[prediction>0] = 255 # 缩放处理
                prediction[prediction<=0] = 0

                prediction = imfill(prediction)
                
                mask[i] = prediction.astype('uint8')

                tq.update(1)

            tq.close()
            sio.savemat(matPath+'/'+name+'_mask.mat',{'Mask':mask})

    matList = glob.glob(matPath + '/*') # 读取.mat文件列表
    names = os.listdir(matList) # 读取文件名，便于保存结果影像

    for idx,mat in enumerate(matList):

        data = sio.loadmat(mat) # 读取.mat文件

        for key in data.keys(): # 查找影像文件
            if type(data[key])==np.ndarray and len(data[key].shape)>2:
                data = data[key].T

        dataScale = data.shape[-2:]

        modelPath = findmodel(dataScale)

        print('Predicting...')
        if data.shape==3:
            predict(model,modelPath,data,matPath,names[idx])
        else:
            predict(model,modelPath,data[0],matPath,names[idx])
            

def predictDicom(model,dcmPath): # 直接分割影像数据
    
    # 以便于单次批量分割不同病例的多时序多序列影像文件
    # 建议DCM文件目录格式：
    #     DCM 
    #       --> 病例文件夹 
    #           --> 不同时间拍摄影像文件夹
    #                --> 不同参数影像序列 
    #                     --> dcm文件列表
    # 如文件结构不同，可根据需要自行调整相关代码

    patients = glob(dcmPath+'/*')
    
    for pn,p in enumerate(patients):
        timeList = glob(p+'/*')
        for tn,t in enumerate(timeList):
            allDcmList = glob(t + '/*/*.dcm')
            dcmList = glob(t +'/1/*.dcm')
            allDcmList.sort()
            dcmList.sort()
            
            mask = predict(model,dcmList,pn,tn) # 模型生成mask数据

            maskPath = t + '/mask.mat' # 保存mask文件
            sio.savemat(maskPath,{'mask':mask})
            
            for idx,dcm in enumerate(allDcmList): # 分割所有序列

                lnum = idx//len(mask) + 1 # 第几个序列序的dcm列表
                midx = idx%len(mask) # mask索引

                maskpath = t+'/'+str(lnum)+'/breast' # 分割完影像保存位置
                imgpath = t+'/'+str(lnum)+'/imgset'
                if not os.path.exists(maskpath):
                    os.mkdir(maskpath)
                    os.mkdir(imgpath)

                dcm = pdc.dcmread(dcm)
                dcm = dcm.pixel_array

                if midx==0:
                    Breast = np.zeros((len(mask),dcm.shape[0],dcm.shape[1]),dtype=np.uint16) # 保存分割完的.mat文件
                    imgset = np.zeros(mask.shape,dtype='uint16')


                #resize = transforms.Resize((dcm.shape[0],dcm.shape[1]))
                plt.imsave(imgpath+f'/{midx}.jpg',dcm,cmap='gray')

                maski = mask[midx]                
                maski = np.array(maski,dtype='uint16')
                maski[maski==255]=1

                img = dcm*maski # 点乘分割

                Breast[midx] = img
                if midx==(len(mask)-1):
                    sio.savemat(maskpath+'/Breast.mat',{'Breast':Breast})

                plt.imsave(maskpath+f'/{midx}.jpg',img,cmap='gray') # 保存图片

            print('Patient ',pn+1,"'s ",tn+1,' Series Complete!')


def predict(model,dcmList,pn,tn):

    # 用于分割单个参数序列的预测函数
    # 可用测试单个病例的单个参数序列的dicom文件的分割
    # 参数：
    #   model : 用于分割的CPFNet模型
    #   dcmList : dicom文件路径列表，可用glob库中glob函数获取
    #   pn : 批量分割时传入的病例序号，单序列时随便传入，多序列时由 predictDicom() 函数自动传入
    #   tn : 该病例的拍摄影像时间序号，同上
    #   
    # return : 用于该列表下所有的dicom文件分割的mask数据
    #   mask : 类型 numpy.ndarray 数据类型uint8
    #          形状：shape = (文件列表长度，影像规格) 如：(80,512,512)
    #          mask 数据索引对应传入的文件列表索引，即 mask[i] 为 dcmList[i] 的mask数据
    #
    #   获取分割影像：
    #   mask[mask>0] = 1
    #   originimg = pydicom.dcmread(dcmList[i]).pixel_array
    #   breastArea = originimg * mask[i] 
    #   此处注意应数据类型转换 mask为uint8类型，originimg可能为uint16
    #   若需生成mask.mat,请自行操作

    try:
        sample = pdc.dcmread(dcmList[0])
    except Exception:
        raise Exception('DCM列表为空！')
    
    shape = sample.pixel_array.shape
    print(shape)
    modelPath = findmodel(shape)

    mask = np.zeros((len(dcmList),shape[0],shape[1]),dtype='uint8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():

        model.to(device)
        model.load_state_dict(torch.load(modelPath,map_location=device)) # 预训练模型读取
        model.eval()
        
        tq = tqdm(total=len(dcmList)) # 进度展示

        to_tensor = transforms.ToTensor()
        #resize = transforms.Resize(size=(384,384))
        for idx,dcm in enumerate(dcmList):
            tq.set_description(f"Patient {pn}'s {tn}th Series")

            dcm = pdc.dcmread(dcm) # 读取dcm文件
            dcm = dcm.pixel_array # 读取影像数据

            img = Image.fromarray(dcm) # 三通道转化
            img = img.convert('RGB')

            #img = resize(img)

            img = np.array(img)

            out = np.zeros(img.shape,dtype='uint8') # 将uint16数据归一化到uint8
            img = cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)

            img = img.astype('uint8')

            img = to_tensor(img) # tensor化处理
            img = img.view(1,3,img.shape[1],img.shape[2]) # batch化处理

            if torch.cuda.is_available(): # gpu加速
                img = img.to(device,dtype=torch.float32)

            pred_,pred =model(img) # 模型预测
            prediction = torch.argmax(torch.exp(pred),dim=1)
            prediction = prediction.data.cpu().numpy()[0] # 读取gpu数据

            prediction[prediction>0] = 255 # one-hot处理
            prediction[prediction<=0] = 0

            prediction = prediction.astype('uint8')

            mask[idx] = prediction # 生成mask数据

            tq.update(1)

        tq.close()

    return mask


def findmodel(scale): # 查找所需的大小的模型

    trainedPath = 'trainedModel'
    modelList = glob(trainedPath + '/*.pth')
    modelPath = ''

    for diction in modelList:
        if diction.split('_')[-2] == str(scale[0]):
            modelPath = diction
            break 
    if modelPath=='':
        raise TypeError('使用该图片规模的模型尚未获得，请重新设置Scale值，或重新训练模型。') 

    return modelPath


def imfill(img): # 孔洞填充处理
    
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    if im_out.all():
        return im_floodfill_inv
    

def main(mode='img',scale=(384,384)): 

    # 主函数代码
    # 参数：
    #     mode : 分割模式选择
    #         img : 分割图片文件，只生成mask图片文件
    #         mat : 分割.mat文件
    #         dcm : 直接分割所有病例所有影像

    imgPath = 'data/test/imgs' # 图片路径（用作测试）
    matPath = 'data/test/mats' # 
    dcmPath = 'data/test/dcms' # 病例影像文件路径，只要dcms目录下的文件格式正确，就可以单次批量分割，
                               # 如有其他目录格式需要可自行修改代码，参考predictDicom()函数

    model = CPFNet(out_planes=2)

    if mode == 'img':
        predictImgs(model,imgPath,scale) # 图片文件分割函数

    elif mode == 'mat':
        predictMats(model,matPath) # .mat文件分割函数

    elif mode=='dcm':
        predictDicom(model,dcmPath) # 分割所有病例所有时序所有参数序列Dicom文件
        
    # 其他根据需要自行coding


if __name__=='__main__':

    main(mode='dcm')
    
    
    














































