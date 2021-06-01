import torch
import glob
import os
import cv2
from torch import functional
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
# from imgaug import augmenters as iaa
# import imgaug as ia
import random
import matplotlib.pyplot as plt


class Breast(torch.utils.data.Dataset):
    def __init__(self, dataset_path, scale, mode='tr'):
        super().__init__()
        self.mode = mode
        self.img_path=dataset_path+'/imgs'
        if self.mode=='ts':
            self.mask_path=dataset_path+'/mask'
        else:
            self.mask_path=dataset_path+'/labels'
        
        self.image_lists = glob.glob(os.path.join(self.img_path) + '/*.jpg')
        self.label_lists = glob.glob(os.path.join(self.mask_path) + '/*.jpg')

        # self.flip =iaa.SomeOf((2,4),[
        #      iaa.Fliplr(0.5),
        #      iaa.Flipud(0.5),
        #      iaa.Affine(rotate=(-30, 30)),
        #      iaa.AdditiveGaussianNoise(scale=(0.0,0.08*255))], random_order=True)
        # resize

    #     InterpolationMode:
    #     inverse_modes_mapping = {
    #     0: InterpolationMode.NEAREST,
    #     2: InterpolationMode.BILINEAR,
    #     3: InterpolationMode.BICUBIC,
    #     4: InterpolationMode.BOX,
    #     5: InterpolationMode.HAMMING,
    #     1: InterpolationMode.LANCZOS,}


        self.resize_label = transforms.Resize(scale,Image.BILINEAR)# interpolation = F.InterpolationMode.NEAREST)
        self.resize_img = transforms.Resize(scale,Image.NEAREST)# interpolation = F.InterpolationMode.BILINEAR )
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        #读取影像&标签
        p1 = np.random.randint(0,2)
        p2 = np.random.randint(0,2)

        # im_aug = transforms.Compose([
        #                           transforms.RandomHorizontalFlip(p1),
        #                           transforms.RandomVerticalFlip(p2),
        #                           #transforms.RandomRotation(10, resample=False, expand=False, center=None),
        #                           transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
        #                           ])

        # seed = random.randint(0,500000)

        img = Image.open(self.image_lists[index])
        label = Image.open(self.label_lists[index]).convert('L')

        # resize
        img = self.resize_img(img)
        label = self.resize_label(label)

        # random.seed(seed)
        # img = im_aug(img)
        # random.seed(seed)
        # label = im_aug(label)


        img = np.asarray(img)
        label = np.array(label)
        

        #one-hot处理
        label[label<64] = 0
        label[label>=64] = 255

        label = self.imfill(label)
        

        img = self.to_tensor(img.copy()).float()
        label = self.to_tensor(label.copy()).float()[0]

        return img, label


    def __len__(self):
        return len(self.image_lists)

    def imfill(self,img):
    
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
         

if __name__ == '__main__':

    dataPath = 'data/train'
    data = Breast(dataPath, (512, 512),mode='tr')

    from torch.utils.data import DataLoader
    dataloader_test = DataLoader(
                data,
                batch_size=1,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True )

    for i, (img,label )in enumerate(dataloader_test):
        print(img.shape)
        print(label.shape)
        print(label.max())
        print(img.max())
        break

    imgl = np.asarray(img[0][0])
    la = np.asarray(label[0])
    print(imgl.max())
    # plt.imsave(imgl,'gray')
    # plt.imsave(la,'gray')
    plt.figure(figsize=(15,12))
    plt.subplot(1,2,1)
    plt.imshow(imgl,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(la,'gray')
    plt.show()


