#coding:utf-8

import random
import torch
from torch.autograd import Variable 
import numpy as np
import os
import crnn_utils
import dataset
from PIL import Image
import models.crnn as crnn
import keys
from math import *
#import mahotas
import cv2

class CRNNRecognizer:

    def __init__(self, model_path, use_gpu=True):
        #def crnnSource(model_path, use_gpu=True):
        alphabet = keys.alphabet # Chinese words
        self.converter = crnn_utils.strLabelConverter(alphabet)
        # note that in https://github.com/bear63/sceneReco support multi GPU.
        # model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cuda()
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256)
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        print('loading pretrained model from %s' % model_path)
        #model_path = './crnn/samples/netCRNN63.pth'
        model.load_state_dict(torch.load(model_path))
        self.model = model
        self.use_gpu = use_gpu
        #return model,converter
        

    def crnnRec(self, im, text_recs):
       texts = []
       index = 0
       for rec in text_recs:
           pt1 = (rec[0],rec[1])
           pt2 = (rec[2],rec[3])
           pt3 = (rec[6],rec[7])
           pt4 = (rec[4],rec[5])
           partImg = self.dumpRotateImage(im,degrees(atan2(pt2[1]-pt1[1],pt2[0]-pt1[0])),pt1,pt2,pt3,pt4)
           #mahotas.imsave('%s.jpg'%index, partImg)
           
    
           image = Image.fromarray(partImg).convert('L')
           #height,width,channel=partImg.shape[:3]
           #print(height,width,channel)
           #print(image.size) 
    
           #image = Image.open('./img/t4.jpg').convert('L')
           scale = image.size[1]*1.0 / 32
           w = image.size[0] / scale
           w = int(w)
           #print(w)
    
           transformer = dataset.resizeNormalize((w, 32))
           image = transformer(image)
           if self.use_gpu and torch.cuda.is_available():
               image = image.cuda()
    
           image = image.view(1, *image.size())
           image = Variable(image)
           self.model.eval()
           preds = self.model(image)
           _, preds = preds.max(2)
           preds = preds.squeeze(0)
           preds = preds.transpose(1, 0).contiguous().view(-1)
           preds_size = Variable(torch.IntTensor([preds.size(0)]))
           raw_pred = self.converter.decode(preds.data, preds_size.data, raw=True)
           sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)
           print('%-20s => %-20s' % (raw_pred, sim_pred))
           #print(index)
           #print(sim_pred)
           index = index + 1
           texts.append(sim_pred)
           
       return texts

    def dumpRotateImage(self, img,degree,pt1,pt2,pt3,pt4):
        height,width=img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)
        
        
        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        imgOut=imgRotation[int(pt1[1]):int(pt3[1]),int(pt1[0]):int(pt3[0])]
        height,width=imgOut.shape[:2]
        return imgOut

