# coding=utf-8
import sys
import numpy as np
from matplotlib import cm
import cv2

class cfg:
    MEAN=np.float32([102.9801, 115.9465, 122.7717])
    TEST_GPU_ID=0
    SCALE=600
    MAX_SCALE=1000

    LINE_MIN_SCORE=0.7
    TEXT_PROPOSALS_MIN_SCORE=0.7
    TEXT_PROPOSALS_NMS_THRESH=0.3
    MAX_HORIZONTAL_GAP=50
    TEXT_LINE_NMS_THRESH=0.3
    MIN_NUM_PROPOSALS=2
    MIN_RATIO=1.2
    MIN_V_OVERLAPS=0.7
    MIN_SIZE_SIM=0.7
    TEXT_PROPOSALS_WIDTH=16

#sys.path.insert(0, "./CTPN/tools")
#sys.path.insert(0, "./CTPN/src")
#import os.path as osp
#from utils.timer import Timer

class CTPNDetector:

    def __init__(self, NET_DEF_FILE, MODEL_FILE, caffe_path):
        sys.path.insert(0, "%s/python"%caffe_path)
        import caffe
        from other import draw_boxes, resize_im, CaffeModel 
        from detectors import TextProposalDetector, TextDetector
        sys.path.remove("%s/python"%caffe_path)
        #def ctpnSource(NET_DEF_FILE, MODEL_FILE, use_gpu):
        #NET_DEF_FILE = "CTPN/models/deploy.prototxt"
        #MODEL_FILE = "CTPN/models/ctpn_trained_model.caffemodel"
        self.caffe = caffe
        #if use_gpu:
        #    caffe.set_mode_gpu()
        #    caffe.set_device(cfg.TEST_GPU_ID)
        #else:
        #    caffe.set_mode_cpu()

        # initialize the detectors
        text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
        self.text_detector = TextDetector(text_proposals_detector)
        self.resize_im = resize_im
        self.draw_boxes = draw_boxes
        #return text_detector
    
    def getCharBlock(self, im, gpu_id=0):
        if gpu_id < 0:
            self.caffe.set_mode_cpu()
        else:
            self.caffe.set_mode_gpu()
            self.caffe.set_device(gpu_id)

        resize_im, resize_ratio = self.resize_im(im, cfg.SCALE, cfg.MAX_SCALE)
        #print "resize", f
        #cv2.imshow("src", im)
        tmp = resize_im.copy()
        #timer=Timer()
        #timer.tic()
        text_lines = self.text_detector.detect(tmp)
    
        #print "Number of the detected text lines: %s"%len(text_lines)
        #print "Time: %f"%timer.toc()
        return text_lines, resize_im, resize_ratio

    # this is deprecated
    def convert_bbox(self, bboxes):
        text_recs = np.zeros((len(bboxes), 8), np.int)
        index = 0
        for box in bboxes:
            b1 = box[6] - box[7] / 2
            b2 = box[6] + box[7] / 2
            x1 = box[0]
            y1 = box[5] * box[0] + b1
            x2 = box[2]
            y2 = box[5] * box[2] + b1
            x3 = box[0]
            y3 = box[5] * box[0] + b2
            x4 = box[2]
            y4 = box[5] * box[2] + b2
            
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX*disX + disY*disY)
            fTmp0 = y3 - y1
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1*disX / width)
            y = np.fabs(fTmp1*disY / width)
            if box[5] < 0:
               x1 -= x
               y1 += y
               x4 += x
               y4 -= y
            else:
               x2 += x
               y2 += y
               x3 -= x
               y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            index = index + 1
        return text_recs
    def draw_boxes8(self, im, bboxes, is_display=True, color=None, caption="Image", wait=True):
        """
            boxes: bounding boxes
        """
        text_recs=np.zeros((len(bboxes), 8), np.int)
    
        im=im.copy()
        index = 0
        for box in bboxes:
            if color==None:
                if len(box)==8 or len(box)==9:
                    c=tuple(cm.jet([box[-1]])[0, 2::-1]*255)
                else:
                    c=tuple(np.random.randint(0, 256, 3))
            else:
                c=color
            
            b1 = box[6] - box[7] / 2
            b2 = box[6] + box[7] / 2
            x1 = box[0]
            y1 = box[5] * box[0] + b1
            x2 = box[2]
            y2 = box[5] * box[2] + b1
            x3 = box[0]
            y3 = box[5] * box[0] + b2
            x4 = box[2]
            y4 = box[5] * box[2] + b2
            
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX*disX + disY*disY)
            fTmp0 = y3 - y1
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1*disX / width)
            y = np.fabs(fTmp1*disY / width)
            if box[5] < 0:
               x1 -= x
               y1 += y
               x4 += x
               y4 -= y
            else:
               x2 += x
               y2 += y
               x3 -= x
               y3 -= y
            cv2.line(im,(int(x1),int(y1)),(int(x2),int(y2)),c,2)
            cv2.line(im,(int(x1),int(y1)),(int(x3),int(y3)),c,2)
            cv2.line(im,(int(x4),int(y4)),(int(x2),int(y2)),c,2)
            cv2.line(im,(int(x3),int(y3)),(int(x4),int(y4)),c,2)
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            index = index + 1
            #cv2.rectangle(im, tuple(box[:2]), tuple(box[2:4]), c,2)  
        if is_display:
            cv2.imshow('result', im)
            #if wait:
                #cv2.waitKey(0)
        return im, text_recs
