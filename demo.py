# encoding: utf-8
import sys
caffe_path = '/home/s02/fyk/frcnn'
sys.path.insert(0, "%s/python"%caffe_path)
import caffe # import first to avoid problems like "CUDNN_STATUS_BAD_PARAM"
sys.path.insert(0, "./CTPN/tools")
sys.path.insert(1, "./CTPN/src")
sys.path.append("./crnn.pytorch")

from ctpnport import CTPNDetector
from crnnport import CRNNRecognizer
import time
import cv2

use_gpu = False
#use_gpu = True
base_dir = './models/'
demo_dir = '/home/s02/hgf/text-recog/sceneReco/test/'
gpu_id = -1
if use_gpu: gpu_id = 0
#    model_path = base_dir + 'netCRNN63.pth'
#else:
# CPU model can be used for both CPU/GPU
model_path = base_dir + 'netCRNNcpu.pth'
#model_path = base_dir + 'crnn.pth'
# another one is crnn.pth

NET_DEF_FILE = base_dir + "CTPN/deploy.prototxt"
MODEL_FILE = base_dir + "CTPN/ctpn_trained_model.caffemodel"

#ctpn
ctpn_detector = CTPNDetector(NET_DEF_FILE, MODEL_FILE, caffe_path)
#crnn
crnn_recog = CRNNRecognizer(model_path)

#timer=Timer()
print "\ninput exit break\n"
while 1 :
    #im_name = raw_input("\nplease input file name:")
    im_name = 'image_8.jpg'
    if im_name == "exit":
       break
    im_path = demo_dir + im_name
    im = cv2.imread(im_path)
    if im is None:
      continue
    #timer.tic()
    start = time.time()
    text_lines, resize_im, resize_ratio = ctpn_detector.getCharBlock(im, gpu_id)
    print 'boxes:',len(text_lines)
    text_recs = ctpn_detector.convert_bbox(text_lines)
    print text_recs
    texts = crnn_recog.crnnRec(resize_im,text_recs, use_gpu)
    print texts
    end = time.time()
    #print "Time: %f"%timer.toc()
    print "Time ms: %f"%(end - start)
    box_im, text_recs = ctpn_detector.draw_boxes8(resize_im,text_lines, is_display=False)
    cv2.imwrite("out.jpg", box_im)
    break
    #cv2.waitKey(0)


