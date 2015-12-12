import cv2
import numpy as np
from glob import glob
import itertools as it
import math as m
from operator import add, truediv
from numpy.linalg import norm
import xml.etree.ElementTree as ET
import pickle
import re


dimCell = 8 # size of cell
dimBlock = 2 # size of block (2x2)Cells
nbins = 9
Err = 0.0000000001


labels = []
samples = []
samples_test = []
labels_test = []

# svm_params = dict( kernel_type = cv2.SVM_LINEAR,
#                     svm_type = cv2.SVM_C_SVC,
#                     C=2.67, gamma=5.383 )

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells


def load_image(fn):

    # Llegim imatge
    img = cv2.imread(fn)
    #image_mean = np.mean(img)
    #img_norm =  img / image_mean
    # print img
    # Normalitzem imatge
    # sum = np.sum(img)
    # print sum

    # if sum > 0.:
    #     return img / sum
    # else:
    #     return img
    # print img.shape
    return img


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)



hog = cv2.HOGDescriptor()
# svm = cv2.SVM()
counter = 0
train_pos = glob('train/pos/*')
train_neg = glob('train/neg/*')
test_pos = glob('test/pos/*')
test_neg = glob('test/neg/*')
train_mit = glob('MIT/Train/*')
test_mit = glob('MIT/Test/*')
set_mit = glob('MIT/*')
sec = glob('foto_prova/*')
# train_prova_neg = glob('prova/neg/*')
iteracion = 0

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
for fn in it.chain(sec):

    try:
        # Retornem imatge en escala de grisos normalitzada
        img = load_image(fn)

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            # labelopencv.pc_test.append([1.])
            # svm = pickle.load(open("svm.pickle"))
            # hog.setSVMDetector( np.array(svm) )
            # del svm
            found, w = hog.detectMultiScale(img)
            found_filtered = []
            for ri, r in enumerate(found):
                for qi, q in enumerate(found):
                    if ri != qi and inside(r, q):
                        break
                else:
                    found_filtered.append(r)
            draw_detections(img, found)

            draw_detections(img, found_filtered, 3)
            cv2.imshow('img', img)
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
               break
    except:
        print 'loading error'
        continue
