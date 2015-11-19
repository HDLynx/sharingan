import cv2
import numpy as np
from glob import glob
import itertools as it
import xml.etree.ElementTree as ET

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
winStride = (8,8)
padding = (8,8)
locations = ((10,20),)
train_set = []
test_set = []
responses_train = []
responses_test = []

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#
# hog = cv2.HOGDescriptor()

svm = cv2.SVM()

train_pos = glob('train/pos/*')
train_neg = glob('train/neg/*')
test_pos = glob('test/pos/*')
test_neg = glob('test/neg/*')
test_mit = glob('MIT/*')
test_prova = glob('prova/aux/*')
train_prova_pos = glob('prova/pos/*')
train_prova_neg = glob('prova/neg/*')
for fn in it.chain(train_pos):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img,winStride,padding,locations)
    # hist_train = hog.compute(img)
    train_set.append(hist_train)
    responses_train.append([1.])

for fn in it.chain(train_neg):
    try:
        img = cv2.imread(fn)
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img,winStride,padding,locations)
    train_set.append(hist_train)
    responses_train.append([0.])

train_set = np.float32(train_set)
responses_train = np.float32(responses_train)

svm.train(train_set,responses_train, params=svm_params)
svm.save('svm_data.dat')


for fn in it.chain(test_pos):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_test = hog.compute(img,winStride,padding,locations)

    test_set.append(hist_test)
    responses_test.append([1.])
for fn in it.chain(test_neg):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)

        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_test = hog.compute(img,winStride,padding,locations)
    test_set.append(hist_test)
    responses_test.append([0.])

test_set = np.float32(test_set)
resp = svm.predict_all(test_set)

responses_test = np.float32(responses_test)

mask = resp == responses_test
correct = np.count_nonzero(mask)

print correct*100.0/resp.size,'%'


