import cv2
import numpy as np
from glob import glob
import itertools as it

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 0
winSigma = -1
histogramNormType = 0
# L2HysThreshold = 2.0000000000000001e-01
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64
winStride = (8,8)
padding = (8,8)
locations = ((0,0),)
train_set = []
test_set = []
responses_train = []
responses_test = []

train_pos = glob('96X160H96/Train/pos/*')
train_MIT = glob('MIT/*')
train_neg = glob('samples_neg/*')
test_neg = glob('train64_128/neg/*')


hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
svm = cv2.SVM()
counter = 0

# svm.load('svmlight.dat')
# svm.save('svmlight.xml')

for fn in it.chain(train_pos):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn, 0)
        # img = cv2.resize(img, (64, 128))
        cv2.imshow('img', img)
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img)
    # hist_train = hog.compute(img)
    train_set.append(hist_train)
    responses_train.append([1.])
for fn in it.chain(train_MIT):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn, 0)
        # img = cv2.resize(img, (64, 128))
        cv2.imshow('img', img)
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img)
    # hist_train = hog.compute(img)
    train_set.append(hist_train)
    responses_train.append([1.])

for fn in it.chain(train_neg):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn, 0)
        # img = cv2.resize(img, (64, 128))
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img)
    # hist_train = hog.compute(img)
    train_set.append(hist_train)
    responses_train.append([-1.])
    print counter
    counter += 1

train_set = np.float32(train_set)
responses_train = np.float32(responses_train)
svm.train(train_set, responses_train, params=svm_params)
svm.save('svmlight.dat')
svm.save('svmlight.xml')

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
    hist_train = hog.compute(img,winStride,padding,locations)
    # hist_train = hog.compute(img)
    test_set.append(hist_train)
    responses_test.append([-1.])

test_set = np.float32(test_set)
responses_test = np.float32(responses_test)

resp = svm.predict_all(test_set)

mask = resp == responses_test
correct = np.count_nonzero(mask)

print correct*100.0/resp.size,'%'
# hog.save('prova.dat','vector_descriptors')

# Primera ronda un 99'507%
# Segona ronda un 99,67%

