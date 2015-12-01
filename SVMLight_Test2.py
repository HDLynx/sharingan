import cv2
import numpy as np
from glob import glob
import itertools as it
import xml.etree.ElementTree as ET
import pickle
import re


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

test_pos1 = glob('70X134H96/Test/pos/*')
test_pos2 = glob('test/pos/*')
test_neg1 = glob('test_neg/*')
test_neg2 = glob('train64_128/neg/*')

hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
svm = cv2.SVM()
counter = 0


svm.load('svmlight.dat')
# svm.save('svmlight.xml')

# tree = ET.parse('svmlight.xml')
# root = tree.getroot()
# # now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
# SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
# rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
# svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
# svmvec.append(-rho)
# pickle.dump(svmvec, open("svm.pickle", 'w'))
# svm = pickle.load(open("svm.pickle"))
# tamany = hog.getDescriptorSize()
# # hog.setSize(34021)
# svm_l = cv2.HOGDescriptor_getDefaultPeopleDetector()
# hog.setSVMDetector( np.array(svm) )
# for fn in it.chain(train_pos):
#     #print fn, ' - ',
#     try:
#         img = cv2.imread(fn, 0)
#         img = cv2.resize(img, (64, 128))
#         cv2.imshow('img', img)
#         if img is None:
#             print 'Failed to load image file:', fn
#             continue
#     except:
#         print 'loading error'
#         continue
#     hist_train = hog.compute(img)
#     # hist_train = hog.compute(img)
#     train_set.append(hist_train)
#     responses_train.append([1.])
#
# for fn in it.chain(train_neg):
#     #print fn, ' - ',
#     try:
#         img = cv2.imread(fn, 0)
#         # img = cv2.resize(img, (64, 128))
#         if img is None:
#             print 'Failed to load image file:', fn
#             continue
#     except:
#         print 'loading error'
#         continue
#     hist_train = hog.compute(img)
#     # hist_train = hog.compute(img)
#     train_set.append(hist_train)
#     responses_train.append([0.])
#     print counter
#     counter += 1
#
# train_set = np.float32(train_set)
# responses_train = np.float32(responses_train)
# svm.train(train_set, responses_train, params=svm_params)
# svm.save('svmlight.dat')
# svm.save('svmlight.xml')
for fn in it.chain(test_pos1):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        img = cv2.resize(img, (64, 128))
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img, winStride,padding,locations)
    # hist_train = hog.compute(img)
    test_set.append(hist_train)
    responses_test.append([1.])
for fn in it.chain(test_pos2):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        img = cv2.resize(img, (64, 128))
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img, winStride,padding,locations)
    # hist_train = hog.compute(img)
    test_set.append(hist_train)
    responses_test.append([1.])
for fn in it.chain(test_neg1):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        img = cv2.resize(img, (64, 128))
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img, winStride,padding,locations)
    # hist_train = hog.compute(img)
    test_set.append(hist_train)
    responses_test.append([-1.])
for fn in it.chain(test_neg2):
    #print fn, ' - ',
    try:
        img = cv2.imread(fn)
        img = cv2.resize(img, (64, 128))
        if img is None:
            print 'Failed to load image file:', fn
            continue
    except:
        print 'loading error'
        continue
    hist_train = hog.compute(img, winStride,padding,locations)
    # hist_train = hog.compute(img)
    test_set.append(hist_train)
    responses_test.append([-1.])

test_set = np.float32(test_set)
responses_test = np.float32(responses_test)

resp = svm.predict_all(test_set)

mask = resp == responses_test
correct = np.count_nonzero(mask)
cont = 0
total = 0
for i in range(len(resp)):
    if resp[i] != responses_test[i]:
        if responses_test[i] == 1.:
            cont += 1
        print responses_test[i], '----', resp[i]
        total += 1
print correct*100.0/resp.size,'%'
print len(responses_test), '=', total, 'errores ---->', cont, 'personas no identificadas'
print len(test_pos1) + len(test_pos2)
# hog.save('prova.dat','vector_descriptors')
