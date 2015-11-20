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

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )

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
    img = cv2.imread(fn, 0)
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


def getCells(img):
    cells = split2d(img, (dimCell, dimCell))
    return cells


def getGradients(img):
    height, width = img.shape
    # gradient_img = np.zeros((height,width))
    line_gradients = []
    gradient_img = []
    counter = 0
    for x in range(0, height):
        for y in range(0, width):
            if x == 0:
                if y == 0 or y == width-1:
                    px = 0.
                    py = 0.
                else:
                    px = img[x,y+1] - img[x,y-1]
                    py = 0.
            elif x == height-1:
                if  y == 0 or y == width-1:
                    px = 0.
                    py = 0.
                else:
                    px = img[x,y+1] - img[x,y-1]
                    py = 0.
            else:
                if y == 0 or y == width-1:
                    px = 0.
                    py = img[x+1,y] - img[x-1,y]
                else:
                    px = img[x,y+1] - img[x,y-1]
                    py = img[x+1,y] - img[x-1,y]

            magnitud = m.sqrt((px**2) + (py**2))
            angle = m.degrees(getAngle(px,py))
            gradient_img.append([magnitud, angle])
    return gradient_img


def getAngle(x,y):
    if  x == 0:
        if y == 0:
            angle = 0.
        else:
            angle = m.pi / 2
    else:
        if y == 0:
            angle = 0.
        else:
            angle = np.arctan(y/x)
    return angle


def calculateHistogramCells(img, height, width):
    nHorizCell = width / dimCell
    nVerticalCell = height / dimCell
    histogramCell = np.zeros(9)
    histogramVector = []

    for j in range(nVerticalCell):
        for i in range(nHorizCell):
            for y in range(j*dimCell, j*dimCell+dimCell):
                for x in range(i*dimCell + y*width, i*dimCell + y*width + dimCell):
                    magnitud, angle = img[x]

                    if (90 >= angle > 70) or angle == -90:
                        percentage1 = ( angle - 70 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[0] += magnitud * percentage1
                        histogramCell[1] += magnitud * percentage2
                    if 70 >= angle > 50:
                        percentage1 = ( angle - 50 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[1] += magnitud * percentage1
                        histogramCell[2] += magnitud * percentage2
                    if 50 >= angle > 30:
                        percentage1 = ( angle - 30 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[2] += magnitud * percentage1
                        histogramCell[3] += magnitud * percentage2
                    if 30 >= angle > 10:
                        percentage1 = ( angle - 10 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[3] += magnitud * percentage1
                        histogramCell[4] += magnitud * percentage2
                    if 10 >= angle > -10:
                        percentage1 = ( angle + 10 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[4] += magnitud * percentage1
                        histogramCell[5] += magnitud * percentage2
                    if -10 >= angle > -30:
                        percentage1 = ( angle +30 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[5] += magnitud * percentage1
                        histogramCell[6] += magnitud * percentage2
                    if -30 >= angle > -50:
                        percentage1 = ( angle + 50 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[6] += magnitud * percentage1
                        histogramCell[7] += magnitud * percentage2
                    if -50 >= angle > -70:
                        percentage1 = ( angle + 70 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[7] += magnitud * percentage1
                        histogramCell[8] += magnitud * percentage2
                    if -70 >= angle > -90:
                        percentage1 = ( angle + 90 ) / 20.
                        percentage2 = 1 - percentage1
                        histogramCell[8] += magnitud * percentage1
                        histogramCell[0] += magnitud * percentage2
                    # print percentage1, percentage2
            histogramVector.append(histogramCell)
            histogramCell = np.zeros(9)
    return histogramVector, nVerticalCell, nHorizCell


def normalizeHistogramsBlocks(vector_hist,width,height):

    dimVert = height - 1
    dimHoriz = width - 1
    resto_j = 0
    resto_i = 0
    blockHistogramVector = []
    sumHistogram = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
    histogramsNormalized = []

    for j in range(dimVert):
        resto_i = 0
        if j != 0:
            resto_j += 1
        for i in range(dimHoriz):
            if i != 0:
                resto_i += 1
            for y in range(j*dimBlock, j*dimBlock + dimBlock):
                control_y = y
                control_y -= resto_j
                for x in range(i*dimBlock + control_y*width, i*dimBlock + control_y*width + dimBlock ):
                    index = x
                    index -= resto_i
                    blockHistogramVector.append(vector_hist[index])
                    sumHistogram = map(add, sumHistogram, vector_hist[index])
            for z in range(len(blockHistogramVector)):
                for w in range(nbins):
                    histogramsNormalized.append((blockHistogramVector[z][w] + Err) / (sumHistogram[w]+Err))

            # histogramsNormalized += blockHistogramVector
            # histogramsNormalized = np.append(histogramsNormalized, blockHistogramVector)
            blockHistogramVector = []
            sumHistogram = [0.,0.,0.,0.,0.,0.,0.,0.,0.]
    # histogramsNormalized = np.float32(histogramsNormalized)
    # histogramsNormalized = np.hstack(histogramsNormalized)
    return histogramsNormalized


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
svm = cv2.SVM()
counter = 0
train_pos = glob('train/pos/*')
train_neg = glob('train/neg/*')
test_pos = glob('test/pos/*')
test_neg = glob('test/neg/*')
train_mit = glob('MIT/Train/*')
test_mit = glob('MIT/Test/*')
set_mit = glob('MIT/*')
# train_prova_pos = glob('prova/pos/*')
# train_prova_neg = glob('prova/neg/*')
iteracion = 0

# svm.load('svm_INRIA_MIT.dat')

# tree = ET.parse('svm_INRIA_MIT.xml')
# root = tree.getroot()
# # now this is really dirty, but after ~3h of fighting OpenCV its what happens :-)
# SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0]
# rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
# svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
# svmvec.append(-rho)
# pickle.dump(svmvec, open("svm.pickle", 'w'))
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
for fn in it.chain(test_pos):

    # iteracion += 1
    # if iteracion == 10:
    #     break
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = load_image(fn)
        # height, width = img.shape
        # if width > height:
        #     img = cv2.transpose(img)
        img = cv2.resize(img, (64, 128))
        # height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            # labels_test.append([1.])
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
    # # Calculem el gradient de la imatge
    # gradient_img = getGradients(img) # Mirar la funcio cartToPolar
    # # Calculem histograma per cells de Ndimensions
    # histogram_vector, nVertCell, nHorizCell = calculateHistogramCells(gradient_img, height, width)
    # # Normalitzem per blocks
    # histogram_vector_normalized = normalizeHistogramsBlocks(histogram_vector, nHorizCell, nVertCell)
    # # histogram_vector_normalized = np.float32(histogram_vector_normalized)
    # samples_test.append(histogram_vector_normalized)
    # counter += 1
    print counter, '----'
# for fn in it.chain(test_neg):
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = load_image(fn)
#         height, width = img.shape
#         if width > height:
#             img = cv2.transpose(img)
#         img = cv2.resize(img, (64, 128))
#         height, width = img.shape
#
#         if img is None:
#             print 'Failed to load image file:', fn
#             continue
#         else:
#             labels_test.append([0.])
#     except:
#         print 'loading error'
#         continue
#     # Calculem el gradient de la imatge
#     gradient_img = getGradients(img) # Mirar la funcio cartToPolar
#     # Calculem histograma per cells de Ndimensions
#     histogram_vector, nVertCell, nHorizCell = calculateHistogramCells(gradient_img, height, width)
#     # Normalitzem per blocks
#     histogram_vector_normalized = normalizeHistogramsBlocks(histogram_vector, nHorizCell, nVertCell)
#     # histogram_vector_normalized = np.float32(histogram_vector_normalized)
#     samples_test.append(histogram_vector_normalized)
#     counter += 1
#
#     print counter, '----'
#
# for fn in it.chain(set_mit):
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = load_image(fn)
#         height, width = img.shape
#         if width > height:
#             img = cv2.transpose(img)
#         img = cv2.resize(img, (64, 128))
#         height, width = img.shape
#
#         if img is None:
#             print 'Failed to load image file:', fn
#             continue
#         else:
#             labels_test.append([1.])
#     except:
#         print 'loading error'
#         continue
#     # Calculem el gradient de la imatge
#     gradient_img = getGradients(img) # Mirar la funcio cartToPolar
#     # Calculem histograma per cells de Ndimensions
#     histogram_vector, nVertCell, nHorizCell = calculateHistogramCells(gradient_img, height, width)
#     # Normalitzem per blocks
#     histogram_vector_normalized = normalizeHistogramsBlocks(histogram_vector, nHorizCell, nVertCell)
#     # histogram_vector_normalized = np.float32(histogram_vector_normalized)
#     samples_test.append(histogram_vector_normalized)
#     counter += 1
#     print counter, '----'
#
# samples_test = np.float32(samples_test)
# labels_test = np.float32(labels_test)
#
# resp = svm.predict_all(samples_test)
#
# mask = resp == labels_test
# correct = np.count_nonzero(mask)
#
# print correct*100.0/resp.size,'%'