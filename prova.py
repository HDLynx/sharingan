import cv2
import numpy as np
from glob import glob
import itertools as it
import math as m
from numpy.linalg import norm
import xml.etree.ElementTree as ET

SZ = 8 # size of cell
BSZ = 16 # size of block

labels = []
width = 0
height = 0


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
    cells = split2d(img, (SZ, SZ))
    return cells
def getGradients(img):
    height, width = img.shape
    # gradient_img = np.zeros((height,width))
    line_gradients = []
    gradient_img = []
    for x in range(1,height-1):
        for y in range(1, width-1):

            px = img[x+1,y] - img[x-1,y]
            py = img[x,y+1] - img[x,y-1]

            gradient = m.sqrt((px**2) + (py**2))
            angle = np.arctan(py/px)
            line_gradients.append([gradient, angle])
        gradient_img.append(line_gradients)
        line_gradients = []
    return gradient_img
def preprocess_hog(cells):
    samples = []

    for cell in cells:
        print cell
        # for x in range(0,SZ):
        #     for y in range(0, SZ):
        #
        #         px = cell[x+1,y] - cell[x-1,y]
        #         py = cell[x,y+1] - cell[x,y-1]
        #
        #         gradient = m.sqrt((px**2) + (py**2))
        #         angle = np.arctan(py/px)
        #         gradient_img.append([gradient,angle])
    return np.float32(samples)

hog = cv2.HOGDescriptor()
svm = cv2.SVM()

train_pos = glob('train/pos/*')
train_neg = glob('train/neg/*')
test_pos = glob('test/pos/*')
test_neg = glob('test/neg/*')
test_mit = glob('MIT/*')
test_prova = glob('prova/*')

for fn in it.chain(test_prova):
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = load_image(fn)
        height, width = img.shape
        # print img
        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels.append([1.])
    except:
        print 'loading error'
        continue
    # Calculem el gradient de la imatge
    gradient_img = getGradients(img)

    # cells = split2d(gradient_img, (SZ, SZ))

    # Separem imatges en cells
    # cells = getCells(img)
    # # Agafem
    # samples = preprocess_hog(cells)




