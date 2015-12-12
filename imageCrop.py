import cv2
import numpy as np
from glob import glob
import itertools as it
import pylab

img = cv2.imread('Kalman/1/a21.jpg')
cv2.imshow('img', img)
crop = img[0:600,100:700]
cv2.imshow('crop', crop)
ch = 0xFF & cv2.waitKey()
if ch == 27:
    cv2.destroyAllWindows()
