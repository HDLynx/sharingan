import cv2
import numpy as np,sys

im = cv2.imread('zoro.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
cv2.imshow('thres', thresh)
x,y,w,h = cv2.boundingRect(thresh)
img = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow('Contorns', img)
cv2.waitKey(0)
cv2.destroyAllWindows()