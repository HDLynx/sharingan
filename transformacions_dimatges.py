import cv2
import numpy as np

img = cv2.imread('zoro.png')

# augmentar tamany
res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
print height
print width
res2 = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

# translation
M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(width,height))

#rotation

N = cv2.getRotationMatrix2D((width/2,height/2),90,1)
dst2 = cv2.warpAffine(img,N,(width,height))

#affine transform
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
dst3 = cv2.warpAffine(img,M,(width,height))

#Perspective transform

pts3 = np.float32([[200,200],[700,200],[200,1000],[700,1000]])
pts4 = np.float32([[0,0],[300,0],[0,300],[300,300]])

L = cv2.getPerspectiveTransform(pts3,pts4)

dst4 = cv2.warpPerspective(img,L,(300,300))

cv2.imshow('Zoom X2',res)
cv2.imshow('Tranlation x->100 y->50',dst)
cv2.imshow('Rotacio 90',dst2)
cv2.imshow('Affine',dst3)
cv2.imshow('Perspective',dst4)
cv2.waitKey(0)
cv2.destroyAllWindows()
