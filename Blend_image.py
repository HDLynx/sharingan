import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('zoro2.png')
img2 = cv2.imread('sharingan.jpg')
print img1.size
print img2.size

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

x = np.uint8([250])
y = np.uint8([6])

print x
print y
print x + y
print x - y
print cv2.add(x,y)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
