import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('images.png') #Llegim imatge
px = img[100,100] #accedim al pixel 100, 100
print px #Imprimim els valors del pixel. ESTA EN BGR


# accessing only blue pixel
print img.item(100,100,2) #Accedir a pixel utilitzant numpy (pixel_x, pixel_y, BGR '0,1,2')

print img.shape #Retorna les files, columnes, i numero de canals

print img.size #Retorna el numero de pixels

print img.dtype #Retorna el tipus d'imatge (Molts errors en tractament es pel tipus d'imatge, MOLT UTIL)
BLUE = [255,0,0]
ball = img[50:70, 200:250]
img[120:140, 100:150] = ball
replicate = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
while(1):
    cv2.imshow('image',img)
    plt.show()

    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
