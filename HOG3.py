import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
# from skimage import data
  
image = cv2.imread("MIT/Train/per00001.ppm", cv2.COLOR_BAYER_RG2GRAY)
image = cv2.resize(image, (64, 128))
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
  
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
  
ax1.axis('off')
ax1.imshow('hola',image)
ax1.set_title('Input image')
  
# Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
  
ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.imshow(hog_image, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
  