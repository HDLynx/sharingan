__author__ = 'xavicosp'

import cv2
import numpy
import image
import glob
import itertools as it

train_prova_pos = glob('prova/neg/*')
for fn in it.chain(train_prova_pos):
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = image(fn)
        img = cv2.resize(img, (64, 128))
        height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels.append([1.])
    except:
        print 'loading error'
        continue

