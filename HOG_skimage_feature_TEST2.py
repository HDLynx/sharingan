import numpy as np
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter
from glob import glob
import itertools as it
import cv2

def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), visualise=False, normalise=False):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the HOG.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    image = np.atleast_2d(image)

    """
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if image.ndim > 3:
        raise ValueError("Currently only supports grey-level images")

    if normalise:
        image = sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 90

    sy, sx = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    for i in range(orientations):
        #create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, 0)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)

        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cy, cx))[cy/2::cy, cx/2::cx]


    # now for each cell, compute the histogram
    #orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))

    radius = min(cx, cy) // 2 - 1
    hog_image = None
    if visualise:
        hog_image = np.zeros((sy, sx), dtype=float)

    if visualise:
        from skimage import draw

        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o in range(orientations):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / orientations * np.pi)
                    dy = radius * sin(float(o) / orientations * np.pi)
                    rr, cc = draw.line_aa(centre[0] - int(dx), centre[1] - int(dy),
                                            centre[0] + int(dx), centre[1] + int(dy))
                    hog_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / sqrt(block.sum() ** 2 + eps)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if visualise:
        return normalised_blocks.ravel(), hog_image
    else:
        return normalised_blocks.ravel()

# train_pos = glob('train/pos/*')
# train_neg = glob('train/neg/*')
# test_pos = glob('test/pos/*')
# test_neg = glob('test/neg/*')
# train_mit = glob('MIT/train/*')
# test_mit = glob('MIT/test/*')
# set_mit = glob('MIT/*')
# train_pos2 = glob('96X160H96/Train/pos/*')
# train_MIT = glob('MIT/*')
# train_neg2 = glob('samples_neg/*')
# test_neg2 = glob('train64_128/neg/*')
test_pos1 = glob('70X134H96/Test/pos/*')
test_pos2 = glob('test/pos/*')
test_neg1 = glob('test_neg/*')
test_neg2 = glob('train64_128/neg/*')
sample_train = []
labels_train = []
sample_test = []
labels_test = []
svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
counter = 0
# for fn in it.chain(train_pos):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
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
#             labels_train.append([1.])
#     except:
#         print 'loading error'
#         continue
#     sample = hog(img)
#     sample_train.append(sample)
#     counter += 1
#     print counter
#
# for fn in it.chain(train_MIT):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
#         # height, width = img.shape
#         # if width > height:
#         #     img = cv2.transpose(img)
#         # img = cv2.resize(img, (64, 128))
#         height, width = img.shape
#
#         if img is None:
#             print 'Failed to load image file:', fn
#             continue
#         else:
#             labels_train.append([1.])
#     except:
#         print 'loading error'
#         continue
#
#     sample_train.append(hog(img))
#     counter += 1
#     print counter
#
# for fn in it.chain(train_neg):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
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
#             labels_train.append([-1.])
#     except:
#         print 'loading error'
#         continue
#
#     sample_train.append(hog(img))
#     counter += 1
#     print counter
# # labels_train[0] = [0.]
# sample_train = np.float32(sample_train)
# labels_train = np.float32(labels_train)
#
svm = cv2.SVM()

svm.load('svm_Skimage.dat')
# for fn in it.chain(test_pos):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
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
#
#     sample_test.append(hog(img))
# for fn in it.chain(test_neg):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
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
#
#     sample_test.append(hog(img))
# for fn in it.chain(test_pos):
#
#     # iteracion += 1
#     # if iteracion == 10:
#     #     break
#     try:
#         # Retornem imatge en escala de grisos normalitzada
#         img = cv2.imread(fn, 0)
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
#
#     sample_test.append(hog(img))


for fn in it.chain(test_pos1):

    # iteracion += 1
    # if iteracion == 10:
    #     break
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = cv2.imread(fn, 0)
        height, width = img.shape
        if width > height:
            img = cv2.transpose(img)
        img = cv2.resize(img, (64, 128))
        height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels_test.append([1.])
    except:
        print 'loading error'
        continue

    sample_test.append(hog(img))
for fn in it.chain(test_pos2):

    # iteracion += 1
    # if iteracion == 10:
    #     break
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = cv2.imread(fn, 0)
        height, width = img.shape
        if width > height:
            img = cv2.transpose(img)
        img = cv2.resize(img, (64, 128))
        height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels_test.append([1.])
    except:
        print 'loading error'
        continue

    sample_test.append(hog(img))
for fn in it.chain(test_neg1):

    # iteracion += 1
    # if iteracion == 10:
    #     break
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = cv2.imread(fn, 0)
        height, width = img.shape
        if width > height:
            img = cv2.transpose(img)
        img = cv2.resize(img, (64, 128))
        height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels_test.append([-1.])
    except:
        print 'loading error'
        continue

    sample_test.append(hog(img))
for fn in it.chain(test_neg2):

    # iteracion += 1
    # if iteracion == 10:
    #     break
    try:
        # Retornem imatge en escala de grisos normalitzada
        img = cv2.imread(fn, 0)
        height, width = img.shape
        if width > height:
            img = cv2.transpose(img)
        img = cv2.resize(img, (64, 128))
        height, width = img.shape

        if img is None:
            print 'Failed to load image file:', fn
            continue
        else:
            labels_test.append([-1.])
    except:
        print 'loading error'
        continue

    sample_test.append(hog(img))
samples_test = np.float32(sample_test)
labels_test = np.float32(labels_test)

resp = svm.predict_all(samples_test)

mask = resp == labels_test
correct = np.count_nonzero(mask)
print correct*100.0/resp.size,'%'