import cv2
import numpy as np
from glob import glob
import itertools as it
import pylab
import pydoc

winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 0
winSigma = -1
histogramNormType = 0
# L2HysThreshold = 2.0000000000000001e-01
L2HysThreshold = 0.2
gammaCorrection = 0
nlevels = 64

iteracion = 0
tp = np.zeros((2,1), np.float32)
frame = np.zeros((1000,1600,3), np.uint8)
meas= []
pred= []
iteracion = 0
index = 0
height = 0
height_pred = 0
width = 0
width_pred = 0
crop = []

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
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), thickness)


def draw_predictions(predictions, mesurements):
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(predictions)-1):
        cv2.line(frame, predictions[i], predictions[i+1], (0,255,0),3)
    for i in range(len(mesurements)-1):
        cv2.line(frame, mesurements[i], mesurements[i+1], (0,0,255),3)

    # Plot all the results we got.
    cv2.destroyAllWindows()
    for i in range(len(predictions)):
        x1.append(predictions[i][0])
        y1.append(predictions[i][1])
        x2.append(mesurements[i][0])
        y2.append(mesurements[i][1])
    pylab.plot(x1,y1,'-',x2,y2,':')
    pylab.xlabel('X position')
    pylab.ylabel('Y position')
    pylab.title('Start Pedestrian Position')
    pylab.legend(('Kalman','true'))
    pylab.show()


def init_Kalman_HOG():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    return kalman, hog


def crop_image(width, height):
    # La funcio DetectMultiScale necessita que la imatge tingui un cert tamany per trobar la persona
    # Creem un padding per agrandir la finestra de cerca
    pad_w = int(0.25*width)
    pad_h = int(0.15*height)

    # Localitzem el pixel inicial que indicara on comensar a retallar
    initial_y = pred[index-1][1]-pad_h
    initial_x = pred[index-1][0]-pad_w

    # Si per culpa del padding sortim de la imatge, ens coloquem al limit
    if initial_y < 0:
        height -= initial_y
        initial_y = 0

    if initial_x < 0:
        width -= initial_x
        initial_x = 0

    # Calculem el pixel final
    final_y = pred[index-1][1] + height + pad_h
    final_x = pred[index-1][0] + width + pad_w

    # Si per culpa del padding sortim de la imatge, ens coloquem al limit
    if final_y > img_h:
        initial_y -= (final_y - img_h)
        final_y = img_h
    if final_x > img_w:
        initial_x -= (final_x - img_w)
        final_x = img_w

    # Agafem el tros d'imatge que ens ha donat la previsio
    image_crop = img[initial_y:final_y, initial_x:final_x]

    # Mostrem el crop per comprovar que la zona que ha predit contingui la persona
    # show_image('cropped image', image_crop)

    return image_crop, initial_x, initial_y


def filter_founds(founds):
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)
    return found_filtered


def predict_next_location(index):
    mesurement = np.array([[np.float32(found_filtered[0][0])], [np.float32(found_filtered[0][1])]])
    kalman.correct(mesurement)
    tp = kalman.predict()
    pred.append((int(tp[0]),int(tp[1])))
    meas.append((int(mesurement[0]),int(mesurement[1])))
    index += 1
    # return index


def check_found_size(founds, percentage):
    result = 9999
    index = 0
    filtered = []
    for i in range(len(founds)):
        diff = abs( width_pred - founds[i][2] ) + abs( height_pred - founds[i][3] )
        if diff < result:
            result = diff
            index = i
    filtered = founds[index]
    return filtered

def show_image(name, image):
    cv2.imshow(name, image)
    # ch = 0xFF & cv2.waitKey()
    cv2.waitKey()
    cv2.destroyAllWindows()
    # if ch == 27:
    #     cv2.destroyAllWindows()


# Inicialitzem els objectes Kalman i HOG
kalman, hog = init_Kalman_HOG()

# Llegim totes les imatges que hi han a la carpeta 'Kalman/1'
# train_prova_pos = glob('foto_prova/*')
cap = cv2.VideoCapture(0)
# for fn in it.chain(train_prova_pos):
while (cap.isOpened()):
    e1 = cv2.getTickCount()
    try:
        # Llegim la imatge
        ret, img = cap.read()
        # img = cv2.imread('afegit_marge.png')

        # Guardem en variables l'altura, l'amplada i el nombre de canals
        img_h, img_w, img_c = img.shape

        # Esperem 3 iteracions a utilitzar les prediccions per
        # donar temps a actualitzar-se i que siguin fiables
        if iteracion > 2:
            # Fem el crop de la imatge i retornem el pixel inicial del crop
            # e3 = cv2.getTickCount()

            crop, initial_x, initial_y = crop_image(width, height)

            # e4 = cv2.getTickCount()
            # t = (e3 - e4)/cv2.getTickFrequency()
            # text = str(iteracion) + ' - crop - ' + str(t)
            # print text

            # cv2.imwrite(str(iteracion) + '_crop.png', crop)
        # Comprovem que la imatge s'ha carregat correctament
        if img is None:
            # print 'Failed to load image file:', fn
            print 'Failed to load image file:'
            continue
        else:
            # Si s'ha fet un crop, detectem la persona dintre la zona
            if len(crop) > 0:
                found, w = hog.detectMultiScale(crop)
                print 'crop'

                # Si no s'ha detectat cap persona busquem a tota la imatge

                if len(found) < 1:
                    crop = []
                    found, w = hog.detectMultiScale(img)
                    print 'crop_found'

            # Si no existeix crop busquem a tota la imatge
            else:
                found, w = hog.detectMultiScale(img)
                print 'general'

            found_filtered = []

            # Si s'han trobat deteccions
            if len(found) > 0:
                # Revisem que les deteccions trobades no hi hagi deteccions dintre de deteccions
                found_filtered = filter_founds(found)

                # Passem la deteccio feta en el crop a la imatge real
                if len(crop):
                    crop = []
                    found_filtered[0][0] += initial_x
                    found_filtered[0][1] += initial_y

                # Si trobem + d'una deteccio agafem la + semblant a l'anterior si existeix una anterior
                if len(found_filtered) > 1 and width_pred != 0 and height_pred != 0:
                    found_filtered = check_found_size(found_filtered)

                # Agafem l'amplada i l'altura de la deteccio
                width = found_filtered[0][2]
                height = found_filtered[0][3]

                # A la primera iteracio actualitzem els valors de predecessor
                if height_pred == 0 and width_pred == 0:
                    height_pred = height
                    width_pred = width

                # Dibuixem les deteccions
                draw_detections(img, found)
                draw_detections(img, found_filtered, 2)

                # Utilitzem calmant per fer les prediccions
                predict_next_location(index)

                # Actualitzem els valors d'amplada i altura de la imatge per comparar amb la seguent
                height_pred = height
                width_pred = width

                # Mostrem imatge amb les deteccions
                show_image('image',img)
                #cv2.imwrite(str(iteracion) + '_global.png', img)

            else:
                crop = []
                print 'No trobat'
    except:
        print 'loading error'
        continue
    # Actualitzem la iteracio (num. imatge)
    iteracion += 1
    e2 = cv2.getTickCount()
    t = (e2 - e1)/cv2.getTickFrequency()
    text = str(iteracion) + '  ' + str(t)
    print text
# Mostrem grafics de deteccions reals i predictives
draw_predictions(pred,meas)
# show_image('results',frame)
# cv2.imwrite('graphic_result.png', frame)

