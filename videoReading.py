import cv2

cap = cv2.VideoCapture(0)

# while(cap.isOpened()):
ret, frame = cap.read()
    # height, width, chanel = frame.shape
    # frame = cv2.resize(frame, (854,480))
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print frame.shape
cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.imwrite('resizela.png',frame)
cap.release()
cv2.destroyAllWindows()