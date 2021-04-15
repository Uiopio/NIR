import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set(cv.CAP_PROP_BRIGHTNESS, 0);
#cap.set(cv.CAP_PROP_CONTRAST, 0);
#cap.set(cv.CAP_PROP_SATURATION, 0);

fourcc = cv.VideoWriter_fourcc(*'MJPG')
cap.set(cv.CAP_PROP_FOURCC, fourcc)


mat = cv.imread("./1.jpg")

k = 0
k2 = 0

def on_trackbar(val):
    global k
    k = val
    print (k)

cv.namedWindow("roi")
cv.createTrackbar('zoom', "roi", 1, 500, on_trackbar)

on_trackbar(0)


while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2] #720 1280

    roi = frame[k:(height - k), (k*2) :(width - (k*2))]
    roi = cv.resize(roi, (1280, 720))

    if cv.waitKey(1) == ord('q'):
        break

    #cv.imshow('frame', frame)
    cv.imshow('roi', roi)



# When everything done, release the capture
cap.release()
cv.destroyAllWindows()