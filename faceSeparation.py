import imghdr
import cv2
import dlib
from imutils import face_utils
import numpy as np


camFacePoints = []
    



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")






def returnConvexHull(points):
    return cv2.convexHull(points=points)


def returnFaceMask(img,camFacePoints):
    
    points = np.array(camFacePoints, np.int32)

    drawing = np.zeros((img.shape[0], img.shape[1],3),dtype=np.uint8)
    
    convexHull = returnConvexHull(points)

    cv2.fillPoly(img=drawing, pts=[np.array(convexHull)], color=(255, 255, 255))
    
    masked = cv2.bitwise_and(src1=np.array(drawing), src2=np.array(img))

    
    return masked


cap = cv2.VideoCapture(0)

while True:
    camFacePoints = []
    ret,frame = cap.read()
    frameCopy = frame.copy()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rectsCam = detector(gray, 1)

    maskedCam = np.zeros((frame.shape[0], frame.shape[1]))
    
    for (i, rect) in enumerate(rectsCam):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        
        for (x, y) in shape:
            cv2.circle(frame,(x,y),3,(0,0,255),-1)
            camFacePoints.append([x, y])
            
    if camFacePoints != []:
        masked=returnFaceMask(frameCopy,camFacePoints=camFacePoints)
    cv2.imshow("masked",masked)

    cv2.imshow("Cam Frame",frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
