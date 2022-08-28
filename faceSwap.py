import imghdr
import cv2
import dlib
from imutils import face_utils
import numpy as np

mainFacePoints = []
camFacePoints = []
    

mainFace = cv2.imread("girl.jpg")
grayMainFace = cv2.cvtColor(mainFace,cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


drawing = np.zeros((mainFace.shape[0], mainFace.shape[1], 3), dtype=np.uint32)

rectsMainFace = detector(grayMainFace, 1)



def returnFaceMask(img,points):
    cv2.fillPoly(img=img, pts=[np.array(points)], color=(0, 255, 0))
    #masked_image = cv2.bitwise_and(img, drawing)
    return img

for (i, rect) in enumerate(rectsMainFace):
        
        shape = predictor(grayMainFace, rect)
        shape = face_utils.shape_to_np(shape)

        
        for (x, y) in shape:
            mainFacePoints.append([x, y])
            

print([np.array(mainFacePoints)])

cap = cv2.VideoCapture(0)

while True:
    camFacePoints = []
    ret,frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rectsCam = detector(gray, 1)
    rectsMainFace = detector(grayMainFace, 1)
    
    for (i, rect) in enumerate(rectsCam):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        
        for (x, y) in shape:
            camFacePoints.append([x, y])
            
    if camFacePoints != []:
        masked=returnFaceMask(frame,points=[camFacePoints[19],camFacePoints[26],camFacePoints[26],camFacePoints[17],camFacePoints[26],camFacePoints[13],camFacePoints[9],camFacePoints[5],camFacePoints[1]])
        #cv2.imshow("masked",masked)

    cv2.imshow("Cam Frame",frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
