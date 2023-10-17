import cv2
import mediapipe as mp
import time
from Modules.Hand_Tracking_Project.hand_tracking_module import handDetector
import os

###################################################
wcam, hcam = 640, 480
###################################################

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ptime = 0
myList = os.listdir("./Images/")

overlayList = []
for imgpath in myList:
    image = cv2.imread("./Images/"+imgpath)
    image = cv2.resize(image,(200,200),)
    overlayList.append(image)

# print(len(overlayList))
# print(overlayList[0].shape)

detector = handDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPositions(img,draw=False)
    # print(lmList)

    if len(lmList)!=0:
        fingers = []
        #thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        #for rest of four fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h,w,c=overlayList[totalFingers-1].shape
        img[0:h,0:w] = overlayList[totalFingers-1]
        cv2.rectangle(img,(28,225),(178,425),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,255),25)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2)
    cv2.imshow("image",img)
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        cv2.destroyAllWindows()


