import cv2
import time
import os
import numpy as np
from Modules.Hand_Tracking_Project.hand_tracking_module import handDetector

brushThick = 15
folderPath = "./Images/Header"
mylist = os.listdir(folderPath)
# print(mylist)
overlay = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlay.append(image)
# print(len(overlay))
# print(overlay)
header = overlay[0]
drawColor = (255,0,255)
eraseThick = 50
# print(header)
cap = cv2.VideoCapture(0)
ptime = 0
detector = handDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.zeros((480,640,3),np.uint8)
while True:
    # import image
    success,img =cap.read()
    img = cv2.flip(img,1)
    # detecting hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findPositions(img,draw=False)

    # tip of index and middle finger
    x1,y1 = None,None
    x2, y2 = None, None

    if len(lmlist)!=0:
        x1,y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

    # checking which fingers are up
    lstFingers = detector.fingersUp()
    # print(lstFingers)

    # if selection mode i.e. two fingers are up
    if lstFingers[1] and lstFingers[2]:
        xp, yp = 0, 0
        print("Selection Mode")
        print(x1)
        if y1 < 85:
            if 140 < x1 < 170:
                header = overlay[0]
                drawColor = (255,0,255)
            elif 285 < x1 < 295:
                header = overlay[1]
                drawColor = (255,0,0)
            elif 385 < x1 < 415:
                header = overlay[2]
                drawColor = (140,140,140)
            elif 505 < x1 < 525:
                header = overlay[3]
                drawColor = (0,0,0)
        cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)

    # if drawing mode i.e. index finger is up
    if lstFingers[1] and lstFingers[2]==False:
        cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
        print("Drawing Mode")
        if xp == 0 and yp == 0:
            xp,yp = x1,y1
        if drawColor == (0,0,0):
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraseThick)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraseThick)
        else:
            cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThick)
            cv2.line(imgCanvas, (xp,yp),(x1,y1),drawColor,brushThick)

        xp,yp = x1,y1
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # setting the header image
    img[0:85,0:640] = header
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)), (20,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    cv2.imshow("image",img)
    cv2.imshow("ImgCanvas",imgCanvas)
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        cv2.destroyAllWindows()
