import cv2
import time
import numpy as np
from Modules.Pose_estimation_project import Pose_module
detector = Pose_module.poseDetector()
cap = cv2.VideoCapture(0)
ptime = 0
count = 0
dir = 0

def checkper(per,count,dir):
    if per == 100:
        if dir == 0:
            count += 0.5
            dir = 1
    if per == 0:
        if dir == 1:
            count += 0.5
            dir = 0
    return count

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lm_list = detector.findPosition(img,draw=False)

    if len(lm_list)!=0:
        #right arm
        righndang = detector.findAngle(img,12,14,16)
        # left arm
        # lefhndang = detector.findAngle(img, 11, 13, 15)

        min=35
        max=175

        # lefper = np.interp(lefhndang,(min,max),(0,100))
        rghper = np.interp(righndang, (min, max), (0, 100))
        rgtbar = np.interp(righndang,(min,max),(145,400))

        #check for the dumble curls
        if rghper==100:
            if dir == 0:
                count+=0.5
                dir = 1
        if rghper==0:
            if dir == 1:
                count+=0.5
                dir = 0

        cv2.rectangle(img, (5, 140), (145, 400), (0, 255, 0), 2)
        cv2.rectangle(img, (5, int(rgtbar)), (145, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{round(rghper,2)} %', (5, 460), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 255), 4)

        cv2.rectangle(img,(5,60),(145,135),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{count}',(10,120),cv2.FONT_HERSHEY_PLAIN,5,(255,0,0),4)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)

    cv2.imshow("image",img)
    if cv2.waitKey(1)==ord('q'):
        cap.release()
        cv2.destroyAllWindows()