import cv2
import time
import numpy as np
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from Modules.Hand_Tracking_Project.hand_tracking_module import handDetector

###################################################
wcam, hcam = 640, 480
###################################################


cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ptime = 0
detector = handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

vol = 0
volBar = 400
volPer = 0
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPositions(img,draw=False)
    # print(lmlist)
    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1],lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2,(y1+y2)//2
        cv2.circle(img,(x1,y1),5,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),5,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)
        length = math.hypot(x2-x1,y2-y1)
        # hand range 20,140
        # volume range -65,0
        vol = np.interp(length,[20,140],[min_vol,max_vol])
        volBar = np.interp(length, [20, 140], [400, 150])
        volPer = np.interp(length, [20, 140], [0, 100])
        # print(length)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img, (cx, cy), 5, (0,255,0), cv2.FILLED)

    cv2.rectangle(img,(50,140),(85,400),(0,255,0),3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(volPer)), (10, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    cv2.imshow("image",img)

    if cv2.waitKey(1) == ord("q"):
        cap.release()
        cv2.destroyAllWindows()