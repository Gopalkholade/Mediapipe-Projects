import cv2
import mediapipe as mp
import time
import math

class poseDetector():
    def __init__(self, mode = False, upbody = False, smooth = True, detectconf = 0.5, trackconf = 0.5):
        self.mode = mode
        self.upbody = upbody
        self.smooth = smooth
        self.detectconf = detectconf
        self.trackconf = trackconf
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     # upper_body_only=self.upbody,
                                     smooth_landmarks= self.smooth,
                                     min_detection_confidence=self.detectconf,
                                     min_tracking_confidence=self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw =True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        self.results = self.pose.process(imgRgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self,img, draw =True):
        self.lmList =[]
        if self.results.pose_landmarks:
            for iD, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([iD,cx,cy])
                if draw:
                    if len(self.lmList) != 0:
                        cv2.circle(img, (cx,cy), 5,(255, 0, 255), cv2.FILLED)
        return  self.lmList
    def findAngle(self,img,p1,p2,p3,draw=True):
        #get Landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        #calculate the angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        # print(angle)
        if angle<0:
            angle=abs(angle)

        #draw
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 0, 255), 3)
            cv2.circle(img,(x1,y1),5,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), 2)
            cv2.circle(img,(x2,y2),5,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 0), 2)
            cv2.circle(img,(x3,y3),5,(255,0,0),cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (255, 0, 0), 2)
            cv2.putText(img,str(int(angle)),(x2-20,y2+50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
            return angle

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmlist = detector.findPosition(img, draw=False)
        print(lmlist)
        if len(lmlist) !=0:
            cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 15, (0, 0, 255), cv2.FILLED)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (78, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("image", img)

        if cv2.waitKey(1) == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
if __name__ == "__main__":
    main()