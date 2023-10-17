import cv2
import mediapipe as mp
import time

class faceDetection():
    def __init__(self,detectconf=0.5,modelsel=None):
        self.detectconf = detectconf
        self.modelsel = modelsel
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.detectconf,self.modelsel)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self,img, draw =True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRgb)
        # print(results.detections)
        bboxes = []
        if self.results.detections:
            for iD, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                bboxC=detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                    int(bboxC.width * iw), int(bboxC.height*ih)
                bboxes.append([iD,bbox,detection.score[0]])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv2.putText(img, str(round(detection.score[0] * 100, 2)), (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        return img, bboxes

    def fancyDraw(self,img,bbox,l=30,thickness=5,rt=1):
        x,y,w,h = bbox
        x1,y1 =x+w,y+h
        cv2.rectangle(img, bbox,(255,0,255),rt)
        # top left
        cv2.line(img, (x,y),(x+l,y),(255,0,255),thickness)
        cv2.line(img, (x,y),(x,y+l),(255,0,255),thickness)

        # top right
        cv2.line(img, (x1,y),(x1-l,y),(255,0,255),thickness)
        cv2.line(img, (x1,y),(x1,y+l),(255,0,255),thickness)

        # bottom left
        cv2.line(img, (x,y1),(x+l,y1),(255,0,255),thickness)
        cv2.line(img, (x,y1),(x,y1-l),(255,0,255),thickness)

        # bottom right
        cv2.line(img, (x1,y1),(x1-l,y1),(255,0,255),thickness)
        cv2.line(img, (x1,y1),(x1,y1-l),(255,0,255),thickness)

        return img

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = faceDetection()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        if cv2.waitKey(1) == ord("q"):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()