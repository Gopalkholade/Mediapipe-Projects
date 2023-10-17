import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ptime = 0

while True:
    success, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRgb)
    # print(results.detections)
    if results.detections:
        for iD, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic = img.shape
            bbox = int(bboxC.xmin*iw), int(bboxC.ymin*ih), \
                int(bboxC.width * iw), int(bboxC.height*ih)
            cv2.rectangle(img, bbox,(255,0,255),2)
            cv2.putText(img, str(round(detection.score[0]*100,2)), (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("image",img)
    if cv2.waitKey(1) == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
