import cv2
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh
mesh=mpFaceMesh.FaceMesh(max_num_faces=1)
mpDraw = mp.solutions.drawing_utils
DrawingStyles = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mesh.process(imgRgb)
    # print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for landmark in results.multi_face_landmarks:
            # print(landmark)
            mpDraw.draw_landmarks(image=img,
                                  landmark_list=landmark,
                                  connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=DrawingStyles,
                                  connection_drawing_spec=DrawingStyles)

            for iD,lm in enumerate(landmark.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x,y=int(lm.x*iw),int(lm.y*ih)
                print(iD,x,y)
    ctime = time.time()

    fps =1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image", img)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

