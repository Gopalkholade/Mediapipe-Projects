import cv2
import mediapipe as mp
import time

mpFaceMesh = mp.solutions.face_mesh
mesh=mpFaceMesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils
mpDrawingStyles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success, img = cap.read()
    imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mesh.process(imgRgb)
    # print(results.multi_face_landmarks)
    if results.multi_face_landmarks:
        for iD,landmark in enumerate(results.multi_face_landmarks):
            # print(iD,landmark)
            mpDraw.draw_landmarks(image=img,
                                  landmark_list=landmark,
                                  connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=mpDrawingStyles.DrawingSpec((0,0,255),1,1),
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style())
    ctime = time.time()

    fps =1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

    cv2.imshow("image", img)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()

