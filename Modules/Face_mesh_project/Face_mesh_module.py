import cv2
import mediapipe as mp
import time

class faceMesh():
    def __init__(self,mode=False,
                 max_faces=1,
                 refine_landmarks=False,
                 min_det_con=0.5,
                 min_trac_con=0.5,
                 thickness=1,
                 circle_radius=2):
        self.mode = mode
        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        self.min_det_con = min_det_con
        self.min_track_con = min_trac_con
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mesh=self.mpFaceMesh.FaceMesh(static_image_mode=self.mode,
                                           max_num_faces=self.max_faces,
                                           refine_landmarks=self.refine_landmarks,
                                           min_detection_confidence=self.min_det_con,
                                           min_tracking_confidence=self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.DrawingStyles = self.mpDraw.DrawingSpec(thickness=self.thickness,
                                                     circle_radius=self.circle_radius)

    def findFaceMesh(self,img,draw = True):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(imgRgb)
        faces = []
        if draw:
            if results.multi_face_landmarks:
                for landmark in results.multi_face_landmarks:
                    # self.mpDraw.draw_landmarks(image=img,
                    #                            landmark_list=landmark,
                    #                            connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                    #                            landmark_drawing_spec=self.DrawingStyles,
                    #                            connection_drawing_spec=self.DrawingStyles)
                    face=[]
                    for iD,lm in enumerate(landmark.landmark):
                        # print(lm)
                        ih, iw, ic = img.shape
                        x,y=int(lm.x*iw),int(lm.y*ih)
                        cv2.putText(img, str(iD), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
                        face.append([x,y])
                    faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    while True:
        success, img = cap.read()
        detector = faceMesh(max_faces=1)
        img, faces = detector.findFaceMesh(img=img)
        ctime = time.time()

        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("image", img)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()