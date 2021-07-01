# Author : Shubham Kumar

# Face Recognition Basics



import cv2
import  mediapipe as mp
import time

cap = cv2.VideoCapture(0)
ptime = 0

npFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = npFaceDetection.FaceDetection(0.75)

while True:
    success, frame = cap.read()

    imgRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    results =  FaceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bboxC = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, bboxC, (255, 0, 255), 2)
            cv2.putText(frame, f"{int(detection.score[0]*100)}%", (bboxC[0], bboxC[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255 , 51, 255), 1)


    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame, f"FPS:{int(fps)}", (4, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)



    cv2.imshow("FRAME", frame)

    if cv2.waitKey(1) == 27:
        break
