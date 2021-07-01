# Author : Shubham Kumar

# Face Recognition Basics

import cv2
import  mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon = 0.5):

        self.minDetectionCon = minDetectionCon
        self.npFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceDetection = self.npFaceDetection.FaceDetection(self.minDetectionCon)

    def findfaces(self, frame, draw = True):
        imgRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        self.results = self.FaceDetection.process(imgRGB)
        bboxes =[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bboxC = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id ,bboxC, detection.score])
                if draw:
                    frame = self.fancyDraw(frame, bboxC)
                    cv2.putText(frame, f"{int(detection.score[0]*100)}%", (bboxC[0], bboxC[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 204, 102), 1)
        return frame, bboxes

    def fancyDraw(self, frame, bboxC, l=30, t=3):
        x, y, w, h = bboxC
        x1, y1 = x+w, y+h
        cv2.rectangle(frame, bboxC, (0, 0, 204), 1)
        # Top left x,y
        cv2.line(frame, (x, y), (x + l, y), (0, 204, 204), t)
        cv2.line(frame, (x, y), (x, y+l), (0, 204, 204), t)
        # Top right x1,y
        cv2.line(frame, (x1, y), (x1 - l, y), (0, 204, 204), t)
        cv2.line(frame, (x1, y), (x1, y + l), (0, 204, 204), t)
        # bottom left x,y1
        cv2.line(frame, (x, y1), (x + l, y1), (0, 204, 204), t)
        cv2.line(frame, (x, y1), (x, y1 - l), (0, 204, 204), t)
        # bottom right x1,y1
        cv2.line(frame, (x1, y1), (x1 - l, y1), (0, 204, 204), t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), (0, 204, 204), t)

        return frame


def main():
    cap = cv2.VideoCapture(0)
    ptime = 0
    detector = FaceDetector()
    while True:
        success, frame = cap.read()
        frame, bboxes = detector.findfaces(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(frame, f"FPS:{int(fps)}", (4, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.imshow("FRAME", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ =="__main__":
    main()