import cv2 as cv
import time
import pose_estimation_module as pem
import numpy as np

cap = cv.VideoCapture(0)
pTime = 0
detector = pem.PoseDetector()
count = 0
dir = 0

while True:
    suc, vid = cap.read()
    vid = detector.makePose(vid, False)

    lmList = detector.makePostions(vid)
    if len(lmList) != 0:
        angle = detector.findAngle(vid, 12, 14, 16)
        per = np.interp(angle, (55, 165), (0, 100))

        if per == 100:
            if dir ==0:
                count +=0.5
                dir = 1
        if per == 0:
            if dir==1:
                count+=0.5
                dir = 0

        cv.putText(vid, f"Count={int(count)}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(vid, f"FPS:{int(fps)}", (420, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    cv.imshow("VIDEO", vid)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break