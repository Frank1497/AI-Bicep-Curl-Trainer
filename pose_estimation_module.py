import cv2 as cv
import mediapipe as mp
import time
import math

class PoseDetector:

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)

    def makePose(self, vid, draw=True):
        vidRGB = cv.cvtColor(vid, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(vidRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(vid, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return vid

    def makePostions(self, vid, draw=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                if draw:
                    h, w, c = vid.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(vid, (cx, cy), 3, (0, 0, 255), cv.FILLED)
                    self.lmlist.append([id, cx, cy])
        return self.lmlist


    def findAngle(self, vid, p1, p2, p3, draw=True):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        angle = math.degrees(math.atan2(y3-y2, x3-x2)-math.atan2(y1-y2, x1-x2))
        if angle<0:
            angle+=360

        if draw:
            cv.line(vid, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv.line(vid, (x2, y2), (x3, y3), (0, 255, 0), 3)
            cv.circle(vid, (x1, y1), 10, (255, 0, 0), cv.FILLED)
            cv.circle(vid, (x1, y1), 15, (255, 0, 0), 2)
            cv.circle(vid, (x2, y2), 10, (255, 0, 0), cv.FILLED)
            cv.circle(vid, (x2, y2), 15, (255, 0, 0), 2)
            cv.circle(vid, (x3, y3), 10, (255, 0, 0), cv.FILLED)
            cv.circle(vid, (x3, y3), 15, (255, 0, 0), 2)
            cv.putText(vid, str(int(angle)), (x2 - 30,y2 +50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        return angle

def main():
    pTime = 0
    video = cv.VideoCapture(0)  # 0  'football.mp4' ufc
    detector = PoseDetector()
    while True:
        success, vid = video.read()
        vid = detector.makePose(vid)

        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(vid, f"FPS={int(fps)}", (5, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv.imshow("VIDEO", vid)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()