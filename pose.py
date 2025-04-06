import cv2 as cv
import mediapipe as mp 
import time
import math

class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True,
                detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=1 if not self.up_body else 0,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        
    def find_pose(self, img, draw=True):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, 
                                            self.mp_pose.POSE_CONNECTIONS)
        return img
        
    def find_position(self, img, draw=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return self.lmlist
    
    def find_angle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2)
                            - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv.circle(img, (x1, y1), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv.circle(img, (x2, y2), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv.circle(img, (x3, y3), 10, (0, 0, 255), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    # cap = cv.VideoCapture('./6194051-uhd_3840_2160_25fps.mp4')
    cap = cv.VideoCapture('./2795746-uhd_2160_3840_25fps.mp4')
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        if not success:
            break
        img = detector.find_pose(img)
        lmList = detector.find_position(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (70, 50), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        img = cv.resize(img, (800,650))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()