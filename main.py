from ultralytics import YOLO
#import numpy as np
import cv2
import math

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(r"C:\Users\Alser\Downloads\Watch Your Step! Funny Slips and Falls Compilation _ FailArmy.mp4")


def detect_fall(keypoints, bbox):
    try:
        left_shoulder_y = keypoints[6][1]
        left_hip_y = keypoints[12][1]
        left_foot_y = keypoints[16][1]
        # left_shoulder_z = keypoints[6][2]
        # left_hip_z = keypoints[12][2]
        # left_foot_z = keypoints[16][2]

        xmin, ymin, xmax, ymax = bbox
        if xmax - xmin >= (ymax - ymin) * 1.15:
            return True

        len_factor = left_shoulder_y - left_hip_y
        print("Len_factor", len_factor)
        print(left_shoulder_y > left_foot_y - len_factor,left_hip_y > left_foot_y - (len_factor / 2), left_shoulder_y > left_hip_y - (len_factor / 2) )
        print("Left Sholder", left_shoulder_y, "Left foot", left_foot_y, "left body", left_hip_y)

        if (
                left_shoulder_y > left_foot_y - len_factor
                and left_hip_y > left_foot_y - (len_factor / 2)
                and left_shoulder_y > left_hip_y - (len_factor / 2)
        ):
            return True

        # print(left_shoulder_z, left_hip_z, left_foot_z)
        return False
    except IndexError as e:
        print("Problem with keypoints:", e)
        return False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = results[0].plot()

    for result in results:
        if result.keypoints is not None:
            keypoints = results[0].keypoints.data[0].cpu().numpy()

            xmin = 0
            ymin = 0
            xmax = 0
            ymax = 0
            bbox = None
            if results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
                bbox = results[0].boxes.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = bbox

            if detect_fall(keypoints, bbox):
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=3)
                cv2.putText(frame, "FALL DETECTED!", (int(xmin), int(ymin) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # if results[0].keypoints is not None:
    #     keypoints = results[0].keypoints.data[0].cpu().numpy()
    #
    #     xmin = 0
    #     ymin = 0
    #     xmax = 0
    #     ymax = 0
    #     bbox = None
    #     if results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
    #         bbox = results[0].boxes.xyxy[0].cpu().numpy()
    #         xmin, ymin, xmax, ymax = bbox
    #
    #     if detect_fall(keypoints, bbox):
    #         cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=3)
    #         cv2.putText(frame, "FALL DETECTED!", (int(xmin), int(ymin) - 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
