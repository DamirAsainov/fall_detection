from ultralytics import YOLO
import cv2


class FallDetector:
    def __init__(self, model_path="yolo11n-pose.pt"):
        self.model = YOLO(model_path)

    def detect_fall(self, keypoints, bbox):
        try:
            required_points = [6, 12, 16]  # Левое плечо, бедро и стопа
            for point in required_points:
                if keypoints[point][2] < 0.5:  # Коэффициент уверенности меньше 50%
                    print("Недостаточная уверенность для ключевых точек")
                    return False

            left_shoulder_y = keypoints[6][1]
            left_hip_y = keypoints[12][1]
            left_foot_y = keypoints[16][1]

            xmin, ymin, xmax, ymax = bbox
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin
            if bbox_width > bbox_height * 1.5:
                print("Wide bbox, fall detected")
                return True

            len_factor = left_shoulder_y - left_hip_y

            if (
                    left_shoulder_y > left_foot_y - len_factor
                    and left_hip_y > left_foot_y - (len_factor / 2)
                    and left_shoulder_y > left_hip_y - (len_factor / 2)
            ):
                print("Fall Detected")
                return True

            return False
        except IndexError as e:
            print("Error with keypoint:", e)
            return False

    def process_frame(self, frame):
        results = self.model(frame)
        frame = results[0].plot()

        for result in results:
            if result.keypoints is not None:
                keypoints = results[0].keypoints.data[0].cpu().numpy()

                bbox = None
                if results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
                    bbox = results[0].boxes.xyxy[0].cpu().numpy()

                if bbox is not None and self.detect_fall(keypoints, bbox):
                    xmin, ymin, xmax, ymax = bbox
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 0, 255), thickness=3)
                    cv2.putText(frame, "FALL DETECTED!", (int(xmin), int(ymin) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame
