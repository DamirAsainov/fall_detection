from ultralytics import YOLO
import cv2


class FallDetector:
    def __init__(self, model_path="yolo11s-pose.pt"):
        self.model = YOLO(model_path)

    def detect_fall(self, keypoints, bbox):
        try:
            required_points = [5, 6, 11, 12, 15, 16]  # Плечи, бедра, стопы
            visible_points = [point for point in required_points if keypoints[point][2] >= 0.5]

            if len(visible_points) < 4:  # Если меньше 4 точек видно, пропускаем
                print("Недостаточно видимых ключевых точек")
                return False

            # Используем только видимые точки
            left_shoulder_y = keypoints[5][1] if 5 in visible_points else None
            right_shoulder_y = keypoints[6][1] if 6 in visible_points else None
            left_hip_y = keypoints[11][1] if 11 in visible_points else None
            right_hip_y = keypoints[12][1] if 12 in visible_points else None
            left_foot_y = keypoints[15][1] if 15 in visible_points else None
            right_foot_y = keypoints[16][1] if 16 in visible_points else None

            # Средние значения для устойчивости
            shoulder_y = (left_shoulder_y + right_shoulder_y) / 2 if left_shoulder_y and right_shoulder_y else None
            hip_y = (left_hip_y + right_hip_y) / 2 if left_hip_y and right_hip_y else None
            foot_y = (left_foot_y + right_foot_y) / 2 if left_foot_y and right_foot_y else None

            xmin, ymin, xmax, ymax = bbox
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin


            if bbox_width > bbox_height * 1.5:
                print("Wide bbox, fall detected")
                return True

            # Проверка на положение тела
            if shoulder_y and hip_y and foot_y:
                len_factor = shoulder_y - hip_y

                print(len_factor)
                if len_factor >= 0:
                    return True

                if (
                        shoulder_y > foot_y - len_factor
                        and hip_y > foot_y - (len_factor / 2)
                        and shoulder_y > hip_y - (len_factor / 2)
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
