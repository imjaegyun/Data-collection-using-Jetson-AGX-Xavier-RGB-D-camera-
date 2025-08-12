from ultralytics import YOLO
import cv2
import json
import os
import time

def generate_filename():
    """Generates a filename based on the current timestamp with year, month, day, hour, and minute."""
    return time.strftime("detections_%Y-%m-%d_%H%M.json", time.localtime())

def get_current_timestamp():
    """Returns the current time in 'year-month-day hour:minute:second' format using a 24-hour clock."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def append_data_to_json(file_path, new_data):
    """Appends new data to a JSON file. Creates the file if it does not exist."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(new_data, file, indent=4, sort_keys=True)
    else:
        with open(file_path, 'r+') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:  # If file is empty
                data = []
            data.extend(new_data)
            file.seek(0)
            json.dump(data, file, indent=4, sort_keys=True)

def run_yolo_webcam_detection():
    model = YOLO('yolov8n-pose.pt')
    cap = cv2.VideoCapture(0)
    last_file_creation_time = time.time()
    filename = generate_filename()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results_list = model(frame, show=True)
        data_to_save = []

        current_time = time.time()
        if current_time - last_file_creation_time >= 300:  # Every 5 minutes
            last_file_creation_time = current_time
            filename = generate_filename()

        for results in results_list:
            if len(results.boxes) > 0:
                boxes = results.boxes.xyxy
                confidences = results.boxes.conf
                class_ids = results.boxes.cls
                keypoints = results.keypoints.data if 'keypoints' in dir(results) else None

                for i, bbox in enumerate(boxes):
                    class_id = class_ids[i].item()
                    class_name = results.names[int(class_id)]

                    if class_name == 'person':
                        bbox_list = bbox.tolist()
                        confidence = confidences[i].item()
                        keypoints_list = keypoints[i].tolist() if keypoints is not None else None
                        detection_timestamp = get_current_timestamp()

                        data_to_save.append({
                            "timestamp": detection_timestamp,
                            "bbox": bbox_list,
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                            "keypoints": keypoints_list,
                        })

        if data_to_save:
            append_data_to_json(filename, data_to_save)

        cv2.imshow('YOLOv8 Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_webcam_detection()
