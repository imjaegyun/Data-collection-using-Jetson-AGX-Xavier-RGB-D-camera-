## install
install jetpack 5.1.1

install pytorch / torchvision // 0.12.0 / 0.13.0

trtexec build

install ultralytics

install onnx / onnxruntime / onnxruntime-gpu

git clone ultralytics yolov8

## export
yolo export model=yolov8n-pose.pt format=engine dynamic 
## execute
python3 jetson_pose_estimation.py
## key points

0 == "nose"

1 == "right_eye"

2 == "left_eye"

3 == "right_ear"

4 == "left_ear"

5 == "right_shoulder"

6 == "left_shoulder"

7 == "right_elbow"

8 == "left_elbow"

9 == "right_wrist"

10 == "left_wrist"

11 == "right_hip"

12 == "left_hip"

13 == "right_knee"

14 == "left_knee"

15 == "right_ankle"

16 == "left_ankle"
