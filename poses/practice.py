import mediapipe as mp
import cv2

# MediaPipe Pose 모듈 및 드로잉 유틸리티 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 이미지 파일 경로 설정 (여기서는 'your_image.jpg'를 사용자의 이미지 파일 경로로 변경하세요)
image_path = r'C:\Users\박치우\Desktop\image1.jpg'

# 이미지를 불러오고 포즈 감지를 위해 BGR에서 RGB로 변환
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pose 감지 실행
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        # 포즈 랜드마크 드로잉
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 결과 이미지 출력
        cv2.imshow("Pose Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No pose landmarks detected.")