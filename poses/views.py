from django.shortcuts import render
import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import time
# Create your views here.


def index(request):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # webcam / video 불러오기
    cap = cv2.VideoCapture(0)
    start_time = time.time()  # 현재 시간을 기록

    df = pd.DataFrame()  # 초기 빈 DataFrame 생성

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and time.time() - start_time <= 4:  # 4초가 지나면 종료
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 이미지 좌우 반전
            image = cv2.flip(image, 1)

            # 이미지 처리
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 이미지에 포즈 그리기
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # 각 랜드마크의 인덱스 번호 그리기 및 데이터 수집
                detected_landmarks = len(results.pose_landmarks.landmark)
                row = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if idx >= detected_landmarks:  # 감지된 랜드마크 수를 초과하는 경우 루프 중단
                        break

                    # 랜드마크의 화면 좌표 변환 및 데이터 수집
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    row.extend([landmark.x, landmark.y, landmark.z])

                # 첫 번째 프레임에서 dataframe 컬럼 생성
                if df.empty:
                    columns = []
                    for i in range(detected_landmarks):
                        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
                    df = pd.DataFrame(columns=columns)

                # 데이터프레임에 랜드마크 데이터 추가
                tmp = pd.DataFrame([row], columns=df.columns)
                df = pd.concat([df, tmp])

            # 이미지 표시
            cv2.imshow('MediaPipe Pose', image)  # 이미지 좌우 반전 후 표시
            if cv2.waitKey(5) & 0xFF == 27:
                break

        # 마지막 프레임 저장을 위한 이미지 복사
        last_image = image.copy()

    cap.release()
    cv2.destroyAllWindows()

    # 결과를 'hello.csv' 파일로 저장
    df.to_csv(r'C:\Users\chiwoopark\Coli\hello.csv', index=False)

    # 마지막 프레임을 이미지 파일로 저장
    cv2.imwrite(r'C:\Users\chiwoopark\Coli\last_image.jpg', last_image)
    
    return render(request,'poses/hello.html')
