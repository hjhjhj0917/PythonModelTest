import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh 설정 (Iris 모델 포함)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Iris 랜드마크 활성화를 위해 필수
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 거리 계산을 위한 상수 (평균 인간 홍채 직경: 11.7mm)
IRIS_DIAMETER_MM = 11.7

# 웹캠 시작
cap = cv2.VideoCapture(0)

print("거리 측정 테스트를 시작합니다. 'q'를 누르면 종료합니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 이미지 반전 및 RGB 변환
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    img_h, img_w, _ = image.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 1. 왼쪽 홍채 랜드마크 추출 (468번 이후가 Iris 포인트)
            # MediaPipe Iris 포인트: 왼쪽(468-472), 오른쪽(473-477)
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                    for p in face_landmarks.landmark])

            # 왼쪽 홍채의 중심과 주변 포인트 계산 (거리 추정용)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[468:473])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)

            # 2. 픽셀 거리 기반 실제 거리(Metric Distance) 계산
            # 초점 거리(focal length) 추정치 필요 (일반적인 웹캠 기준 약 700~900)
            # 여기서는 표준적인 추정 공식 d = (실제직경 * 초점거리) / 픽셀직경 사용
            # 간단한 테스트를 위해 초점 거리를 약 800으로 가정 (환경에 따라 보정 필요)
            focal_length = 550
            pixel_iris_diameter = l_radius * 2

            if pixel_iris_diameter > 0:
                # 거리(mm) = (실제 홍채 직경 * 초점 거리) / 화면상 홍채 픽셀 직경
                distance_mm = (IRIS_DIAMETER_MM * focal_length) / pixel_iris_diameter
                distance_cm = distance_mm / 10

                # 3. 결과 시각화
                # 홍채 외곽 그리기
                cv2.circle(image, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)

                # 거리 정보 표시
                cv2.putText(image, f"Distance: {distance_cm:.1f} cm", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 6dB 법칙 적용을 위한 변수 확인용
                cv2.putText(image, f"Iris Pixel: {pixel_iris_diameter:.2f}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('MediaPipe Iris Distance Test', image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()