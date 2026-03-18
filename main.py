import cv2
import mediapipe as mp
from analyzer import ParkinsonRiskAnalyzer

mp_face_mesh = mp.solutions.face_mesh

def main():
    cap = cv2.VideoCapture(0)
    analyzer = ParkinsonRiskAnalyzer()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("카메라를 읽을 수 없습니다.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            display_result = None

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                display_result = analyzer.update(landmarks, w, h)

                cv2.putText(frame, f"Blink/min: {display_result['blink_per_min']}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Mean EAR: {display_result['mean_ear']}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Mouth STD: {display_result['mouth_std']}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Tilt STD: {display_result['tilt_std']}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Motion Mean: {display_result['motion_mean']}", (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"PD Risk: {display_result['risk_score']} ({display_result['risk_label']})", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("PD Webcam Screening Prototype", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and display_result is not None:
                analyzer.save_result(display_result)
                print("결과 저장 완료: results/session_logs/last_result.json")

            if key == ord('q'):
                if display_result is not None:
                    analyzer.save_result(display_result)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()