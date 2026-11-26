
import cv2
import mediapipe as mp

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure hand detection
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not detected!")
    exit()

timestamp_ms = 0

try:
    with HandLandmarker.create_from_options(options) as detector:

        while True:
            success, frame = cap.read()

            if not success:
                print("Unable to read camera frame")
                break

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms += 33

            results = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            detected_hands = 0

            if results.hand_landmarks:

                for index, hand_landmarks in enumerate(
                    results.hand_landmarks
                ):
                    detected_hands += 1

                    # Get left/right hand label
                    hand_label = (
                        results.handedness[index][0].category_name
                    )

                    # Draw all 21 landmarks
                    for landmark_id, landmark in enumerate(
                        hand_landmarks
                    ):
                        h, w, _ = frame.shape

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        cv2.circle(
                            frame, (x, y), 5,
                            (0, 255, 0), -1
                        )

                        cv2.putText(
                            frame,
                            str(landmark_id),
                            (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 0, 0),
                            1
                        )

                    # Display hand label near wrist
                    wrist = hand_landmarks[0]
                    x = int(wrist.x * frame.shape[1])
                    y = int(wrist.y * frame.shape[0])

                    cv2.putText(
                        frame,
                        hand_label + " Hand",
                        (x, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

            cv2.putText(
                frame,
                f"Hands Detected: {detected_hands}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # Mirror the preview after detection
            frame = cv2.flip(frame, 1)

            cv2.imshow("AI Virtual Mouse - Hand Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
