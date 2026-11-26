
import cv2
import mediapipe as mp
from pathlib import Path

# -------------------------------
# 1. Initialize MediaPipe Tasks
# -------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = Path(__file__).with_name("hand_landmarker.task")

if not model_path.exists():
    raise FileNotFoundError(
        "hand_landmarker.task not found. "
        "Place it beside finger_detection.py."
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(model_path)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------------------
# 2. Finger landmark IDs
# -------------------------------

TIP_IDS = [4, 8, 12, 16, 20]

FINGER_NAMES = [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Little"
]

# Hand connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# -------------------------------
# 3. Finger detection function
# -------------------------------

def detect_fingers(landmarks, hand_label):

    fingers = []

    # Thumb detection
    if hand_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    else:
        if landmarks[4].x > landmarks[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Index, Middle, Ring and Little
    for tip_id in TIP_IDS[1:]:

        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# -------------------------------
# 4. Start webcam
# -------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not detected!")

print("Finger Detection Started!")

timestamp_ms = 0

# -------------------------------
# 5. Run hand detection
# -------------------------------

try:

    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to read camera frame.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

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

            # -------------------------------
            # 6. Process detected hands
            # -------------------------------

            if results.hand_landmarks:

                for hand_index, landmarks in enumerate(
                    results.hand_landmarks
                ):

                    hand_label = (
                        results.handedness[hand_index][0].category_name
                    )

                    # Draw landmarks
                    h, w, _ = frame.shape

                    for landmark_id, landmark in enumerate(landmarks):

                        x = int(landmark.x * w)
                        y = int(landmark.y * h)

                        cv2.circle(
                            frame, (x, y), 5,
                            (0, 255, 0), -1
                        )

                    # Draw connections
                    for start, end in HAND_CONNECTIONS:

                        x1 = int(landmarks[start].x * w)
                        y1 = int(landmarks[start].y * h)

                        x2 = int(landmarks[end].x * w)
                        y2 = int(landmarks[end].y * h)

                        cv2.line(
                            frame, (x1, y1), (x2, y2),
                            (255, 0, 255), 2
                        )

                    # Detect fingers
                    finger_status = detect_fingers(
                        landmarks, hand_label
                    )

                    # Hand label
                    wrist_x = int(landmarks[0].x * w)
                    wrist_y = int(landmarks[0].y * h)

                    cv2.putText(
                        frame,
                        f"{hand_label} Hand",
                        (wrist_x, wrist_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                    # Display finger status
                    y_position = 90 + (hand_index * 220)

                    for i, status in enumerate(finger_status):

                        if status == 1:
                            state = "OPEN"
                            color = (0, 255, 0)
                        else:
                            state = "CLOSED"
                            color = (0, 0, 255)

                        cv2.putText(
                            frame,
                            f"{hand_label} {FINGER_NAMES[i]}: {state}",
                            (20, y_position + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                    total_fingers = sum(finger_status)

                    cv2.putText(
                        frame,
                        f"Total: {total_fingers}",
                        (20, y_position + 185),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

            # Show camera
            cv2.imshow("AI Finger Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Finger Detection Closed!")
