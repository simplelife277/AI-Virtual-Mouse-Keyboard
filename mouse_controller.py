
import cv2
import mediapipe as mp
import math
import time
import tkinter as tk

from pynput.mouse import Controller, Button
from pathlib import Path


# ==========================================
# 1. MOUSE SETTINGS
# ==========================================

mouse = Controller()

PINCH_THRESHOLD = 0.04
CLICK_COOLDOWN = 0.5

# Cursor smoothing
SMOOTHING = 0.25

# Scrolling sensitivity
SCROLL_SENSITIVITY = 20

last_click_time = 0
was_pinching = False

smooth_x = None
smooth_y = None

previous_scroll_y = None
scroll_accumulator = 0


# ==========================================
# 2. SCREEN DIMENSIONS
# ==========================================

root = tk.Tk()
root.withdraw()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.destroy()


# ==========================================
# 3. MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = Path(__file__).with_name(
    "hand_landmarker.task"
)

if not model_path.exists():
    raise FileNotFoundError(
        "hand_landmarker.task not found!"
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


# ==========================================
# 4. FINGER DETECTION
# ==========================================

def is_finger_open(landmarks, tip_id):

    return (
        landmarks[tip_id].y
        < landmarks[tip_id - 2].y
    )


# ==========================================
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not detected!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

timestamp_ms = 0

print("AI Virtual Mouse Started!")
print("--------------------------------")
print("Move index finger  : Move cursor")
print("Thumb + index pinch: Left click")
print("Index + middle up  : Scroll mode")
print("Move fingers up    : Scroll up")
print("Move fingers down  : Scroll down")
print("Press Q            : Exit")
print("--------------------------------")


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

try:

    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = cap.read()

            if not success:
                print("Camera frame unavailable!")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = max(
                timestamp_ms + 1,
                int(time.monotonic() * 1000)
            )

            results = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            hand_found = False

            if results.hand_landmarks:

                for hand_index, landmarks in enumerate(
                    results.hand_landmarks
                ):

                    hand_label = (
                        results.handedness[hand_index][0].category_name
                    )

                    # Use RIGHT hand only
                    if hand_label != "Right":
                        continue

                    hand_found = True

                    # ==================================
                    # 7. GET LANDMARKS
                    # ==================================

                    thumb = landmarks[4]
                    index = landmarks[8]
                    middle = landmarks[12]

                    index_x = int(index.x * w)
                    index_y = int(index.y * h)

                    # Draw index fingertip
                    cv2.circle(
                        frame,
                        (index_x, index_y),
                        10,
                        (0, 255, 0),
                        -1
                    )

                    # ==================================
                    # 8. PINCH DETECTION
                    # ==================================

                    distance = math.hypot(
                        index.x - thumb.x,
                        index.y - thumb.y
                    )

                    is_pinching = (
                        distance < PINCH_THRESHOLD
                    )

                    # ==================================
                    # 9. SCROLL GESTURE
                    # ==================================

                    index_open = is_finger_open(
                        landmarks, 8
                    )

                    middle_open = is_finger_open(
                        landmarks, 12
                    )

                    scroll_mode = (
                        index_open and middle_open
                        and not is_pinching
                    )

                    # ==================================
                    # 10. SCROLL CONTROL
                    # ==================================

                    if scroll_mode:

                        current_y = (
                            index.y + middle.y
                        ) / 2

                        if previous_scroll_y is not None:

                            movement = (
                                previous_scroll_y - current_y
                            ) * h

                            scroll_accumulator += movement

                            if abs(scroll_accumulator) >= SCROLL_SENSITIVITY:

                                scroll_steps = int(
                                    scroll_accumulator
                                    / SCROLL_SENSITIVITY
                                )

                                # Positive = up
                                mouse.scroll(0, scroll_steps)

                                scroll_accumulator -= (
                                    scroll_steps
                                    * SCROLL_SENSITIVITY
                                )

                        previous_scroll_y = current_y

                        # Disable clicking during scrolling
                        was_pinching = False

                        status = "SCROLL MODE"
                        color = (255, 255, 0)

                        cv2.putText(
                            frame,
                            "INDEX + MIDDLE: SCROLL",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                    else:

                        previous_scroll_y = None
                        scroll_accumulator = 0

                        # ==================================
                        # 11. SMOOTH CURSOR MOVEMENT
                        # ==================================

                        if smooth_x is None:

                            smooth_x = (
                                index.x * screen_width
                            )

                            smooth_y = (
                                index.y * screen_height
                            )

                        else:

                            target_x = (
                                index.x * screen_width
                            )

                            target_y = (
                                index.y * screen_height
                            )

                            smooth_x = (
                                SMOOTHING * target_x
                                + (1 - SMOOTHING) * smooth_x
                            )

                            smooth_y = (
                                SMOOTHING * target_y
                                + (1 - SMOOTHING) * smooth_y
                            )

                        mouse_x = int(
                            max(
                                0,
                                min(
                                    screen_width - 1,
                                    smooth_x
                                )
                            )
                        )

                        mouse_y = int(
                            max(
                                0,
                                min(
                                    screen_height - 1,
                                    smooth_y
                                )
                            )
                        )

                        mouse.position = (
                            mouse_x,
                            mouse_y
                        )

                        # ==================================
                        # 12. LEFT CLICK
                        # ==================================

                        current_time = time.time()

                        if (
                            is_pinching
                            and not was_pinching
                            and current_time - last_click_time
                            > CLICK_COOLDOWN
                        ):

                            mouse.click(Button.left, 1)

                            last_click_time = current_time

                            print("Left Click!")

                        was_pinching = is_pinching

                        if is_pinching:

                            status = "CLICKING"
                            color = (0, 0, 255)

                        else:

                            status = "MOUSE MOVEMENT"
                            color = (0, 255, 0)

                        cv2.putText(
                            frame,
                            status,
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2
                        )

                    # ==================================
                    # 13. DISPLAY INFORMATION
                    # ==================================

                    cv2.putText(
                        frame,
                        "RIGHT HAND - VIRTUAL MOUSE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    # Draw thumb and index connection
                    thumb_x = int(thumb.x * w)
                    thumb_y = int(thumb.y * h)

                    cv2.line(
                        frame,
                        (thumb_x, thumb_y),
                        (index_x, index_y),
                        color,
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Pinch: {distance:.3f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    break

            # ==================================
            # 14. NO HAND DETECTED
            # ==================================

            if not hand_found:

                was_pinching = False
                previous_scroll_y = None
                scroll_accumulator = 0

                cv2.putText(
                    frame,
                    "SHOW RIGHT HAND",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            # ==================================
            # 15. SHOW CAMERA
            # ==================================

            cv2.imshow(
                "AI Virtual Mouse - Advanced",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# ==========================================
# 16. CLEANUP
# ==========================================

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("AI Virtual Mouse Closed!")
