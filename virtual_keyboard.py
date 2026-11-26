
import cv2
import mediapipe as mp
import math
import time

from pathlib import Path
from pynput.keyboard import Controller, Key


# ==========================================
# 1. SETTINGS
# ==========================================

keyboard = Controller()

PINCH_THRESHOLD = 0.05
HOVER_DWELL_SECONDS = 0.9
KEY_COOLDOWN = 0.35

typed_text = ""
last_action_time = 0

# All five fingers on both hands
FINGERS = {
    "THUMB": 4,
    "INDEX": 8,
    "MIDDLE": 12,
    "RING": 16,
    "LITTLE": 20
}

# Hover and pinch states for each finger
hover_state = {}
pinch_state = {}

# Keyboard layout
KEYBOARD_KEYS = [
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    list("ZXCVBNM"),
    ["SPACE", "BACKSPACE", "ENTER"]
]

# Colors for each hand
HAND_COLORS = {
    "Left": (255, 150, 0),
    "Right": (0, 255, 255)
}


# ==========================================
# 2. MEDIAPIPE SETUP
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
# 3. CREATE KEYBOARD
# ==========================================

def create_keyboard(frame):

    h, w, _ = frame.shape

    key_height = 42
    gap = 6

    normal_width = 48

    special_widths = {
        "SPACE": 130,
        "BACKSPACE": 150,
        "ENTER": 100
    }

    key_positions = []

    total_height = (
        len(KEYBOARD_KEYS) * (key_height + gap)
    )

    start_y = h - total_height - 8

    for row_index, row in enumerate(KEYBOARD_KEYS):

        widths = [
            special_widths.get(key, normal_width)
            for key in row
        ]

        total_width = sum(widths) + (
            len(row) - 1
        ) * gap

        start_x = (w - total_width) // 2

        current_x = start_x

        y = start_y + row_index * (
            key_height + gap
        )

        for key, width in zip(row, widths):

            key_positions.append(
                (
                    key,
                    current_x,
                    y,
                    width,
                    key_height
                )
            )

            current_x += width + gap

    return key_positions


# ==========================================
# 4. FIND KEY UNDER FINGERTIP
# ==========================================

def find_key(x, y, key_positions):

    for key, kx, ky, width, height in key_positions:

        if (
            kx <= x <= kx + width
            and ky <= y <= ky + height
        ):
            return key

    return None


# ==========================================
# 5. DRAW KEYBOARD
# ==========================================

def draw_keyboard(frame, key_positions, active_keys):

    for key, x, y, width, height in key_positions:

        if key in active_keys:
            color = (0, 180, 0)
        else:
            color = (65, 65, 65)

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            -1
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (255, 255, 255),
            1
        )

        text_size = cv2.getTextSize(
            key,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )[0]

        text_x = x + (
            width - text_size[0]
        ) // 2

        text_y = y + (
            height + text_size[1]
        ) // 2

        cv2.putText(
            frame,
            key,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )


# ==========================================
# 6. TYPE INTO REAL APPLICATION
# ==========================================

def type_key(key):

    global typed_text

    if key == "SPACE":

        keyboard.press(Key.space)
        keyboard.release(Key.space)

        typed_text += " "

    elif key == "BACKSPACE":

        keyboard.press(Key.backspace)
        keyboard.release(Key.backspace)

        typed_text = typed_text[:-1]

    elif key == "ENTER":

        keyboard.press(Key.enter)
        keyboard.release(Key.enter)

        typed_text += "\n"

    elif key is not None:

        keyboard.press(key.lower())
        keyboard.release(key.lower())

        typed_text += key.lower()


# ==========================================
# 7. PRESS KEY WITH COOLDOWN
# ==========================================

def select_key(key):

    global last_action_time

    current_time = time.monotonic()

    if current_time - last_action_time < KEY_COOLDOWN:
        return False

    type_key(key)

    last_action_time = current_time

    print("Typed:", key)

    # Prevent multiple fingers hovering over
    # the same key from typing it repeatedly.
    for state in hover_state.values():

        if state["key"] == key:
            state["fired"] = True

    return True


# ==========================================
# 8. START CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Camera not detected!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

timestamp_ms = 0

print("-----------------------------------")
print(" TWO-HAND AI VIRTUAL KEYBOARD")
print("-----------------------------------")
print("Left hand  : Five fingers")
print("Right hand : Five fingers")
print("Pinch      : Press a key")
print("Hover      : Hold for 0.9 seconds")
print("SPACE      : Insert space")
print("BACKSPACE  : Delete character")
print("ENTER      : New line")
print("Q          : Exit")
print("-----------------------------------")


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

try:

    with HandLandmarker.create_from_options(options) as detector:

        while True:

            success, frame = cap.read()

            if not success:
                print("Camera frame unavailable!")
                break

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

            key_positions = create_keyboard(frame)

            active_keys = set()
            detected_hands = set()

            # ==================================
            # 10. DETECT BOTH HANDS
            # ==================================

            if results.hand_landmarks:

                for hand_index, landmarks in enumerate(
                    results.hand_landmarks
                ):

                    hand_label = (
                        results.handedness[hand_index][0]
                        .category_name
                    )

                    detected_hands.add(hand_label)

                    hand_color = HAND_COLORS.get(
                        hand_label,
                        (255, 255, 255)
                    )

                    # ==================================
                    # 11. TRACK ALL FIVE FINGERS
                    # ==================================

                    thumb = landmarks[4]

                    for finger_name, tip_id in FINGERS.items():

                        fingertip = landmarks[tip_id]

                        finger_x = int(
                            fingertip.x * w
                        )

                        finger_y = int(
                            fingertip.y * h
                        )

                        pointer_id = (
                            hand_label,
                            finger_name
                        )

                        # Find key under fingertip
                        selected_key = find_key(
                            finger_x,
                            finger_y,
                            key_positions
                        )

                        if selected_key is not None:
                            active_keys.add(selected_key)

                        # ==================================
                        # 12. DRAW FINGERTIP CURSOR
                        # ==================================

                        cv2.circle(
                            frame,
                            (finger_x, finger_y),
                            9,
                            hand_color,
                            -1
                        )

                        cv2.circle(
                            frame,
                            (finger_x, finger_y),
                            12,
                            (255, 255, 255),
                            2
                        )

                        # Display finger name
                        cv2.putText(
                            frame,
                            finger_name[:1],
                            (finger_x + 8, finger_y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1
                        )

                        # ==================================
                        # 13. HOVER SELECTION
                        # ==================================

                        if pointer_id not in hover_state:

                            hover_state[pointer_id] = {
                                "key": None,
                                "start": 0,
                                "fired": False
                            }

                        state = hover_state[pointer_id]

                        if selected_key != state["key"]:

                            state["key"] = selected_key
                            state["start"] = time.monotonic()
                            state["fired"] = False

                        if (
                            selected_key is not None
                            and not state["fired"]
                        ):

                            elapsed = (
                                time.monotonic()
                                - state["start"]
                            )

                            if elapsed >= HOVER_DWELL_SECONDS:

                                if select_key(selected_key):
                                    state["fired"] = True

                        # ==================================
                        # 14. PINCH SELECTION
                        # ==================================

                        is_pinching = False

                        # Thumb is the pinch anchor.
                        # Other four fingertips can pinch it.
                        if finger_name != "THUMB":

                            distance = math.hypot(
                                fingertip.x - thumb.x,
                                fingertip.y - thumb.y
                            )

                            is_pinching = (
                                distance < PINCH_THRESHOLD
                            )

                        previous_pinch = pinch_state.get(
                            pointer_id,
                            False
                        )

                        # Detect the start of a pinch
                        if (
                            is_pinching
                            and not previous_pinch
                            and selected_key is not None
                        ):

                            if select_key(selected_key):
                                state["fired"] = True

                        pinch_state[pointer_id] = is_pinching

                    # ==================================
                    # 15. DISPLAY HAND LABEL
                    # ==================================

                    cv2.putText(
                        frame,
                        hand_label + " HAND",
                        (
                            15 if hand_label == "Left" else w - 155,
                            110
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        hand_color,
                        2
                    )

            # ==================================
            # 16. RESET MISSING HAND STATES
            # ==================================

            for pointer_id, state in hover_state.items():

                hand_name = pointer_id[0]

                if hand_name not in detected_hands:

                    state["key"] = None
                    state["start"] = time.monotonic()
                    state["fired"] = False

                    pinch_state[pointer_id] = False

            # ==================================
            # 17. DISPLAY TYPED TEXT
            # ==================================

            cv2.rectangle(
                frame,
                (10, 10),
                (w - 10, 75),
                (30, 30, 30),
                -1
            )

            display_text = typed_text[-40:]

            display_text = display_text.replace(
                "\n",
                " "
            )

            cv2.putText(
                frame,
                "TEXT: " + display_text,
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # ==================================
            # 18. DRAW VIRTUAL KEYBOARD
            # ==================================

            draw_keyboard(
                frame,
                key_positions,
                active_keys
            )

            # ==================================
            # 19. SHOW CAMERA WINDOW
            # ==================================

            cv2.imshow(
                "AI Two-Hand Virtual Keyboard",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# ==========================================
# 20. CLEANUP
# ==========================================

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("AI Virtual Keyboard Closed!")
