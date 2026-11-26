# 🤖 AI Virtual Mouse & Keyboard

An AI-powered virtual mouse and keyboard system that allows users to control their computer using hand gestures through a webcam.

This project uses **Computer Vision and Hand Tracking** to detect hand movements and finger gestures, enabling touch-free interaction with a computer.

## 🚀 Project Overview

The AI Virtual Mouse & Keyboard allows users to:
- Move the mouse cursor using hand movements.
- Perform mouse clicks using finger gestures.
- Scroll up and down using hand gestures.
- Type using a virtual keyboard displayed on the screen.
- Use both left and right hands for keyboard interaction.
- Select keyboard keys using pinch gestures and hover gestures.

## ✨ Features

### 🖱️ AI Virtual Mouse
- Real-time hand tracking.
- Smooth mouse cursor movement.
- Left-click using pinch gestures.
- Two-finger scrolling.
- Supports left and right hand detection.

### ⌨️ AI Virtual Keyboard
- Virtual QWERTY keyboard.
- Supports both hands and all ten fingers.
- Hover-based key selection.
- Pinch-based key selection.
- Supports Space, Backspace, and Enter.
- Sends actual keyboard input to the active application.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Webcam access and image processing |
| MediaPipe | Hand and finger tracking |
| NumPy | Numerical operations |
| pynput | Mouse and keyboard control |

## 📂 Project Structure

```text
AI_Virtual_Mouse_Keyboard/
│
├── hand_detection.py
├── finger_detection.py
├── mouse_controller.py
├── virtual_keyboard.py
├── hand_landmarker.task
├── README.md
└── .venu/
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
https://github.com/simplelife277/AI-Virtual-Mouse-Keyboard
```

### 2. Open the Project Folder

```bash
cd AI_Virtual_Mouse_Keyboard
```

### 3. Install Required Libraries

```bash
python -m pip install opencv-contrib-python mediapipe numpy pynput
```

### 4. Download the MediaPipe Model

Download `hand_landmarker.task` from:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place the downloaded file in the project root folder.

## ▶️ How to Run

### Run Hand Detection

```bash
python hand_detection.py
```

### Run Finger Detection

```bash
python finger_detection.py
```

### Run AI Virtual Mouse

```bash
python mouse_controller.py
```

### Run AI Virtual Keyboard

```bash
python virtual_keyboard.py
```

Press **Q** to close the camera window.

## 🧠 How It Works

1. The webcam captures real-time video.
2. MediaPipe detects hands and identifies 21 landmarks on each hand.
3. Finger positions are tracked using landmark coordinates.
4. Python processes the detected gestures.
5. OpenCV displays the tracking results and virtual keyboard.
6. pynput sends mouse and keyboard actions to the computer.

## 🎯 Project Objectives

- Explore real-time computer vision.
- Understand hand landmark detection.
- Build gesture-based human-computer interaction.
- Control mouse and keyboard without physical contact.
- Improve Python programming and problem-solving skills.

## 🔮 Future Enhancements

- Add voice commands.
- Improve gesture recognition accuracy.
- Add customizable keyboard layouts.
- Support multiple languages.
- Develop a graphical settings interface.
- Improve gesture calibration and accessibility.

## 👨‍💻 Author

**Saravanan T.**

Python Developer | Computer Vision | Backend Development

GitHub: https://github.com/simplelife277/AI-Virtual-Mouse-Keyboard

---

⭐ If you find this project interesting, consider giving it a star!

**Built with Python and Computer Vision.**
