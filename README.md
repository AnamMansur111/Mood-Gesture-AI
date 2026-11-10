# Mood & Gesture AI

Mood & Gesture AI is a Python project that uses a webcam to detect facial expressions and hand gestures in real time. It uses OpenCV and MediaPipe to track facial and hand landmarks and identifies expressions and gestures based on their positions.

## Features

**Facial expressions**
- Happy
- Sad
- Angry
- Surprised
- Neutral

**Hand gestures**
- Open palm
- Fist
- Thumbs up and thumbs down
- Peace sign
- Pointing
- OK sign
- Rock sign
- Call me gesture

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

The application captures video from the webcam and uses MediaPipe to detect facial and hand landmarks. It then checks the positions of facial features and fingers to identify expressions and gestures. The results are displayed on the live camera feed.

The project uses predefined rules to classify expressions and gestures rather than training a model from scratch. It also uses recent predictions to make the displayed labels more stable.

## How to Run

1.⁠ ⁠Clone or download this repository.


```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install the required libraries:

```bash
pip install -r requirements.txt
```

5. Start the application:

```bash
python app.py
```

Allow camera access if macOS asks for permission.

## Controls

- Press `q` or `Esc` to close the application.
- Press `s` to save a screenshot.
- Use `python app.py --camera 1` to try another camera.
- Run `python app.py --self-test` to test the classification logic without opening the webcam.

## Project Structure

```text
Mood-Gesture-AI/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```
