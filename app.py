"""
Mood & Gesture AI

"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from types import SimpleNamespace
from typing import List

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh landmark indices
L_MOUTH, R_MOUTH = 61, 291
UP_LIP, LO_LIP = 13, 14
L_INNER_BROW, R_INNER_BROW = 107, 336
L_EYE_TOP, L_EYE_BOT = 159, 145
R_EYE_TOP, R_EYE_BOT = 386, 374
NOSE, CHIN = 1, 152

# MediaPipe Hands
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

MOOD_COLOR = {
    "Happy": (80, 220, 120),
    "Sad": (220, 160, 80),
    "Angry": (60, 60, 255),
    "Surprised": (0, 210, 255),
    "Neutral": (200, 200, 200),
    "No face": (140, 140, 140),
}


def dist(a, b) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


class LabelSmoother:
    """Majority vote over recent frames so labels do not flicker."""

    def __init__(self, window: int = 8):
        self.votes: deque = deque(maxlen=window)

    def update(self, label: str) -> str:
        self.votes.append(label)
        return Counter(self.votes).most_common(1)[0][0]


def classify_mood(lm) -> str:
    """Rule-based expression from Face Mesh landmarks (normalized 0–1)."""
    mouth_w = dist(lm[L_MOUTH], lm[R_MOUTH])
    mouth_h = dist(lm[UP_LIP], lm[LO_LIP])
    face_h = dist(lm[NOSE], lm[CHIN]) or 1e-6

    open_ratio = mouth_h / (mouth_w + 1e-6)
    brow_eye = (
        abs(lm[L_INNER_BROW].y - lm[L_EYE_TOP].y)
        + abs(lm[R_INNER_BROW].y - lm[R_EYE_TOP].y)
    ) / 2
    eye_open = (
        dist(lm[L_EYE_TOP], lm[L_EYE_BOT]) + dist(lm[R_EYE_TOP], lm[R_EYE_BOT])
    ) / 2
    mouth_mid_y = (lm[UP_LIP].y + lm[LO_LIP].y) / 2
    corner_lift = mouth_mid_y - (lm[L_MOUTH].y + lm[R_MOUTH].y) / 2
    smile_width = mouth_w / face_h

    if open_ratio > 0.35 and eye_open > 0.025:
        return "Surprised"
    if corner_lift > 0.005 and smile_width > 0.35:
        return "Happy"
    if corner_lift < -0.008:
        return "Sad"
    if brow_eye < 0.035 and open_ratio < 0.20:
        return "Angry"
    return "Neutral"


def finger_up(lm, tip: int, pip: int) -> bool:
    return dist(lm[tip], lm[WRIST]) > dist(lm[pip], lm[WRIST]) * 1.12


def thumb_extended(lm, handedness: str) -> bool:
    tip, mcp = lm[THUMB_TIP], lm[THUMB_MCP]
    sideways = (tip.x < mcp.x) if handedness == "Right" else (tip.x > mcp.x)
    return sideways and dist(tip, lm[WRIST]) > dist(lm[THUMB_IP], lm[WRIST])


def classify_gesture(lm, handedness: str) -> str:
    thumb = thumb_extended(lm, handedness)
    index = finger_up(lm, INDEX_TIP, INDEX_PIP)
    middle = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring = finger_up(lm, RING_TIP, RING_PIP)
    pinky = finger_up(lm, PINKY_TIP, PINKY_PIP)

    n = sum([thumb, index, middle, ring, pinky])
    # OK = thumb–index pinch with the other fingers still up (a collapsed fist is not OK)
    if dist(lm[THUMB_TIP], lm[INDEX_TIP]) < 0.05 and (middle or ring or pinky):
        return "OK"
    if n == 5:
        return "Open palm"
    if n == 0:
        return "Fist"
    if thumb and not any([index, middle, ring, pinky]):
        return "Thumbs up" if lm[THUMB_TIP].y < lm[WRIST].y else "Thumbs down"
    if index and middle and not ring and not pinky and not thumb:
        return "Peace"
    if index and not middle and not ring and not pinky:
        return "Pointing"
    if index and pinky and not middle and not ring:
        return "Rock"
    if thumb and pinky and not index and not middle and not ring:
        return "Call me"
    return "Unknown"


def draw_panel(frame, mood: str, gestures: List[str], fps: float) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (420, 140), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    accent = MOOD_COLOR.get(mood, (200, 200, 200))
    cv2.rectangle(frame, (16, 16), (22, 140), accent, -1)
    cv2.putText(frame, "Mood & Gesture AI", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"Mood:  {mood}", (36, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent, 2)
    gtxt = "  |  ".join(gestures) if gestures else "No hands"
    cv2.putText(frame, f"Hands: {gtxt}", (36, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 255, 170), 2)
    cv2.putText(frame, f"{fps:.0f} FPS   q quit   s snapshot", (w - 340, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)


def _pt(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0)


def _blank(n: int) -> List[SimpleNamespace]:
    return [_pt(0.5, 0.5) for _ in range(n)]


def self_test() -> None:
    """Sanity-check classifiers without a camera (useful for a resume demo write-up)."""
    face = _blank(468)
    face[NOSE], face[CHIN] = _pt(0.5, 0.45), _pt(0.5, 0.80)
    face[L_MOUTH], face[R_MOUTH] = _pt(0.38, 0.62), _pt(0.62, 0.62)
    face[UP_LIP], face[LO_LIP] = _pt(0.50, 0.60), _pt(0.50, 0.64)
    face[L_INNER_BROW], face[R_INNER_BROW] = _pt(0.40, 0.32), _pt(0.60, 0.32)
    face[L_EYE_TOP], face[L_EYE_BOT] = _pt(0.40, 0.38), _pt(0.40, 0.42)
    face[R_EYE_TOP], face[R_EYE_BOT] = _pt(0.60, 0.38), _pt(0.60, 0.42)
    assert classify_mood(face) == "Neutral", classify_mood(face)

    smile = list(face)
    smile[L_MOUTH], smile[R_MOUTH] = _pt(0.32, 0.60), _pt(0.68, 0.60)
    assert classify_mood(smile) == "Happy", classify_mood(smile)

    hand = _blank(21)
    # Fist: all tips close to wrist
    for i in range(21):
        hand[i] = _pt(0.5, 0.5)
    assert classify_gesture(hand, "Right") == "Fist"

    # Pointing: only index stretched away from wrist
    hand[INDEX_TIP] = _pt(0.5, 0.15)
    hand[INDEX_PIP] = _pt(0.5, 0.30)
    assert classify_gesture(hand, "Right") == "Pointing", classify_gesture(hand, "Right")

    ok_hand = _blank(21)
    for i in range(21):
        ok_hand[i] = _pt(0.5, 0.55)
    ok_hand[THUMB_TIP] = _pt(0.48, 0.40)
    ok_hand[INDEX_TIP] = _pt(0.50, 0.40)
    ok_hand[MIDDLE_TIP], ok_hand[MIDDLE_PIP] = _pt(0.5, 0.10), _pt(0.5, 0.30)
    ok_hand[RING_TIP], ok_hand[RING_PIP] = _pt(0.55, 0.10), _pt(0.55, 0.30)
    ok_hand[PINKY_TIP], ok_hand[PINKY_PIP] = _pt(0.60, 0.12), _pt(0.60, 0.32)
    assert classify_gesture(ok_hand, "Right") == "OK", classify_gesture(ok_hand, "Right")
    print("self-test: ok")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect facial mood and hand gestures from a webcam.")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--self-test", action="store_true", help="Run classifier checks and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    face_spec = mp_draw.DrawingSpec(color=(80, 180, 255), thickness=1, circle_radius=1)
    hand_spec = mp_draw.DrawingSpec(color=(0, 255, 140), thickness=2, circle_radius=2)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open camera. On macOS: System Settings > Privacy & Security > Camera, "
            "and allow Terminal or your IDE."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mood_smooth = LabelSmoother(10)
    gesture_smooth = [LabelSmoother(8), LabelSmoother(8)]
    prev = time.time()
    fps = 0.0

    try:
        with mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Lost camera frame. Is another app using the webcam?")
                    break

                # Selfie view. MediaPipe handedness is relative to the unflipped image,
                # so swap Left/Right after the flip.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                face_res = face_mesh.process(rgb)
                hand_res = hands.process(rgb)
                rgb.flags.writeable = True

                mood = "No face"
                if face_res.multi_face_landmarks:
                    flm = face_res.multi_face_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame,
                        flm,
                        mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=face_spec,
                    )
                    mood = mood_smooth.update(classify_mood(flm.landmark))

                shown: List[str] = []
                if hand_res.multi_hand_landmarks:
                    labels = [h.classification[0].label for h in (hand_res.multi_handedness or [])]
                    for i, hlm in enumerate(hand_res.multi_hand_landmarks):
                        mp_draw.draw_landmarks(frame, hlm, mp_hands.HAND_CONNECTIONS, hand_spec, hand_spec)
                        raw = labels[i] if i < len(labels) else "Right"
                        hand = "Left" if raw == "Right" else "Right"
                        label = classify_gesture(hlm.landmark, hand)
                        if i < len(gesture_smooth):
                            label = gesture_smooth[i].update(label)
                        shown.append(f"{hand}: {label}")

                now = time.time()
                dt = now - prev
                prev = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                draw_panel(frame, mood, shown, fps)

                cv2.imshow("Mood & Gesture AI", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    name = time.strftime("snapshot_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(name, frame)
                    print(f"saved {name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
