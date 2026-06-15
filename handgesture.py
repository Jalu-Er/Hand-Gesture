import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@dataclass
class HandInfo:
    label: str
    digit: int
    open_fingers: list
    is_fist: bool
    center_x: float
    center_y: float
    landmarks: list


class HandGesture:
    def __init__(self, camera=0, cooldown=1.0, swipe_threshold=0.13, dry_run=False):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "File hand_landmarker.task belum ada. Letakkan model MediaPipe Hand Landmarker "
                "di folder yang sama dengan handgesture.py."
            )

        self.camera = camera
        self.cooldown = cooldown
        self.swipe_threshold = swipe_threshold
        self.dry_run = dry_run
        self.last_action_time = 0
        self.last_action = "Idle"
        self.pending_slide = None
        self.hold_slide = None
        self.hold_started_at = None
        self.hold_seconds = 1.0
        self.candidate_slide = None
        self.candidate_started_at = None
        self.stable_seconds = 0.20
        self.was_confirming = False
        self.active_min_x = 0.15
        self.active_max_x = 0.85
        self.is_frozen = False
        self.freeze_started_at = None
        self.freeze_seconds = 2.0
        self.freeze_latched = False
        self.position_history = deque(maxlen=24)

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        pyautogui.PAUSE = 0.03

    def can_trigger(self):
        return time.time() - self.last_action_time >= self.cooldown

    def press(self, key, action):
        if not self.can_trigger():
            return
        if not self.dry_run:
            pyautogui.press(key)
        self.last_action = action
        self.last_action_time = time.time()

    def jump(self, slide_number):
        if slide_number <= 0 or not self.can_trigger():
            return
        if not self.dry_run:
            pyautogui.write(str(slide_number))
            pyautogui.press("enter")
        self.last_action = f"Jump slide {slide_number}"
        self.pending_slide = None
        self.hold_slide = None
        self.hold_started_at = None
        self.candidate_slide = None
        self.candidate_started_at = None
        self.was_confirming = True
        self.position_history.clear()
        self.last_action_time = time.time()

    def distance(self, a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    def normalize_frame_for_detection(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())

        if brightness > 190:
            return cv2.convertScaleAbs(frame, alpha=0.82, beta=-18)
        if brightness < 70:
            return cv2.convertScaleAbs(frame, alpha=1.18, beta=16)
        return frame

    def is_in_active_zone(self, center_x):
        return self.active_min_x <= center_x <= self.active_max_x

    def is_not_pointing_down(self, landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        hand_size = max(self.distance(wrist, middle_mcp), 0.001)
        return middle_mcp.y <= wrist.y + (hand_size * 0.08)

    def open_fingers(self, landmarks):
        fingers = {}

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        hand_size = max(self.distance(wrist, middle_mcp), 0.001)
        palm_left = min(index_mcp.x, pinky_mcp.x)
        palm_right = max(index_mcp.x, pinky_mcp.x)
        palm_width = max(palm_right - palm_left, 0.001)
        thumb_outside_palm = thumb_tip.x < palm_left - (palm_width * 0.18) or thumb_tip.x > palm_right + (palm_width * 0.18)
        thumb_far_from_palm = self.distance(thumb_tip, index_mcp) > hand_size * 0.48
        thumb_extended_from_wrist = self.distance(thumb_tip, wrist) > self.distance(thumb_ip, wrist) + (hand_size * 0.02)
        fingers["thumb"] = thumb_outside_palm and thumb_far_from_palm and thumb_extended_from_wrist

        fingers["index"] = landmarks[8].y < landmarks[6].y - 0.025
        fingers["middle"] = landmarks[12].y < landmarks[10].y - 0.025
        fingers["ring"] = landmarks[16].y < landmarks[14].y - 0.025
        fingers["pinky"] = landmarks[20].y < landmarks[18].y - 0.025

        return fingers

    def finger_digit(self, fingers):
        thumb = fingers["thumb"]
        index = fingers["index"]
        middle = fingers["middle"]
        ring = fingers["ring"]
        pinky = fingers["pinky"]

        pattern = (thumb, index, middle, ring, pinky)
        digit_map = {
            (False, False, False, False, False): 0,
            (False, True, False, False, False): 1,
            (False, True, True, False, False): 2,
            (False, True, True, True, False): 3,
            (False, True, True, True, True): 4,
            (True, False, False, False, False): 5,
            (True, True, False, False, False): 6,
            (True, True, True, False, False): 7,
            (True, False, False, False, True): 8,
            (True, True, True, True, True): 9,
        }

        return digit_map.get(pattern)

    def center(self, landmarks):
        return (
            sum(point.x for point in landmarks) / len(landmarks),
            sum(point.y for point in landmarks) / len(landmarks),
        )

    def read_hands(self, result):
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            center_x, center_y = self.center(landmarks)
            if not self.is_in_active_zone(center_x) or not self.is_not_pointing_down(landmarks):
                continue

            label = handedness[0].category_name
            fingers = self.open_fingers(landmarks)
            digit = self.finger_digit(fingers)
            is_fist = not any(fingers.values())
            if is_fist:
                digit = 0
            hands.append(
                HandInfo(
                    label=label,
                    digit=digit,
                    open_fingers=list(fingers.values()),
                    is_fist=is_fist,
                    center_x=center_x,
                    center_y=center_y,
                    landmarks=landmarks,
                )
            )
        return hands

    def handle_freeze(self, hands):
        freeze_hand = None
        if len(hands) == 1 and hands[0].center_x >= 0.50 and hands[0].digit == 1:
            freeze_hand = hands[0]

        if freeze_hand is None:
            self.freeze_started_at = None
            self.freeze_latched = False
            return

        now = time.time()
        if self.freeze_started_at is None:
            self.freeze_started_at = now
            return

        if not self.freeze_latched and now - self.freeze_started_at >= self.freeze_seconds:
            self.is_frozen = not self.is_frozen
            self.freeze_latched = True
            self.last_action = "Frozen" if self.is_frozen else "Unfrozen"
            self.position_history.clear()
            self.pending_slide = None
            self.hold_slide = None
            self.hold_started_at = None
            self.candidate_slide = None
            self.candidate_started_at = None

    def handle_jump(self, hands):
        if len(hands) != 2:
            self.candidate_slide = None
            self.candidate_started_at = None
            return

        left, right = sorted(hands, key=lambda hand: hand.center_x)

        if left.is_fist and right.is_fist:
            if self.was_confirming:
                return
            hold_ready = (
                self.hold_slide is not None
                and self.hold_started_at is not None
                and time.time() - self.hold_started_at >= self.hold_seconds
            )
            if hold_ready:
                self.jump(self.hold_slide)
            elif self.hold_slide is not None:
                self.last_action = "Hold target first"
            return

        self.was_confirming = False

        if left.digit is None or right.digit is None:
            return

        current_slide = (left.digit * 10) + right.digit
        now = time.time()
        if self.candidate_slide != current_slide:
            self.candidate_slide = current_slide
            self.candidate_started_at = now
            return

        if self.candidate_started_at is None or now - self.candidate_started_at < self.stable_seconds:
            return

        self.pending_slide = current_slide

        if self.hold_slide != current_slide:
            self.hold_slide = current_slide
            self.hold_started_at = now

    def handle_swipe(self, hands):
        if len(hands) != 1:
            self.position_history.clear()
            return

        hand = hands[0]
        if hand.is_fist:
            self.position_history.clear()
            return

        self.position_history.append((time.time(), hand.center_x, hand.center_y))
        if len(self.position_history) < 6:
            return

        start_time, start_x, _ = self.position_history[0]
        end_time, end_x, _ = self.position_history[-1]
        elapsed = end_time - start_time
        delta_x = end_x - start_x

        if elapsed > 2.8:
            self.position_history.clear()
            return

        vertical_drift = max(abs(point[2] - self.position_history[0][2]) for point in self.position_history)
        if abs(delta_x) >= self.swipe_threshold and vertical_drift < 0.22:
            if delta_x > 0:
                self.press("right", "Next slide")
            else:
                self.press("left", "Previous slide")
            self.position_history.clear()

    def draw_swipe_tail(self, frame):
        if len(self.position_history) < 2:
            return

        height, width = frame.shape[:2]
        points = [(int(x * width), int(y * height)) for _, x, y in self.position_history]
        total = len(points) - 1
        for index in range(total):
            alpha = (index + 1) / total
            color = (0, int(120 + 135 * alpha), 255)
            thickness = max(2, int(6 * alpha))
            cv2.line(frame, points[index], points[index + 1], color, thickness)

        cv2.circle(frame, points[-1], 9, (0, 255, 255), -1)

    def draw_panel(self, frame, x1, y1, x2, y2, fill=(22, 26, 34), border=(70, 82, 96)):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)

    def draw_progress_bar(self, frame, x, y, width, height, progress):
        progress = max(0.0, min(1.0, progress))
        cv2.rectangle(frame, (x, y), (x + width, y + height), (52, 60, 72), -1)
        cv2.rectangle(frame, (x, y), (x + int(width * progress), y + height), (0, 210, 150), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 112, 128), 1)

    def draw_hand(self, frame, hand):
        height, width = frame.shape[:2]
        points = [(int(p.x * width), int(p.y * height)) for p in hand.landmarks]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (245, 248, 255), 2)

        for index, point in enumerate(points):
            color = (0, 220, 255) if index in TIP_IDS else (80, 220, 130)
            cv2.circle(frame, point, 6, (18, 22, 28), -1)
            cv2.circle(frame, point, 4, color, -1)

        x = int(hand.center_x * width)
        y = int(hand.center_y * height)
        digit = "?" if hand.digit is None else str(hand.digit)
        self.draw_panel(frame, x - 58, y - 56, x + 58, y + 8, fill=(18, 22, 28), border=(78, 92, 110))
        cv2.putText(frame, f"{hand.label}", (x - 45, y - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 235), 1)
        cv2.putText(frame, digit, (x + 24, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2)
        states = "".join(name if is_open else "-" for name, is_open in zip("TIMRP", hand.open_fingers))
        cv2.putText(frame, states, (x - 45, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)

    def draw_ui(self, frame, hands, fps):
        height, width = frame.shape[:2]
        left_zone = int(width * self.active_min_x)
        right_zone = int(width * self.active_max_x)
        cv2.line(frame, (left_zone, 0), (left_zone, height), (0, 210, 255), 1)
        cv2.line(frame, (right_zone, 0), (right_zone, height), (0, 210, 255), 1)

        self.draw_panel(frame, 12, 12, width - 12, 132, fill=(18, 22, 28), border=(82, 96, 116))
        cv2.putText(frame, "handgesture", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, "presentation control", (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (165, 178, 195), 1)

        target = "--" if self.pending_slide is None else f"{self.pending_slide:02d}"
        hold_progress = 0.0
        if self.hold_slide is not None and self.hold_started_at is not None:
            elapsed = time.time() - self.hold_started_at
            hold_left = max(0.0, self.hold_seconds - elapsed)
            hold_progress = min(1.0, elapsed / self.hold_seconds)
            hold_text = "READY" if hold_left <= 0 else f"HOLD {hold_left:.1f}s"
            hold_color = (0, 230, 150) if hold_left <= 0 else (0, 210, 255)
        else:
            hold_text = "HOLD -"
            hold_color = (170, 180, 195)

        freeze_progress = 0.0
        if self.freeze_started_at is not None:
            freeze_progress = min(1.0, (time.time() - self.freeze_started_at) / self.freeze_seconds)

        dry = "ON" if self.dry_run else "OFF"

        card_w = 120
        start_x = width - 398
        cards = [
            ("TARGET", target, (0, 230, 255)),
            ("STATUS", "FROZEN" if self.is_frozen else hold_text, (90, 170, 255) if self.is_frozen else hold_color),
            ("FPS", f"{fps:.0f}", (210, 220, 235)),
        ]
        for i, (label, value, color) in enumerate(cards):
            x = start_x + (card_w + 12) * i
            self.draw_panel(frame, x, 28, x + card_w, 108, fill=(28, 34, 44), border=(74, 88, 106))
            cv2.putText(frame, label, (x + 13, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 174, 192), 1)
            cv2.putText(frame, value, (x + 13, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)

        self.draw_progress_bar(frame, start_x + card_w + 25, 96, card_w - 26, 7, hold_progress)
        cv2.putText(frame, f"Action: {self.last_action}", (28, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (125, 220, 255), 2)
        cv2.putText(frame, f"Dry-run {dry}", (width - 116, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 205), 1)

        self.draw_panel(frame, 14, height - 112, 360, height - 14, fill=(18, 22, 28), border=(72, 86, 104))
        cv2.putText(frame, "LEFT SIDE = TENS", (32, height - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 235, 245), 1)
        cv2.putText(frame, "RIGHT SIDE = UNITS", (32, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 235, 245), 1)
        cv2.putText(frame, "Hold 1s until READY, then make both fists", (32, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 230, 190), 1)

        self.draw_panel(frame, width - 286, height - 112, width - 14, height - 14, fill=(18, 22, 28), border=(72, 86, 104))
        cv2.putText(frame, "Right side 1 = freeze", (width - 268, height - 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
        cv2.putText(frame, "Hold 2s to toggle", (width - 268, height - 51), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1)
        self.draw_progress_bar(frame, width - 268, height - 38, 166, 7, freeze_progress)
        cv2.putText(frame, "Q", (width - 52, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 230, 255), 2)

        if self.is_frozen:
            self.draw_panel(frame, width // 2 - 130, height // 2 - 45, width // 2 + 130, height // 2 + 45, fill=(18, 26, 42), border=(90, 170, 255))
            cv2.putText(frame, "FROZEN", (width // 2 - 82, height // 2 + 9), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (90, 190, 255), 3)

        self.draw_swipe_tail(frame)
        for hand in hands:
            self.draw_hand(frame, hand)

    def run(self):
        cap = cv2.VideoCapture(self.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Kamera index {self.camera} tidak bisa dibuka.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        prev_time = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                detection_frame = self.normalize_frame_for_detection(frame)
                rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(time.time() * 1000)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                hands = self.read_hands(result)

                self.handle_freeze(hands)
                if self.is_frozen:
                    self.position_history.clear()
                else:
                    self.handle_jump(hands)
                    self.handle_swipe(hands)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 0.001)
                prev_time = now

                self.draw_ui(frame, hands, fps)
                cv2.imshow("handgesture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.landmarker.close()


def parse_args():
    parser = argparse.ArgumentParser(description="handgesture - kontrol slide presentasi dengan gesture tangan.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--cooldown", type=float, default=1.0)
    parser.add_argument("--swipe-threshold", type=float, default=0.13)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    app = HandGesture(
        camera=args.camera,
        cooldown=args.cooldown,
        swipe_threshold=args.swipe_threshold,
        dry_run=args.dry_run,
    )
    app.run()


if __name__ == "__main__":
    main()
