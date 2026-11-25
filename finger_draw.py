from __future__ import annotations

import os
import time
import urllib.request
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TOOLBAR_HEIGHT = 75


def ensure_hand_model() -> str:
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task (~12 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_toolbar(width: int) -> List[Dict]:
    items = [
        {"id": "black", "label": "BLACK", "color": (25, 25, 25), "type": "color"},
        {"id": "red", "label": "RED", "color": (45, 45, 235), "type": "color"},
        {"id": "blue", "label": "BLUE", "color": (235, 115, 30), "type": "color"},
        {"id": "green", "label": "GREEN", "color": (45, 185, 65), "type": "color"},
        {"id": "eraser", "label": "ERASER", "color": (210, 210, 210), "type": "tool"},
        {"id": "size", "label": "SIZE", "color": (90, 90, 90), "type": "action"},
        {"id": "undo", "label": "UNDO", "color": (90, 90, 90), "type": "action"},
        {"id": "clear", "label": "CLEAR", "color": (90, 90, 90), "type": "action"},
        {"id": "paper", "label": "PAPER", "color": (90, 90, 90), "type": "action"},
    ]
    btn_w = width // len(items)
    for idx, item in enumerate(items):
        item["x1"] = idx * btn_w
        item["x2"] = width if idx == len(items) - 1 else (idx + 1) * btn_w
        item["y1"] = 0
        item["y2"] = TOOLBAR_HEIGHT
    return items


def draw_toolbar(
    frame: np.ndarray,
    buttons: List[Dict],
    active_tool: str,
    brush_size: int,
    paper_mode: bool,
):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], TOOLBAR_HEIGHT), (38, 38, 42), -1)

    for btn in buttons:
        x1, y1, x2, y2 = btn["x1"], btn["y1"], btn["x2"], btn["y2"]
        pad = 6
        bx1, by1, bx2, by2 = x1 + pad, y1 + pad, x2 - pad, y2 - pad

        label = btn["label"]
        if btn["id"] == "size":
            label = f"SIZE:{brush_size}"
        elif btn["id"] == "paper":
            label = "CAM" if paper_mode else "PAPER"

        fill_color = btn["color"]
        if btn["id"] == "paper" and paper_mode:
            fill_color = (60, 140, 220)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), fill_color, -1)

        # Highlight currently active pen or eraser
        is_selected = btn["id"] == active_tool
        border_color = (0, 230, 255) if is_selected else (150, 150, 150)
        border_thick = 3 if is_selected else 1
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_color, border_thick)

        text_color = (20, 20, 20) if btn["id"] == "eraser" else (255, 255, 255)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        tx = bx1 + max((bx2 - bx1 - tw) // 2, 4)
        ty = by1 + (by2 - by1 + th) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 2)


def run_finger_drawer(source: int = 0, weights: str = "yolo26n.pt"):
    device = get_device()
    yolo_model = YOLO(weights)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_hand_model()),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source}")

    strokes_bgr: Optional[np.ndarray] = None
    strokes_mask: Optional[np.ndarray] = None
    undo_stack: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=15)

    prev_point: Optional[Tuple[int, int]] = None
    smooth_point: Optional[Tuple[int, int]] = None

    active_tool = "black"
    brush_color = (25, 25, 25)
    brush_sizes = [4, 8, 16]
    size_idx = 1
    brush_thickness = brush_sizes[size_idx]
    eraser_thickness = 34

    paper_mode = True   # Start on clean white paper (press P or tap toolbar to switch to Camera AR)
    show_yolo = True
    smoothing = 0.65

    start_time = time.perf_counter()
    prev_time = start_time
    last_timestamp_ms = -1
    last_action_time = 0.0
    buttons: Optional[List[Dict]] = None

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if strokes_bgr is None or strokes_mask is None:
                strokes_bgr = np.zeros((h, w, 3), dtype=np.uint8)
                strokes_mask = np.zeros((h, w), dtype=np.uint8)
                buttons = build_toolbar(w)

            # 1. Run YOLO26 detection on Apple Silicon MPS
            if show_yolo:
                results = yolo_model.predict(
                    frame, device=device, classes=[0], conf=0.45, verbose=False
                )
                cam_view = results[0].plot()
            else:
                cam_view = frame.copy()

            # 2. Detect hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            hand_results = landmarker.detect_for_video(mp_image, timestamp_ms)
            mode_text = "No Hand"
            is_drawing_gesture = False

            if hand_results.hand_landmarks:
                lm = hand_results.hand_landmarks[0]
                index_tip, index_pip = lm[8], lm[6]
                middle_tip, middle_pip = lm[12], lm[10]

                raw_x = int(index_tip.x * w)
                raw_y = int(index_tip.y * h)

                if smooth_point is None:
                    smooth_point = (raw_x, raw_y)
                else:
                    sx = int(smoothing * raw_x + (1 - smoothing) * smooth_point[0])
                    sy = int(smoothing * raw_y + (1 - smoothing) * smooth_point[1])
                    smooth_point = (sx, sy)

                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y

                if index_up and not middle_up:
                    is_drawing_gesture = True
                    px, py = smooth_point

                    # Check if finger is tapping the top toolbar
                    if py < TOOLBAR_HEIGHT and buttons is not None:
                        prev_point = None
                        mode_text = "TOOLBAR"
                        now = time.perf_counter()

                        for btn in buttons:
                            if btn["x1"] <= px <= btn["x2"]:
                                if btn["type"] in ("color", "tool"):
                                    active_tool = btn["id"]
                                    if btn["type"] == "color":
                                        brush_color = btn["color"]
                                elif btn["type"] == "action" and (now - last_action_time) > 0.65:
                                    last_action_time = now
                                    if btn["id"] == "size":
                                        size_idx = (size_idx + 1) % len(brush_sizes)
                                        brush_thickness = brush_sizes[size_idx]
                                    elif btn["id"] == "undo" and undo_stack:
                                        strokes_bgr, strokes_mask = undo_stack.pop()
                                    elif btn["id"] == "clear":
                                        undo_stack.append((strokes_bgr.copy(), strokes_mask.copy()))
                                        strokes_bgr.fill(0)
                                        strokes_mask.fill(0)
                                    elif btn["id"] == "paper":
                                        paper_mode = not paper_mode
                                break
                    else:
                        # Save snapshot before starting a new stroke for Undo
                        if prev_point is None:
                            undo_stack.append((strokes_bgr.copy(), strokes_mask.copy()))

                        if active_tool == "eraser":
                            mode_text = "ERASING"
                            if prev_point is not None:
                                cv2.line(strokes_mask, prev_point, smooth_point, 0, eraser_thickness)
                            cv2.circle(strokes_mask, smooth_point, eraser_thickness // 2, 0, -1)
                        else:
                            mode_text = f"DRAW ({active_tool.upper()})"
                            if prev_point is not None:
                                cv2.line(strokes_bgr, prev_point, smooth_point, brush_color, brush_thickness)
                                cv2.line(strokes_mask, prev_point, smooth_point, 255, brush_thickness)
                            cv2.circle(strokes_bgr, smooth_point, brush_thickness // 2, brush_color, -1)
                            cv2.circle(strokes_mask, smooth_point, brush_thickness // 2, 255, -1)

                        prev_point = smooth_point
                else:
                    mode_text = "HOVER (Pen Lifted)"
                    prev_point = None
            else:
                prev_point = None
                smooth_point = None

            # 3. Build background (White Paper or Live Camera) and composite strokes
            if paper_mode:
                base_canvas = np.full((h, w, 3), (248, 248, 245), dtype=np.uint8)
            else:
                base_canvas = cam_view.copy()

            inv_mask = cv2.bitwise_not(strokes_mask)
            clean_artwork = cv2.bitwise_and(base_canvas, base_canvas, mask=inv_mask)
            fg_strokes = cv2.bitwise_and(strokes_bgr, strokes_bgr, mask=strokes_mask)
            clean_artwork = cv2.add(clean_artwork, fg_strokes)

            display = clean_artwork.copy()

            # 4. In Paper Mode, show a small webcam PiP window in the bottom-right corner
            if paper_mode:
                pip_w, pip_h = w // 5, h // 5
                pip = cv2.resize(cam_view, (pip_w, pip_h))
                if smooth_point is not None:
                    pip_pt = (int(smooth_point[0] * pip_w / w), int(smooth_point[1] * pip_h / h))
                    cv2.circle(pip, pip_pt, 5, (0, 200, 0), -1)
                x_off, y_off = w - pip_w - 15, h - pip_h - 40
                display[y_off : y_off + pip_h, x_off : x_off + pip_w] = pip
                cv2.rectangle(display, (x_off, y_off), (x_off + pip_w, y_off + pip_h), (120, 120, 120), 2)

            # 5. Draw finger cursor indicator
            if smooth_point is not None:
                if active_tool == "eraser":
                    cv2.circle(display, smooth_point, eraser_thickness // 2, (80, 80, 80), 2)
                elif is_drawing_gesture:
                    cv2.circle(display, smooth_point, max(brush_thickness // 2, 5), brush_color, -1)
                else:
                    cv2.circle(display, smooth_point, 10, brush_color, 2)

            # 6. Draw top toolbar & footer HUD
            if buttons is not None:
                draw_toolbar(display, buttons, active_tool, brush_thickness, paper_mode)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            hud_color = (40, 40, 40) if paper_mode else (255, 255, 255)
            cv2.putText(
                display,
                f"FPS: {fps:.1f} | {mode_text} | [Z] Undo  [C] Clear  [P] Paper/Cam  [Y] YOLO  [S] Save  [Q] Quit",
                (15, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                hud_color,
                1,
            )

            cv2.imshow("YOLO + Finger Paper Canvas", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z") and undo_stack:
                strokes_bgr, strokes_mask = undo_stack.pop()
                prev_point = None
            elif key == ord("c"):
                undo_stack.append((strokes_bgr.copy(), strokes_mask.copy()))
                strokes_bgr.fill(0)
                strokes_mask.fill(0)
                prev_point = None
            elif key == ord("p"):
                paper_mode = not paper_mode
            elif key == ord("y"):
                show_yolo = not show_yolo
            elif key == ord("e"):
                active_tool = "eraser"
            elif key == ord("s"):
                # Save clean artwork without the toolbar or PiP box overlaid
                cv2.imwrite("finger_artwork.png", clean_artwork)
                print("Saved clean drawing to finger_artwork.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_finger_drawer(source=0, weights="yolo26n.pt")
