# FingerCanvas

A real-time computer vision air-drawing application that turns your index fingertip into a precision digital stylus. Built with **PyTorch**, **Ultralytics YOLO26**, **MediaPipe HandLandmarker**, and **OpenCV**, optimized for Apple Silicon (`MPS`), CUDA, and CPU.

## Features

- **Adaptive Precision Tracking:** Tracks Index Fingertip (Landmark 8) via MediaPipe's `HandLandmarker` Tasks API, stabilized with a velocity-adaptive **One Euro Filter** to eliminate hand tremor during fine detail work while keeping latency near zero on fast strokes.
- **Anti-Aliased Bezier Ink:** Interpolates quadratic Bezier curves across stroke midpoints with alpha-blended anti-aliasing (`cv2.LINE_AA`) for smooth, natural curves.
- **Touch-Friendly Virtual Toolbar:** Tap on-screen buttons directly with your fingertip to switch between **Black, Red, Blue, and Green** ink, **Eraser**, **Brush Sizes (`3 / 6 / 12 / 20 px`)**, **Undo** (20-step history), **Clear**, and **Paper / Camera** modes.
- **White Paper & AR Camera Modes:** Sketch on a clean off-white digital paper sheet with a live Picture-in-Picture (PiP) webcam preview, or toggle directly onto the live camera feed with real-time **YOLO26** detection.
- **HD Camera Enhancement Pipeline:** Negotiates 1080p capture via macOS AVFoundation with real-time LAB-space CLAHE contrast balancing and unsharp-mask sharpening.
- **Clean Artwork Export:** Save full-resolution transparent-composited PNGs (`finger_artwork.png`) without UI toolbars or webcam previews overlaid.

## Controls

### Hand Gestures
| Gesture | Action |
| :--- | :--- |
| **Index finger up** (middle finger down) | **Draw / Erase** on canvas or tap top toolbar buttons |
| **Index + Middle fingers up** (peace sign) | **Hover / Lift Pen** with precision crosshair reticle |

### Keyboard Shortcuts
| Key | Action |
| :--- | :--- |
| `P` | Toggle between **White Paper** and **Live Camera** background |
| `E` | Switch to **Eraser** |
| `Z` | **Undo** last stroke |
| `C` | **Clear** canvas |
| `H` | Toggle **HD Contrast & Sharpening** enhancement |
| `Y` | Toggle **YOLO26** bounding box overlay |
| `S` | Save clean artwork to `finger_artwork.png` |
| `Q` | Quit application |

## Quickstart

1. **Clone the repository**
   ```bash
   git clone [https://github.com/bhr007/FingerCanvas.git](https://github.com/bhr007/FingerCanvas.git)
   cd FingerCanvas
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install torch torchvision ultralytics mediapipe opencv-python
   ```

3. **Run the finger canvas**
   ```bash
   python3 finger_draw.py
   ```
   *(On first launch, `yolo26n.pt` and `hand_landmarker.task` download automatically.)*

## Project Structure

```text
FingerCanvas/
├── finger_draw.py      # Main HD finger-tracking sketchpad + YOLO26 detector
├── yolo_realtime.py    # Standalone real-time YOLO26 webcam object detector
├── requirements.txt    # Python dependencies
└── .gitignore          # Excludes virtual environments, weights, and saved images
```