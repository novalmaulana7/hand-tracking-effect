"""
Retro Lens - Real-time Hand Gesture Filter Pipeline
"""

from dataclasses import dataclass
import os
import random
import time
from typing import Dict, List, Tuple, Callable

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")

# Filter yang mahal di resolusi penuh; diproses di resolusi lebih kecil lalu di-upscale.
HEAVY_FILTERS = {"cartoon", "sketch", "rainbow_wave"}


@dataclass
class PipelineConfig:
    cam_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    detection_max_width: int = 640
    filter_max_dim: int = 640
    pinch_threshold_px: float = 45.0
    filter_cooldown_sec: float = 0.15
    mode_cooldown_sec: float = 1.2
    fist_dist_threshold_px: float = 80.0


class FilterBank:
    @staticmethod
    def dual_tone(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        out = np.zeros_like(roi)
        out[mask == 255] = (10, 140, 255)
        out[mask == 0] = (180, 30, 220)
        return out

    @staticmethod
    def thermal(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    @staticmethod
    def sketch(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def pixelate(roi: np.ndarray, block_size: int = 14) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def glitch(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        if h < 2 or w < 2:
            return roi
        b, g, r = cv2.split(roi)
        shift = random.randint(4, 12)
        r = np.roll(r, shift, axis=1)
        b = np.roll(b, -shift, axis=1)
        out = cv2.merge([b, g, r])
        for _ in range(2):
            y = random.randint(0, h - 1)
            out[y : y + 1, :] = np.random.randint(0, 255, (1, w, 3), dtype=np.uint8)
        return out

    @staticmethod
    def invert(roi: np.ndarray) -> np.ndarray:
        return 255 - roi

    @staticmethod
    def red_channel(roi: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(roi)
        zeros = np.zeros_like(b)
        return cv2.merge([zeros, zeros, r])

    @staticmethod
    def edge(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        colored = cv2.applyColorMap(edges, cv2.COLORMAP_SUMMER)
        return cv2.bitwise_and(colored, colored, mask=edges)

    @staticmethod
    def blur(roi: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(roi, (25, 25), 0)

    @staticmethod
    def cartoon(roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    @staticmethod
    def rainbow_wave(roi: np.ndarray) -> np.ndarray:
        h, w = roi.shape[:2]
        t = time.time() * 5.0
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
        pattern = np.sin((x_coords + y_coords) * 0.05 + t) * 127 + 128
        rainbow = cv2.applyColorMap(pattern.astype(np.uint8), cv2.COLORMAP_HSV)
        return cv2.addWeighted(roi, 0.3, rainbow, 0.7, 0)


class GeometryUtils:
    @staticmethod
    def euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def is_fist_closed(landmarks, w: int, h: int, threshold: float) -> bool:
        wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
        tips = [8, 12, 16, 20]
        distances = [np.linalg.norm(np.array([landmarks[t].x * w, landmarks[t].y * h]) - wrist) for t in tips]
        return float(np.mean(distances)) < threshold

    @staticmethod
    def is_hand_rotated(thumb: Tuple[int, int], index: Tuple[int, int]) -> bool:
        dx, dy = index[0] - thumb[0], index[1] - thumb[1]
        return (dy > 25) or (abs(dx) > abs(dy) * 1.1)

    @staticmethod
    def sort_quad_clean(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[0], rightmost[1], leftmost[1]], dtype=np.int32)

    @staticmethod
    def sort_quad_bowtie(pts: List[Tuple[int, int]]) -> np.ndarray:
        arr = np.array(pts, dtype=np.float32)
        x_sorted = arr[np.argsort(arr[:, 0]), :]
        leftmost = x_sorted[:2, :][np.argsort(x_sorted[:2, 1]), :]
        rightmost = x_sorted[2:, :][np.argsort(x_sorted[2:, 1]), :]
        return np.array([leftmost[0], rightmost[1], rightmost[0], leftmost[1]], dtype=np.int32)


class PortalProcessor:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.filters: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
            "dual-tone": FilterBank.dual_tone,
            "thermal": FilterBank.thermal,
            "sketch": FilterBank.sketch,
            "pixelate": FilterBank.pixelate,
            "glitch": FilterBank.glitch,
            "invert": FilterBank.invert,
            "red-channel": FilterBank.red_channel,
            "edge": FilterBank.edge,
            "blur": FilterBank.blur,
            "cartoon": FilterBank.cartoon,
            "rainbow-wave": FilterBank.rainbow_wave,
        }
        self.filter_keys = list(self.filters.keys())
        self.active_filter_idx = 0
        self.is_3d_mode = False
        
        self.last_switch_time = 0.0
        self.last_mode_toggle = 0.0

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"[ERROR] Model tidak ditemukan: {MODEL_PATH}\n"
                "Download dulu: models/hand_landmarker.task (lihat README)."
            )
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(options)
        self._last_ts_ms = -1
        self._fps = 0.0
        self._prev_frame_time = 0.0
        self._detect_buf = None
        self._canvas = None

    @property
    def current_filter_name(self) -> str:
        return self.filter_keys[self.active_filter_idx]

    @property
    def secondary_filter_name(self) -> str:
        return self.filter_keys[(self.active_filter_idx + 1) % len(self.filter_keys)]

    def cycle_filter(self, step: int = 1) -> None:
        self.active_filter_idx = (self.active_filter_idx + step) % len(self.filter_keys)

    def render_portal(self, frame: np.ndarray, pts: List[Tuple[int, int]], filter_key: str) -> np.ndarray:
        poly = np.array(pts, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)

        if w <= 10 or h <= 10:
            return frame

        roi = frame[y : y + h, x : x + w].copy()
        if filter_key in HEAVY_FILTERS and self.cfg.filter_max_dim and max(w, h) > self.cfg.filter_max_dim:
            scale = self.cfg.filter_max_dim / max(w, h)
            small_w = max(1, int(round(w * scale)))
            small_h = max(1, int(round(h * scale)))
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            processed_roi = self.filters[filter_key](small)
            processed_roi = cv2.resize(processed_roi, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            processed_roi = self.filters[filter_key](roi)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly - [x, y]], 255)
        mask_3c = cv2.merge([mask, mask, mask])

        bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask_3c))
        fg = cv2.bitwise_and(processed_roi, mask_3c)
        frame[y : y + h, x : x + w] = cv2.add(bg, fg)

        cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
        return frame

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        target_w, target_h = self.cfg.frame_width, self.cfg.frame_height
        h, w = frame.shape[:2]
        if (w, h) == (target_w, target_h):
            return frame

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        if (
            self._canvas is None
            or self._canvas.shape != (target_h, target_w, frame.shape[2])
            or self._canvas.dtype != frame.dtype
        ):
            self._canvas = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
        else:
            self._canvas.fill(0)

        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        self._canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
        return self._canvas

    def _detection_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w = min(w, self.cfg.detection_max_width)
        if target_w >= w:
            return frame

        target_h = max(1, int(round(h * target_w / w)))
        if self._detect_buf is None or self._detect_buf.shape[:2] != (target_h, target_w):
            self._detect_buf = np.empty((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
        return cv2.resize(frame, (target_w, target_h), dst=self._detect_buf, interpolation=cv2.INTER_AREA)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        frame = self._fit_frame(frame)
        detect_frame = self._detection_frame(frame)
        rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        now = time.time()

        if self._prev_frame_time:
            instant_fps = 1.0 / max(now - self._prev_frame_time, 1e-6)
            self._fps = instant_fps if self._fps == 0.0 else (self._fps * 0.9 + instant_fps * 0.1)
        self._prev_frame_time = now

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(int(now * 1000), self._last_ts_ms + 1)
        self._last_ts_ms = ts_ms
        results = self.detector.detect_for_video(mp_image, ts_ms)

        all_hand_tips = []
        fist_count = 0
        is_bowtie = False

        if results.hand_landmarks:
            for lm in results.hand_landmarks:
                self._draw_hand(frame, lm)

                tips = [(int(lm[i].x * self.cfg.frame_width), int(lm[i].y * self.cfg.frame_height)) for i in [4, 8, 12, 16, 20]]
                all_hand_tips.append(tips)

                # Fast Pinch
                if GeometryUtils.euclidean_dist(tips[0], tips[4]) < self.cfg.pinch_threshold_px:
                    if now - self.last_switch_time > self.cfg.filter_cooldown_sec:
                        self.cycle_filter(1)
                        self.last_switch_time = now

                if GeometryUtils.is_fist_closed(lm, self.cfg.frame_width, self.cfg.frame_height, self.cfg.fist_dist_threshold_px):
                    fist_count += 1

            # Dual Fist Mode Switch
            if fist_count == 2 and (now - self.last_mode_toggle > self.cfg.mode_cooldown_sec):
                self.is_3d_mode = not self.is_3d_mode
                self.last_mode_toggle = now

            if self.is_3d_mode:
                if len(all_hand_tips) == 2:
                    t1, t2 = all_hand_tips[0], all_hand_tips[1]
                    frame = self.render_portal(frame, [t1[0], t1[1], t1[2], t2[2], t2[1], t2[0]], self.current_filter_name)
                    frame = self.render_portal(frame, [t1[2], t1[3], t1[4], t2[4], t2[3], t2[2]], self.secondary_filter_name)
                elif len(all_hand_tips) == 1:
                    frame = self.render_portal(frame, all_hand_tips[0], self.current_filter_name)
            else:
                if len(all_hand_tips) == 2:
                    corners = [all_hand_tips[0][0], all_hand_tips[0][1], all_hand_tips[1][0], all_hand_tips[1][1]]
                    if GeometryUtils.is_hand_rotated(corners[0], corners[1]) or GeometryUtils.is_hand_rotated(corners[2], corners[3]):
                        quad = GeometryUtils.sort_quad_bowtie(corners)
                        is_bowtie = True
                    else:
                        quad = GeometryUtils.sort_quad_clean(corners)
                    frame = self.render_portal(frame, quad, self.current_filter_name)
                elif len(all_hand_tips) == 1:
                    t = all_hand_tips[0]
                    frame = self.render_portal(frame, [t[0], t[1], t[2], t[4]], self.current_filter_name)

        self._draw_hud(frame, is_bowtie)
        return frame

    def _draw_hand(self, frame: np.ndarray, lm) -> None:
        w, h = self.cfg.frame_width, self.cfg.frame_height
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 255, 0), -1)

    def _draw_hud(self, frame: np.ndarray, is_bowtie: bool) -> None:
        mode_str = "3D Mesh" if self.is_3d_mode else ("2D Bowtie" if is_bowtie else "2D Quad")
        cv2.putText(frame, f"MODE: {mode_str} [Key 'C' / Dual Fist]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f"FILTER: {self.current_filter_name.upper()} [Pinch / Key 'N'/'P']", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        fps_text = f"FPS: {self._fps:4.1f}"
        (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        x = max(15, frame.shape[1] - tw - 15)
        cv2.putText(frame, fps_text, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)


def main() -> None:
    cfg = PipelineConfig()
    processor = PortalProcessor(cfg)
    cap = cv2.VideoCapture(cfg.cam_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi!")
        return

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Kamera: {cam_w}x{cam_h} @ {cam_fps:.0f} FPS -> output {cfg.frame_width}x{cfg.frame_height} (aspect ratio dijaga)")
    if cam_fps and cam_fps < 25:
        print("[WARN] Kamera melaporkan FPS rendah (<25). Coba mode MJPG / turunkan resolusi kamera / perbaiki pencahayaan.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame kamera.")
            break

        out_frame = processor.process_frame(frame)
        cv2.imshow("RetroLens Engine", out_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            processor.is_3d_mode = not processor.is_3d_mode
        elif key == ord("n"):
            processor.cycle_filter(1)
        elif key == ord("p"):
            processor.cycle_filter(-1)
        elif key == ord("s"):
            cv2.imwrite(f"cap_{int(time.time())}.png", out_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()