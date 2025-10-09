#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA Highlight Generator

Given a full-game video, this script attempts to detect key scoring moments and
produces a single highlights video composed of short clips around those moments.

Primary method:
- Scoreboard OCR (pytesseract + OpenCV): detect score changes across sampled frames.

Fallback method:
- Audio peak detection (ffmpeg + numpy): find crowd/announcer excitement peaks.

Video segmenting and concatenation are performed via ffmpeg.

Requirements (Python packages):
- opencv-python
- numpy
- pytesseract
- Pillow (pytesseract dependency)

System dependencies:
- ffmpeg (and ffprobe) on PATH
- Tesseract OCR installed (recommended) on Windows, typically at:
  C:\\Program Files\\Tesseract-OCR\\tesseract.exe

Usage example:
  python highlight_nba.py --input "full_game.mp4" --output "highlights.mp4" \
    --pre 6 --post 4 --sample-fps 2 --max-clips 40

Optional: set Tesseract explicitly if not auto-detected
  setx TESSERACT_CMD "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

"""

import argparse
import contextlib
import dataclasses
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:
    print("ERROR: opencv-python is required: pip install opencv-python", file=sys.stderr)
    raise

try:
    import pytesseract  # type: ignore
except Exception as e:
    print("ERROR: pytesseract is required: pip install pytesseract Pillow", file=sys.stderr)
    raise


@dataclasses.dataclass
class ROI:
    name: str
    x: int
    y: int
    w: int
    h: int


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_ffmpeg() -> None:
    if which("ffmpeg") is None or which("ffprobe") is None:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH. Please install ffmpeg and ensure ffmpeg and ffprobe are available."
        )


def detect_tesseract() -> bool:
    # Allow user override via env var
    t_cmd = os.environ.get("TESSERACT_CMD")
    if t_cmd and os.path.isfile(t_cmd):
        pytesseract.pytesseract.tesseract_cmd = t_cmd
    else:
        # Try common Windows install path
        win_default = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.isfile(win_default):
            pytesseract.pytesseract.tesseract_cmd = win_default
        # else rely on PATH
    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def run_ffprobe_duration(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration = float(out.decode("utf-8", errors="ignore").strip())
        return max(0.0, duration)
    except Exception as e:
        raise RuntimeError(f"Failed to read duration via ffprobe: {e}")


def time_points(duration: float, fps: float) -> List[float]:
    if fps <= 0:
        return []
    step = 1.0 / fps
    t = 0.0
    pts = []
    while t < duration:
        pts.append(t)
        t += step
    return pts


def read_frame_at_time(cap: cv2.VideoCapture, t: float) -> Optional[np.ndarray]:
    # Set by milliseconds
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def build_candidate_rois(frame_w: int, frame_h: int) -> List[ROI]:
    # Typical scoreboard placements across broadcasts
    rois: List[ROI] = []
    # Fractions
    def rf(x, y, w, h, name):
        return ROI(name=name, x=int(x * frame_w), y=int(y * frame_h), w=int(w * frame_w), h=int(h * frame_h))

    # Bottom center long bar
    rois.append(rf(0.20, 0.88, 0.60, 0.10, "bottom_center"))
    # Bottom left and right blocks
    rois.append(rf(0.02, 0.90, 0.28, 0.09, "bottom_left"))
    rois.append(rf(0.70, 0.90, 0.28, 0.09, "bottom_right"))
    # Top left and right
    rois.append(rf(0.02, 0.02, 0.28, 0.10, "top_left"))
    rois.append(rf(0.70, 0.02, 0.28, 0.10, "top_right"))
    # Center top strip
    rois.append(rf(0.25, 0.02, 0.50, 0.10, "top_center"))
    return rois


_score_re = re.compile(r"\d+")


def preprocess_roi(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Increase contrast
    gray = cv2.equalizeHist(gray)
    # Adaptive threshold
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 10)
    # Invert if mostly dark
    if th.mean() < 127:
        th = 255 - th
    # Optional morphology to join digits
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def parse_scores(text: str) -> Optional[Tuple[int, int]]:
    # Extract digit groups and try to pick two reasonable scores (0-199)
    nums = [int(x) for x in _score_re.findall(text)]
    # Try to find two consecutive numbers that look like scores
    pairs: List[Tuple[int, int]] = []
    for i in range(len(nums) - 1):
        a, b = nums[i], nums[i + 1]
        if 0 <= a <= 199 and 0 <= b <= 199:
            pairs.append((a, b))
    if not pairs:
        return None
    # Heuristic: choose the pair with minimal absolute diff change from a typical basketball close game (diff <= 40)
    pairs.sort(key=lambda p: abs(p[0] - p[1]))
    return pairs[0]


def ocr_scores(img_roi: np.ndarray) -> Optional[Tuple[int, int]]:
    pre = preprocess_roi(img_roi)
    # Configure Tesseract for numeric OCR
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    try:
        text = pytesseract.image_to_string(pre, config=config)
    except Exception:
        return None
    return parse_scores(text)


def find_best_roi(cap: cv2.VideoCapture, duration: float, sample_fps: float, max_samples: int = 300) -> Optional[ROI]:
    # Read first valid frame to get size
    first = read_frame_at_time(cap, 0.5)
    if first is None:
        return None
    h, w = first.shape[:2]
    rois = build_candidate_rois(w, h)

    times = time_points(duration, sample_fps)
    if max_samples > 0 and len(times) > max_samples:
        # Evenly sample
        idx = np.linspace(0, len(times) - 1, max_samples).astype(int).tolist()
        times = [times[i] for i in idx]

    best_roi = None
    best_score = -1
    for roi in rois:
        last_pair = None
        changes = 0
        reads = 0
        for t in times:
            frame = read_frame_at_time(cap, t)
            if frame is None:
                continue
            x, y, rw, rh = roi.x, roi.y, roi.w, roi.h
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            crop = frame[y : y + rh, x : x + rw]
            pair = ocr_scores(crop)
            if pair is None:
                continue
            reads += 1
            if last_pair is not None:
                if (pair[0] + pair[1]) != (last_pair[0] + last_pair[1]):
                    changes += 1
            last_pair = pair
        # Scoring: prefer many reads and some changes
        score = reads + 2 * changes
        if score > best_score and reads >= 5:
            best_score = score
            best_roi = roi
    return best_roi


def detect_score_change_events(
    cap: cv2.VideoCapture,
    duration: float,
    roi: ROI,
    sample_fps: float,
    min_gap_s: float = 4.0,
) -> List[float]:
    times = time_points(duration, sample_fps)
    last_sum: Optional[int] = None
    last_event_time: float = -1e9
    events: List[float] = []
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    for t in times:
        frame = read_frame_at_time(cap, t)
        if frame is None:
            continue
        x, y, rw, rh = roi.x, roi.y, roi.w, roi.h
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))
        crop = frame[y : y + rh, x : x + rw]
        pair = ocr_scores(crop)
        if pair is None:
            continue
        s = pair[0] + pair[1]
        if last_sum is not None and s != last_sum:
            if (t - last_event_time) >= min_gap_s:
                events.append(t)
                last_event_time = t
        last_sum = s
    return events


def ffmpeg_extract_segment(
    input_path: str,
    start_s: float,
    duration_s: float,
    out_path: str,
    reencode: bool = True,
) -> None:
    # Ensure positive clamp
    start_s = max(0.0, start_s)
    duration_s = max(0.01, duration_s)
    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration_s:.3f}",
            "-analyzeduration",
            "0",
            "-probesize",
            "32M",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            out_path,
        ]
    else:
        # Stream copy (fast but cut points may be off if not on keyframes)
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-i",
            input_path,
            "-t",
            f"{duration_s:.3f}",
            "-c",
            "copy",
            out_path,
        ]
    subprocess.run(cmd, check=True)


def ffmpeg_concat_filelist(filelist_path: str, output_path: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        filelist_path,
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def ffmpeg_overlay_watermark(
    input_path: str,
    watermark_path: str,
    output_path: str,
    margin: int = 20,
    wm_width: int = 96,
) -> None:
    # Build filter to place watermark at bottom-right with margin.
    if wm_width and wm_width > 0:
        filter_expr = f"[1:v]scale={wm_width}:-1[wm];[0:v][wm]overlay=W-w-{margin}:H-h-{margin}:format=auto[v]"
    else:
        filter_expr = f"[0:v][1:v]overlay=W-w-{margin}:H-h-{margin}:format=auto[v]"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-i",
        watermark_path,
        "-filter_complex",
        filter_expr,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def extract_audio_wav(input_path: str, wav_path: str, ar: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(ar),
        "-acodec",
        "pcm_s16le",
        wav_path,
    ]
    subprocess.run(cmd, check=True)


def audio_rms_envelope(wav_path: str, window_ms: int = 200, hop_ms: int = 100) -> Tuple[np.ndarray, np.ndarray, int]:
    with contextlib.closing(wave.open(wav_path, "rb")) as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()  # bytes
        fr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    # Expecting mono, 16-bit
    if n_channels != 1 or sampwidth != 2:
        # Attempt to interpret anyway if mono 16-bit not matched
        pass
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    # Normalize to [-1, 1]
    samples /= 32768.0
    win = max(1, int(fr * (window_ms / 1000.0)))
    hop = max(1, int(fr * (hop_ms / 1000.0)))
    n = len(samples)
    rms = []
    times = []
    i = 0
    while i + win <= n:
        seg = samples[i : i + win]
        rms_val = float(np.sqrt(np.mean(seg * seg) + 1e-9))
        rms.append(rms_val)
        center = i + win // 2
        times.append(center / fr)
        i += hop
    return np.array(times, dtype=np.float32), np.array(rms, dtype=np.float32), fr


def pick_peaks(
    times: np.ndarray,
    rms: np.ndarray,
    threshold_std: float = 1.2,
    min_gap_s: float = 6.0,
    max_peaks: Optional[int] = None,
) -> List[float]:
    if len(rms) == 0:
        return []
    mu = float(np.mean(rms))
    sigma = float(np.std(rms) + 1e-6)
    thr = mu + threshold_std * sigma
    # Simple local maxima detection
    peaks: List[int] = []
    for i in range(1, len(rms) - 1):
        if rms[i] > thr and rms[i] > rms[i - 1] and rms[i] >= rms[i + 1]:
            peaks.append(i)
    # Enforce min gap
    sel: List[int] = []
    last_t = -1e9
    for idx in sorted(peaks, key=lambda i: rms[i], reverse=True):
        t = float(times[idx])
        if (t - last_t) >= min_gap_s:
            sel.append(idx)
            last_t = t
        if max_peaks and len(sel) >= max_peaks:
            break
    sel_sorted = sorted(sel)
    return [float(times[i]) for i in sel_sorted]


def clamp_segments(events: List[float], duration: float, pre_s: float, post_s: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    for t in events:
        start = max(0.0, t - pre_s)
        end = min(duration, t + post_s)
        segs.append((start, max(0.01, end - start)))
    return segs


def dedupe_close_events(events: List[float], min_gap_s: float) -> List[float]:
    if not events:
        return []
    events_sorted = sorted(events)
    out = [events_sorted[0]]
    for t in events_sorted[1:]:
        if (t - out[-1]) >= min_gap_s:
            out.append(t)
    return out


def main():
    parser = argparse.ArgumentParser(description="NBA Highlight Generator: find scoring moments and cut highlights.")
    parser.add_argument("--input", required=True, help="Path to full game video file")
    parser.add_argument("--output", default="highlights.mp4", help="Output highlights video path")
    parser.add_argument("--pre", type=float, default=6.0, help="Seconds before event to include in each clip")
    parser.add_argument("--post", type=float, default=4.0, help="Seconds after event to include in each clip")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="FPS to sample frames for OCR")
    parser.add_argument("--min-gap", type=float, default=6.0, help="Minimum seconds between detected events")
    parser.add_argument("--max-clips", type=int, default=60, help="Max number of clips in the final highlight (0 = no limit)")
    parser.add_argument("--reencode", action="store_true", help="Re-encode segments (more accurate cuts, recommended)")
    parser.add_argument("--copy", dest="reencode", action="store_false", help="Fast stream copy (cuts may be imprecise)")
    parser.add_argument("--watermark", default="fb-watermark.png", help="Path to PNG watermark image (set empty string to disable)")
    parser.add_argument("--wm-width", type=int, default=96, help="Watermark width in pixels (0 = original size)")
    parser.add_argument("--wm-margin", type=int, default=20, help="Margin in pixels from bottom-right corner")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        logging.error("Input file not found: %s", input_path)
        sys.exit(1)

    try:
        ensure_ffmpeg()
    except Exception as e:
        logging.error(str(e))
        sys.exit(1)

    duration = run_ffprobe_duration(input_path)
    logging.info("Video duration: %.1f s", duration)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logging.error("Failed to open video with OpenCV: %s", input_path)
        sys.exit(1)

    have_tess = detect_tesseract()
    if have_tess:
        logging.info("Tesseract detected. Attempting scoreboard OCR for score change events...")
    else:
        logging.warning("Tesseract not detected. Will fallback to audio-based peak detection.")

    events: List[float] = []
    roi_used: Optional[ROI] = None

    if have_tess:
        try:
            roi = find_best_roi(cap, duration, sample_fps=args.sample_fps)
            if roi is not None:
                roi_used = roi
                logging.info("Selected ROI: %s at x=%d y=%d w=%d h=%d", roi.name, roi.x, roi.y, roi.w, roi.h)
                events = detect_score_change_events(cap, duration, roi, sample_fps=args.sample_fps, min_gap_s=max(2.0, args.min_gap / 2.0))
                logging.info("OCR detected %d score-change events before dedupe.", len(events))
            else:
                logging.warning("Failed to determine a reliable scoreboard ROI.")
        except Exception as e:
            logging.warning("OCR path failed: %s", e)

    # Fallback or supplement with audio peaks if OCR yielded nothing
    if not events:
        logging.info("Falling back to audio peak detection...")
        with tempfile.TemporaryDirectory(prefix="nba_audio_") as td:
            wav_path = os.path.join(td, "audio.wav")
            try:
                extract_audio_wav(input_path, wav_path, ar=16000)
                t_arr, rms_arr, _ = audio_rms_envelope(wav_path, window_ms=250, hop_ms=125)
                events = pick_peaks(t_arr, rms_arr, threshold_std=1.2, min_gap_s=args.min_gap, max_peaks=args.max_clips if args.max_clips > 0 else None)
                logging.info("Audio peaks detected %d events.", len(events))
            except Exception as e:
                logging.error("Audio-based detection failed: %s", e)
                sys.exit(1)

    # Dedupe close events
    events = dedupe_close_events(events, args.min_gap)

    if args.max_clips > 0 and len(events) > args.max_clips:
        logging.info("Limiting events from %d to top %d by temporal spread.", len(events), args.max_clips)
        # Spread selection: take evenly spaced indices
        idx = np.linspace(0, len(events) - 1, args.max_clips).astype(int).tolist()
        events = [events[i] for i in idx]

    logging.info("Using %d events to build highlights.", len(events))
    if not events:
        logging.error("No events detected. Cannot create highlights.")
        sys.exit(2)

    segments = clamp_segments(events, duration, args.pre, args.post)

    with tempfile.TemporaryDirectory(prefix="nba_highlight_") as td:
        seg_paths: List[str] = []
        for i, (start, dur) in enumerate(segments):
            out_seg = os.path.join(td, f"seg_{i:04d}.mp4")
            try:
                ffmpeg_extract_segment(input_path, start, dur, out_seg, reencode=args.reencode)
                seg_paths.append(out_seg)
            except subprocess.CalledProcessError as e:
                logging.warning("Failed to cut segment %d at %.2fs (%.2fs): %s", i, start, dur, e)
                continue

        if not seg_paths:
            logging.error("No segments were successfully cut.")
            sys.exit(3)

        # Write concat list
        list_path = os.path.join(td, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in seg_paths:
                # Use forward slashes or escape backslashes
                fp = p.replace("\\", "/")
                f.write(f"file '{fp}'\n")

        # First, concatenate the segments into a temporary file
        concat_tmp = os.path.join(td, "concat_tmp.mp4")
        try:
            ffmpeg_concat_filelist(list_path, concat_tmp)
        except subprocess.CalledProcessError:
            logging.warning("Concat with stream copy failed. Re-encoding concat as fallback...")
            # Fallback: re-encode concat using filter_complex
            cmd = ["ffmpeg", "-y"]
            for p in seg_paths:
                cmd.extend(["-i", p])
            n = len(seg_paths)
            streams = []
            for i in range(n):
                streams.append(f"[{i}:v][{i}:a]")
            filter_concat = "".join(streams) + f"concat=n={n}:v=1:a=1[v][a]"
            cmd.extend([
                "-filter_complex",
                filter_concat,
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                concat_tmp,
            ])
            subprocess.run(cmd, check=True)

        # Apply watermark if available, else move concat to final output
        wm_path = os.path.abspath(args.watermark).strip() if isinstance(args.watermark, str) else ""
        apply_wm = len(wm_path) > 0 and os.path.isfile(wm_path)
        if apply_wm:
            logging.info("Applying watermark: %s", wm_path)
            try:
                ffmpeg_overlay_watermark(concat_tmp, wm_path, output_path, margin=args.wm_margin, wm_width=args.wm_width)
            except subprocess.CalledProcessError as e:
                logging.error("Watermark overlay failed: %s", e)
                sys.exit(4)
        else:
            if len(wm_path) > 0:
                logging.warning("Watermark file not found at %s. Proceeding without watermark.", wm_path)
            shutil.move(concat_tmp, output_path)

    logging.info("Highlights written to: %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
