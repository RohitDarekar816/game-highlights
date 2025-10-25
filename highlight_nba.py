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

Fighting scene detection:
- Motion analysis: detect high-intensity movement and optical flow patterns.
- Visual intensity: analyze frame differences and brightness changes.
- Audio analysis: detect combat sounds and audio spikes.

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
    --pre 6 --post 4 --sample-fps 2 --max-clips 40 --pitch 1.5

Fighting scene detection:
  python highlight_nba.py --input "game_footage.mp4" --output "fighting_highlights.mp4" \
    --fighting-mode --motion-threshold 20 --fighting-sample-fps 2 --max-clips 30

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
    pitch_factor: float = 1.0,
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
        ]
        
        # Add audio processing with pitch adjustment if needed
        if pitch_factor != 1.0:
            # Use atempo filter for pitch adjustment (range: 0.5 to 2.0)
            # For values outside this range, we need to chain multiple atempo filters
            audio_filter = ""
            temp_pitch = pitch_factor
            while temp_pitch > 2.0:
                audio_filter += "atempo=2.0,"
                temp_pitch /= 2.0
            while temp_pitch < 0.5:
                audio_filter += "atempo=0.5,"
                temp_pitch /= 0.5
            if temp_pitch != 1.0:
                audio_filter += f"atempo={temp_pitch:.3f}"
            if audio_filter.endswith(","):
                audio_filter = audio_filter[:-1]
            
            cmd.extend([
                "-af", audio_filter,
                "-c:a", "aac",
                "-b:a", "160k",
            ])
        else:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "160k",
            ])
        
        cmd.extend([
            "-movflags",
            "+faststart",
            out_path,
        ])
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
    pitch_factor: float = 1.0,
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
    ]
    
    # Add audio processing with pitch adjustment if needed
    if pitch_factor != 1.0:
        # Use atempo filter for pitch adjustment (range: 0.5 to 2.0)
        # For values outside this range, we need to chain multiple atempo filters
        audio_filter = ""
        temp_pitch = pitch_factor
        while temp_pitch > 2.0:
            audio_filter += "atempo=2.0,"
            temp_pitch /= 2.0
        while temp_pitch < 0.5:
            audio_filter += "atempo=0.5,"
            temp_pitch /= 0.5
        if temp_pitch != 1.0:
            audio_filter += f"atempo={temp_pitch:.3f}"
        if audio_filter.endswith(","):
            audio_filter = audio_filter[:-1]
        
        cmd.extend([
            "-af", audio_filter,
            "-c:a", "aac",
            "-b:a", "160k",
        ])
    else:
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "160k",
        ])
    
    cmd.extend([
        "-movflags",
        "+faststart",
        output_path,
    ])
    subprocess.run(cmd, check=True)


def ffmpeg_finalize_clip(
    input_path: str,
    output_path: str,
    watermark_path: str = "",
    wm_margin: int = 20,
    wm_width: int = 96,
    pitch_factor: float = 1.0,
    bgm_path: str = "",
    video_volume: float = 1.0,
    bgm_volume: float = 0.25,
) -> None:
    """
    Finalize a clip by optionally overlaying a watermark and mixing background music.

    - If watermark_path is a valid file, overlay it on video.
    - If bgm_path is a valid file, mix it with the clip's audio at specified volumes.
    - Applies audio pitch/tempo adjustment if pitch_factor != 1.0.
    """
    inputs = ["-i", input_path]
    map_video_label = "[0:v]"
    filter_parts: List[str] = []
    audio_inputs = []  # labels that will be mixed

    # Watermark input (index 1)
    apply_wm = watermark_path and os.path.isfile(watermark_path)
    if apply_wm:
        inputs += ["-i", watermark_path]
        if wm_width and wm_width > 0:
            # Scale watermark then overlay
            filter_parts.append(f"[1:v]scale={wm_width}:-1[wm]")
            filter_parts.append(f"[0:v][wm]overlay=W-w-{wm_margin}:H-h-{wm_margin}:format=auto[v]")
        else:
            filter_parts.append(f"[0:v][1:v]overlay=W-w-{wm_margin}:H-h-{wm_margin}:format=auto[v]")
        map_video_label = "[v]"

    # Background music input (after watermark if present)
    apply_bgm = bgm_path and os.path.isfile(bgm_path)
    if apply_bgm:
        # Loop BGM to cover the clip duration; will be trimmed by -shortest or amix duration
        inputs = ["-stream_loop", "-1", *inputs]  # loop applies to the next input
        inputs += ["-i", bgm_path]

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
    ]

    # Audio chain
    # Start with original audio if present
    audio_filter_chain = []
    # Pitch/tempo adjustment for original audio only if requested
    audio_atempo_chain = []
    if pitch_factor != 1.0:
        temp_pitch = pitch_factor
        while temp_pitch > 2.0:
            audio_atempo_chain.append("atempo=2.0")
            temp_pitch /= 2.0
        while temp_pitch < 0.5:
            audio_atempo_chain.append("atempo=0.5")
            temp_pitch /= 0.5
        if temp_pitch != 1.0:
            audio_atempo_chain.append(f"atempo={temp_pitch:.3f}")

    # Original audio label
    # We will guard mapping with optional map later; for filter graph, use 0:a if exists
    # Apply pitch and volume
    if audio_atempo_chain:
        audio_filter_chain.append(f"[0:a]{','.join(audio_atempo_chain)}[a0t]")
        audio_filter_chain.append(f"[a0t]volume={max(0.0, float(video_volume)):.3f}[a0]")
    else:
        audio_filter_chain.append(f"[0:a]volume={max(0.0, float(video_volume)):.3f}[a0]")
    audio_inputs.append("[a0]")

    # BGM chain (index depends on whether watermark was added). Compute its stream index:
    # input indices: 0 = clip, 1 = wm (if any), 2 = bgm (if wm present) or 1 = bgm (if no wm)
    if apply_bgm:
        bgm_input_index = 2 if apply_wm else 1
        audio_filter_chain.append(f"[{bgm_input_index}:a]volume={max(0.0, float(bgm_volume)):.3f}[abgm]")
        audio_inputs.append("[abgm]")

    # Build combined filter_complex
    fc_parts = []
    fc_parts.extend(filter_parts)
    fc_parts.extend(audio_filter_chain)

    # Mix or select audio
    if len(audio_inputs) == 2:
        # Use video length as duration so mix stops with the clip
        fc_parts.append(f"{''.join(audio_inputs)}amix=inputs=2:duration=first:dropout_transition=2[a]")
    elif len(audio_inputs) == 1:
        fc_parts.append(f"{audio_inputs[0]}anull[a]")
    # else: no audio available; we'll map none

    cmd.append(";".join(fc_parts))

    # Mapping and codecs
    cmd += [
        "-map",
        map_video_label,
    ]
    if len(audio_inputs) >= 1:
        cmd += ["-map", "[a]"]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
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


def calculate_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate the magnitude of optical flow between two frames using Farneback method."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    try:
        # Use Farneback optical flow which doesn't require feature points
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate magnitude of flow vectors
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return float(np.mean(magnitude))
    except Exception:
        # Fallback to simple frame difference if optical flow fails
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff))


def calculate_frame_difference_intensity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate the intensity of differences between two frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff))


def calculate_motion_energy(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate motion energy using frame differencing and optical flow."""
    # Frame difference intensity
    diff_intensity = calculate_frame_difference_intensity(frame1, frame2)
    
    # Optical flow magnitude
    flow_magnitude = calculate_optical_flow_magnitude(frame1, frame2)
    
    # Combine both metrics (weighted average)
    motion_energy = 0.7 * diff_intensity + 0.3 * flow_magnitude
    return motion_energy


def detect_fighting_scenes(
    cap: cv2.VideoCapture,
    duration: float,
    sample_fps: float = 1.0,
    motion_threshold: float = 15.0,
    min_gap_s: float = 3.0,
    max_clips: Optional[int] = None,
) -> List[float]:
    """
    Detect fighting scenes using motion analysis and visual intensity.
    Optimized for large videos with efficient sampling and early termination.
    
    Args:
        cap: Video capture object
        duration: Video duration in seconds
        sample_fps: Frames per second to sample
        motion_threshold: Minimum motion energy to consider as fighting
        min_gap_s: Minimum gap between detected fighting scenes
        max_clips: Maximum number of fighting scenes to detect
    
    Returns:
        List of timestamps where fighting scenes are detected
    """
    # For very long videos, use more aggressive sampling
    if duration > 3600:  # > 1 hour
        sample_fps = min(sample_fps, 0.5)  # Max 0.5 FPS for long videos
        logging.info("Large video detected (%.1f s), using reduced sampling rate: %.1f FPS", duration, sample_fps)
    
    times = time_points(duration, sample_fps)
    if len(times) < 2:
        return []
    
    # Limit processing for very long videos
    max_samples = 2000 if duration > 3600 else 5000
    if len(times) > max_samples:
        # Sample evenly across the video
        indices = np.linspace(0, len(times) - 1, max_samples).astype(int)
        times = [times[i] for i in indices]
        logging.info("Reduced sampling to %d frames for efficiency", len(times))
    
    motion_energies = []
    valid_times = []
    
    # Calculate motion energy for each frame with progress logging
    prev_frame = None
    total_frames = len(times)
    
    for i, t in enumerate(times):
        if i % 100 == 0 and total_frames > 200:
            logging.info("Processing frame %d/%d (%.1f%%)", i, total_frames, 100.0 * i / total_frames)
        
        frame = read_frame_at_time(cap, t)
        if frame is None:
            continue
            
        if prev_frame is not None:
            # Use simplified motion detection for efficiency
            motion_energy = calculate_frame_difference_intensity(prev_frame, frame)
            motion_energies.append(motion_energy)
            valid_times.append(t)
        
        prev_frame = frame
        
        # Early termination if we have enough high-energy scenes
        if max_clips and len(motion_energies) > max_clips * 3:
            # Check if we have enough high-energy scenes to stop early
            if len(motion_energies) > 0:
                current_threshold = np.percentile(motion_energies, 80)
                high_energy_count = sum(1 for e in motion_energies if e > current_threshold)
                if high_energy_count >= max_clips:
                    logging.info("Early termination: found %d high-energy scenes", high_energy_count)
                    break
    
    if len(motion_energies) == 0:
        return []
    
    # Convert to numpy arrays for easier processing
    motion_array = np.array(motion_energies)
    time_array = np.array(valid_times)
    
    # Calculate threshold based on statistics
    mean_motion = np.mean(motion_array)
    std_motion = np.std(motion_array)
    threshold = max(motion_threshold, mean_motion + 1.5 * std_motion)
    
    logging.info("Motion analysis: mean=%.2f, std=%.2f, threshold=%.2f", mean_motion, std_motion, threshold)
    
    # Find peaks in motion energy
    fighting_scenes = []
    for i in range(1, len(motion_array) - 1):
        if (motion_array[i] > threshold and 
            motion_array[i] > motion_array[i - 1] and 
            motion_array[i] >= motion_array[i + 1]):
            fighting_scenes.append(time_array[i])
    
    logging.info("Found %d potential fighting scenes before filtering", len(fighting_scenes))
    
    # Apply minimum gap filtering
    if min_gap_s > 0 and fighting_scenes:
        filtered_scenes = [fighting_scenes[0]]
        for scene_time in fighting_scenes[1:]:
            if scene_time - filtered_scenes[-1] >= min_gap_s:
                filtered_scenes.append(scene_time)
        fighting_scenes = filtered_scenes
    
    # Limit number of scenes if specified
    if max_clips and len(fighting_scenes) > max_clips:
        # Sort by motion energy and take top scenes
        scene_energies = []
        for scene_time in fighting_scenes:
            # Find closest time index
            closest_idx = np.argmin(np.abs(time_array - scene_time))
            scene_energies.append((scene_time, motion_array[closest_idx]))
        
        # Sort by energy and take top scenes
        scene_energies.sort(key=lambda x: x[1], reverse=True)
        fighting_scenes = [scene[0] for scene in scene_energies[:max_clips]]
        fighting_scenes.sort()
    
    logging.info("Final result: %d fighting scenes detected", len(fighting_scenes))
    return fighting_scenes


def detect_combat_audio_peaks(
    wav_path: str,
    threshold_std: float = 2.0,
    min_gap_s: float = 2.0,
    max_peaks: Optional[int] = None,
) -> List[float]:
    """
    Detect combat audio peaks using more aggressive thresholds than regular audio.
    Combat sounds typically have higher intensity and different frequency characteristics.
    """
    try:
        t_arr, rms_arr, _ = audio_rms_envelope(wav_path, window_ms=100, hop_ms=50)
    except Exception:
        return []
    
    if len(rms_arr) == 0:
        return []
    
    # Use higher threshold for combat detection
    mu = float(np.mean(rms_arr))
    sigma = float(np.std(rms_arr) + 1e-6)
    thr = mu + threshold_std * sigma
    
    # Find peaks
    peaks = []
    for i in range(1, len(rms_arr) - 1):
        if rms_arr[i] > thr and rms_arr[i] > rms_arr[i - 1] and rms_arr[i] >= rms_arr[i + 1]:
            peaks.append(i)
    
    # Apply minimum gap
    sel = []
    last_t = -1e9
    for idx in sorted(peaks, key=lambda i: rms_arr[i], reverse=True):
        t = float(t_arr[idx])
        if (t - last_t) >= min_gap_s:
            sel.append(idx)
            last_t = t
        if max_peaks and len(sel) >= max_peaks:
            break
    
    sel_sorted = sorted(sel)
    return [float(t_arr[i]) for i in sel_sorted]


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
    parser.add_argument("--clips-dir", default="", help="Directory to save individual highlight clips (default: <output_basename>_clips beside output)")
    parser.add_argument("--no-combined", action="store_true", help="Only export individual clips, skip combined highlights video")
    parser.add_argument("--pitch", type=float, default=1.0, help="Audio pitch adjustment factor (1.0 = original, 1.5 = 1.5x speed/pitch, 0.5 = half speed/pitch)")
    parser.add_argument("--bgm", default="", help="Path to background music audio file (mp3/wav/etc). Empty to disable")
    parser.add_argument("--bgm-volume", type=float, default=0.25, help="Background music volume multiplier (e.g., 0.25)")
    parser.add_argument("--video-volume", type=float, default=1.0, help="Original video audio volume multiplier (e.g., 0.8)")
    parser.add_argument("--fighting-mode", action="store_true", help="Enable fighting scene detection mode (overrides scoreboard OCR)")
    parser.add_argument("--motion-threshold", type=float, default=15.0, help="Motion energy threshold for fighting scene detection")
    parser.add_argument("--fighting-sample-fps", type=float, default=1.0, help="FPS to sample frames for fighting scene detection")
    parser.add_argument("--combat-audio-threshold", type=float, default=2.0, help="Audio threshold multiplier for combat sound detection")
    parser.add_argument("--max-duration", type=float, default=0, help="Maximum video duration to process in seconds (0 = no limit)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Validate pitch parameter
    if args.pitch <= 0:
        logging.error("Pitch factor must be positive (got %.3f)", args.pitch)
        sys.exit(1)
    if args.pitch < 0.1 or args.pitch > 10.0:
        logging.warning("Pitch factor %.3f is outside recommended range (0.1-10.0)", args.pitch)

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
    
    # Apply duration limit if specified
    if args.max_duration > 0 and duration > args.max_duration:
        logging.info("Limiting processing to first %.1f seconds (%.1f%% of video)", args.max_duration, 100.0 * args.max_duration / duration)
        duration = args.max_duration
    
    if args.pitch != 1.0:
        logging.info("Audio pitch adjustment: %.3fx (tempo will be adjusted)", args.pitch)
    if isinstance(args.bgm, str) and args.bgm.strip():
        bgm_abs = os.path.abspath(args.bgm)
        if os.path.isfile(bgm_abs):
            logging.info("Background music enabled: %s (video vol=%.2f, bgm vol=%.2f)", bgm_abs, args.video_volume, args.bgm_volume)
            args.bgm = bgm_abs
        else:
            logging.warning("Background music file not found at %s. Proceeding without BGM.", bgm_abs)
            args.bgm = ""

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

    if args.fighting_mode:
        logging.info("Fighting scene detection mode enabled.")
        try:
            # Primary: Motion-based fighting scene detection
            events = detect_fighting_scenes(
                cap, 
                duration, 
                sample_fps=args.fighting_sample_fps,
                motion_threshold=args.motion_threshold,
                min_gap_s=args.min_gap,
                max_clips=args.max_clips if args.max_clips > 0 else None
            )
            logging.info("Motion analysis detected %d fighting scenes.", len(events))
            
            # Supplement with combat audio detection if motion detection yielded few results
            if len(events) < 3:
                logging.info("Supplementing with combat audio detection...")
                with tempfile.TemporaryDirectory(prefix="nba_audio_") as td:
                    wav_path = os.path.join(td, "audio.wav")
                    try:
                        extract_audio_wav(input_path, wav_path, ar=16000)
                        combat_events = detect_combat_audio_peaks(
                            wav_path, 
                            threshold_std=args.combat_audio_threshold,
                            min_gap_s=args.min_gap,
                            max_peaks=args.max_clips if args.max_clips > 0 else None
                        )
                        # Merge and dedupe events
                        all_events = events + combat_events
                        all_events = dedupe_close_events(all_events, args.min_gap)
                        events = all_events
                        logging.info("Combined motion and audio detection found %d fighting scenes.", len(events))
                    except Exception as e:
                        logging.warning("Combat audio detection failed: %s", e)
        except Exception as e:
            logging.error("Fighting scene detection failed: %s", e)
            sys.exit(1)
    else:
        # Original NBA highlight detection logic
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

    # Determine clips directory from args or derive from output path
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    default_clips_dir = os.path.join(os.path.dirname(output_path), base_name + "_clips")
    clips_dir = os.path.abspath(args.clips_dir) if getattr(args, "clips_dir", None) else default_clips_dir
    os.makedirs(clips_dir, exist_ok=True)
    logging.info("Saving individual clips to: %s", clips_dir)

    wm_path = os.path.abspath(args.watermark).strip() if isinstance(args.watermark, str) else ""
    apply_wm = len(wm_path) > 0 and os.path.isfile(wm_path)
    if not apply_wm and len(wm_path) > 0:
        logging.warning("Watermark file not found at %s. Proceeding without watermark.", wm_path)

    with tempfile.TemporaryDirectory(prefix="nba_highlight_") as td:
        concat_inputs: List[str] = []
        for i, (start, dur) in enumerate(segments):
            temp_seg = os.path.join(td, f"seg_{i:04d}.mp4")
            try:
                ffmpeg_extract_segment(input_path, start, dur, temp_seg, reencode=args.reencode, pitch_factor=args.pitch)
            except subprocess.CalledProcessError as e:
                logging.warning("Failed to cut segment %d at %.2fs (%.2fs): %s", i, start, dur, e)
                continue

            # Final per-clip output path
            clip_name = f"{base_name}_{i:04d}.mp4"
            clip_out = os.path.join(clips_dir, clip_name)
            try:
                # Finalize each clip: watermark and/or background music as requested
                if apply_wm or (isinstance(args.bgm, str) and len(args.bgm) > 0):
                    ffmpeg_finalize_clip(
                        input_path=temp_seg,
                        output_path=clip_out,
                        watermark_path=wm_path if apply_wm else "",
                        wm_margin=args.wm_margin,
                        wm_width=args.wm_width,
                        pitch_factor=args.pitch,
                        bgm_path=args.bgm if isinstance(args.bgm, str) else "",
                        video_volume=args.video_volume,
                        bgm_volume=args.bgm_volume,
                    )
                else:
                    # No finalize processing requested; just move the cut segment
                    shutil.move(temp_seg, clip_out)
                concat_inputs.append(clip_out)
            except subprocess.CalledProcessError as e:
                logging.warning("Failed to finalize clip %d: %s", i, e)
                continue

        if not concat_inputs:
            logging.error("No clips were successfully created.")
            sys.exit(3)

        if args.no_combined:
            logging.info("Skipping combined highlights output (--no-combined specified).")
        else:
            # Write concat list from per-clip outputs
            list_path = os.path.join(td, "concat.txt")
            with open(list_path, "w", encoding="utf-8") as f:
                for p in concat_inputs:
                    # Use forward slashes or escape backslashes
                    fp = p.replace("\\", "/")
                    f.write(f"file '{fp}'\n")

            # Concatenate
            try:
                ffmpeg_concat_filelist(list_path, output_path)
            except subprocess.CalledProcessError:
                logging.warning("Concat with stream copy failed. Re-encoding combined output as fallback...")
                # Fallback: re-encode concat using filter_complex
                cmd = ["ffmpeg", "-y"]
                for p in concat_inputs:
                    cmd.extend(["-i", p])
                n = len(concat_inputs)
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
                    output_path,
                ])
                subprocess.run(cmd, check=True)

    logging.info("Highlights written to: %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
