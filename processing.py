import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from model import load_model, get_vehicle_classes, MODEL_PATH

# Aesthetic High-Contrast Neon Palette (BGR) for vehicle classes
CLASS_COLORS = [
    (255, 170, 0),    # Electric Cyan
    (0, 215, 255),    # Golden Amber
    (70, 235, 90),    # Emerald Neon
    (255, 65, 180),   # Neon Magenta
    (20, 160, 255),   # Coral Flame
    (230, 140, 255),  # Radiant Purple
    (255, 235, 50),   # Vivid Azure
    (50, 255, 220),   # Aquamarine
]


def get_color_for_id(idx: int):
    return CLASS_COLORS[idx % len(CLASS_COLORS)]


def make_tracker_state(
    height: int,
    vehicle_classes: dict,
    inbound_ratio: float = 0.40,
    outbound_ratio: float = 0.68,
    line_ratio: float = None,
    corridor_gap: float = None,
    zone_buffer: int = 25,
) -> tuple:
    """
    Initializes tracking and natural dual-boundary traffic corridor state.
    
    Args:
        height: Video frame height in pixels
        vehicle_classes: Dict mapping class IDs to vehicle class names
        inbound_ratio: Vertical position of Inbound Line as fraction of frame height (default: 0.40 = 40%)
        outbound_ratio: Vertical position of Outbound Line as fraction of frame height (default: 0.68 = 68%)
        line_ratio: Alternative: center line position as fraction of height
        corridor_gap: Alternative: vertical gap between boundary lines as fraction of height
        zone_buffer: Unused zone buffer parameter (default: 25 pixels)
    
    Returns:
        Tuple containing:
        - boundaries: (inbound_y, outbound_y) in pixels
        - counts: Dict tracking in/out counts per vehicle class
        - counted_ids: Set of track IDs already counted
        - track_history: Deque of 45 frames storing centroid positions per track_id
        - track_state: Dict tracking origin, first_y, and crossing state per track_id
    
    Magic Numbers Explained:
        - 45: Maximum trajectory history length (for motion trails and crossing detection)
        - 10: Minimum margin from frame edges for boundary lines
        - 20: Minimum acceptable corridor gap in pixels
    """
    if line_ratio is not None and corridor_gap is not None:
        center_y = int(height * line_ratio)
        gap_px = max(20, int(height * corridor_gap))
        inbound_y = max(10, center_y - gap_px // 2)
        outbound_y = min(height - 10, center_y + gap_px // 2)
    else:
        inbound_y = max(10, int(height * min(inbound_ratio, outbound_ratio)))
        outbound_y = min(height - 10, int(height * max(inbound_ratio, outbound_ratio)))

    # Ensure a reasonable minimum gap
    if outbound_y - inbound_y < 20:
        outbound_y = min(height - 5, inbound_y + 25)

    counts = defaultdict(lambda: {"in": 0, "out": 0})
    for name in vehicle_classes.values():
        counts[name] = {"in": 0, "out": 0}

    counted_ids = set()
    track_history = defaultdict(lambda: deque(maxlen=45))
    track_state = defaultdict(lambda: {"origin": None, "first_y": None, "crossed_inbound": False, "crossed_outbound": False})

    return (inbound_y, outbound_y), counts, counted_ids, track_history, track_state


def draw_hud(frame, counts: dict, fps: float = 0.0, boundaries: tuple = None):
    """Draws a deep OLED-black aesthetic frosted glass HUD with live traffic flow metrics."""
    h, w = frame.shape[:2]

    total_in = sum(c["in"] for c in counts.values())
    total_out = sum(c["out"] for c in counts.values())
    total_vehicles = total_in + total_out

    active_classes = [name for name, c in counts.items() if (c["in"] + c["out"]) > 0 or len(counts) <= 6]
    row_height = 25
    hud_w = 290
    hud_h = 86 + len(active_classes) * row_height

    hud_x1, hud_y1 = 18, 18
    hud_x2, hud_y2 = hud_x1 + hud_w, hud_y1 + hud_h

    # Deep OLED Black Frosted Card with Sleek Neon Glow Accent
    overlay = frame.copy()
    cv2.rectangle(overlay, (hud_x1, hud_y1), (hud_x2, hud_y2), (4, 6, 10), -1)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

    cv2.rectangle(frame, (hud_x1, hud_y1), (hud_x2, hud_y2), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (hud_x1 + 2, hud_y1 + 2), (hud_x2 - 2, hud_y2 - 2), (45, 55, 70), 1, cv2.LINE_AA)

    # Status LED & Title
    cv2.circle(frame, (hud_x1 + 16, hud_y1 + 24), 4, (0, 255, 128), -1)
    cv2.putText(frame, "TRAFFIC FLOW MONITOR", (hud_x1 + 28, hud_y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)

    # Sub-header Natural KPI Summary
    stat_line = f"TOTAL: {total_vehicles:03d}   |  IN: {total_in:02d}  v   |  OUT: {total_out:02d}  ^"
    cv2.putText(frame, stat_line, (hud_x1 + 14, hud_y1 + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 245, 250), 1, cv2.LINE_AA)

    cv2.line(frame, (hud_x1 + 12, hud_y1 + 62), (hud_x2 - 12, hud_y2 + 62) if False else (hud_x2 - 12, hud_y1 + 62), (50, 60, 75), 1)

    # Class rows
    y_pos = hud_y1 + 84
    for class_name in active_classes:
        c = counts[class_name]
        text_cls = f"{class_name.upper():<10}"
        text_in = f"IN:{c['in']:02d}"
        text_out = f"OUT:{c['out']:02d}"
        text_tot = f"[{c['in'] + c['out']:02d}]"

        cv2.putText(frame, text_cls, (hud_x1 + 14, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 210), 1, cv2.LINE_AA)
        cv2.putText(frame, text_in, (hud_x1 + 120, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, text_out, (hud_x1 + 180, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 155, 70), 1, cv2.LINE_AA)
        cv2.putText(frame, text_tot, (hud_x1 + 245, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        y_pos += row_height


def process_frame(
    model,
    frame,
    vehicle_classes: dict,
    conf_threshold: float,
    boundaries: tuple,
    counts: dict,
    counted_ids: set,
    track_history: dict,
    track_state: dict,
    draw_trails: bool = True,
    show_hud: bool = True,
    imgsz: int = 640,
    **kwargs,
) -> np.ndarray:
    """
    Runs YOLO ByteTrack detection and extended dual-boundary corridor tracking on a single frame.
    
    Args:
        model: YOLO model instance for detection
        frame: Input video frame (BGR format)
        vehicle_classes: Dict mapping class IDs to vehicle names
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        boundaries: Tuple (inbound_y, outbound_y) - line positions in pixels
        counts: Dict tracking in/out counts per class
        counted_ids: Set of already-counted track IDs (to avoid double-counting)
        track_history: Dict of deques (45 frames max) storing centroid positions per track_id
        track_state: Dict tracking origin, first_y, and crossing state per track_id
        draw_trails: Whether to render centroid motion trails
        show_hud: Whether to render HUD overlay with live metrics
    
    Returns:
        Annotated frame with bounding boxes, trails, boundary lines, and HUD
    
    Crossing Logic:
        - Vehicles crossing from top/middle → bottom/exit are counted as INBOUND (IN)
        - Vehicles crossing from bottom/exit → top/middle are counted as OUTBOUND (OUT)
        - Multi-frame ray-casting prevents skipping fast-moving vehicles
        - Minimum trajectory span of 6 pixels required to register crossing
    
    Magic Numbers Explained:
        - 8: Recent trajectory points checked for ray-crossing (robust for speed variance)
        - 6: Minimum vertical displacement to confirm crossing (noise rejection)
        - 2: Minimum crossing confirmation threshold (prevents jitter)
        - 45: Deque maxlen for trajectory history (balance memory vs. accuracy)
    """
    height, width = frame.shape[:2]
    inbound_y, outbound_y = boundaries
    class_ids = list(vehicle_classes.keys())

    # Run YOLO tracking with ByteTrack
    results = model.track(
        frame,
        classes=class_ids if class_ids else None,
        conf=conf_threshold,
        imgsz=imgsz,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    # 1. Render Extended Active Traffic Zone
    zone_overlay = frame.copy()
    cv2.rectangle(zone_overlay, (0, inbound_y), (width, outbound_y), (0, 130, 255), -1)
    cv2.addWeighted(zone_overlay, 0.10, frame, 0.90, 0, frame)

    # Inbound Boundary Line (Incoming Flow Boundary)
    inbound_color = (0, 235, 255)  # Electric Cyan
    cv2.line(frame, (0, inbound_y), (width, inbound_y), inbound_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "--- INBOUND LINE (INCOMING TRAFFIC) ---", (20, max(18, inbound_y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, inbound_color, 1, cv2.LINE_AA)

    # Outbound Boundary Line (Outgoing Flow Boundary)
    outbound_color = (0, 150, 255)  # Coral Flame
    cv2.line(frame, (0, outbound_y), (width, outbound_y), outbound_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "--- OUTBOUND LINE (DEPARTING TRAFFIC) ---", (20, min(height - 8, outbound_y + 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, outbound_color, 1, cv2.LINE_AA)

    boxes = results[0].boxes if len(results) > 0 else None
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist()
        xyxy = boxes.xyxy.int().cpu().tolist()

        for box, track_id, cls_id, conf in zip(xyxy, ids, classes, confs):
            x1, y1, x2, y2 = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            class_name = vehicle_classes.get(cls_id, model.names.get(cls_id, f"class_{cls_id}"))
            color = get_color_for_id(cls_id)

            # Record trajectory centroid history
            history = track_history[track_id]
            history.append((cx, cy))

            # Initialize track origin state
            state = track_state[track_id]
            if state["first_y"] is None:
                state["first_y"] = cy
                if cy < inbound_y:
                    state["origin"] = "north"
                elif cy > outbound_y:
                    state["origin"] = "south"
                else:
                    state["origin"] = "inside_zone"

            # 2. Comprehensive Dual-Boundary Crossing Engine
            if track_id not in counted_ids and len(history) >= 2:
                crossed_in = False
                crossed_out = False

                curr_cx, curr_cy = history[-1]
                start_cx, start_cy = history[0]

                # Calculate trajectory step differences
                total_dy = curr_cy - start_cy
                recent_pts = list(history)[-min(len(history), 8):]
                min_recent_y = min(p[1] for p in recent_pts)
                max_recent_y = max(p[1] for p in recent_pts)

                # Check step-by-step ray crossings across recent points
                for i in range(len(history) - 1, max(0, len(history) - 8), -1):
                    p_y1 = history[i - 1][1]
                    p_y2 = history[i][1]

                    # Downward crossing across Inbound line or Outbound line
                    if p_y2 > p_y1:
                        if (p_y1 <= inbound_y < p_y2) or (p_y1 < outbound_y <= p_y2):
                            if total_dy >= 2:
                                crossed_in = True
                                break

                    # Upward crossing across Outbound line or Inbound line
                    elif p_y2 < p_y1:
                        if (p_y1 >= outbound_y > p_y2) or (p_y1 > inbound_y >= p_y2):
                            if total_dy <= -2:
                                crossed_out = True
                                break

                # Fallback: Whole trajectory span check
                if not crossed_in and not crossed_out and len(history) >= 3:
                    if total_dy > 6:
                        # Moving downward: spans past inbound or outbound boundary
                        if (start_cy <= inbound_y and curr_cy >= inbound_y + 4) or (start_cy < outbound_y and curr_cy >= outbound_y):
                            crossed_in = True
                    elif total_dy < -6:
                        # Moving upward: spans past outbound or inbound boundary
                        if (start_cy >= outbound_y and curr_cy <= outbound_y - 4) or (start_cy > inbound_y and curr_cy <= inbound_y):
                            crossed_out = True

                if crossed_in:
                    counted_ids.add(track_id)
                    counts[class_name]["in"] += 1
                    cv2.line(frame, (0, inbound_y), (width, inbound_y), (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f"+1 {class_name.upper()} [IN]", (cx, cy - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

                elif crossed_out:
                    counted_ids.add(track_id)
                    counts[class_name]["out"] += 1
                    cv2.line(frame, (0, outbound_y), (width, outbound_y), (0, 120, 255), 4, cv2.LINE_AA)
                    cv2.putText(frame, f"+1 {class_name.upper()} [OUT]", (cx, cy - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 255), 2, cv2.LINE_AA)

            # Draw Centroid Motion Trail
            if draw_trails and len(history) > 1:
                pts = np.array(list(history), np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Draw Vehicle Bounding Box & Centroid
            box_thickness = 3 if track_id in counted_ids else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255) if track_id in counted_ids else color, -1)

            # High-Tech Badge Label
            status_tag = " [RECORDED]" if track_id in counted_ids else ""
            label = f"#{track_id} {class_name.upper()} {conf:.2f}{status_tag}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_h - baseline - 6)),
                (x1 + text_w + 8, max(0, y1)),
                (6, 8, 14),
                -1,
            )
            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_h - baseline - 6)),
                (x1 + text_w + 8, max(0, y1)),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Render HUD overlay
    if show_hud:
        draw_hud(frame, counts, boundaries=boundaries)

    return frame


def ensure_web_compatible_video(source_video_path: str, target_video_path: str) -> bool:
    """
    Transcodes a video to H.264 (yuv420p) MP4 using imageio_ffmpeg or system ffmpeg
    so that it can be played directly in web browsers and Streamlit.
    """
    src = Path(source_video_path).resolve()
    dst = Path(target_video_path).resolve()

    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe and src.exists() and src.stat().st_size > 0:
        temp_transcode = dst.with_name(f"temp_h264_{dst.name}")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(src),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-movflags", "+faststart",
            str(temp_transcode),
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if temp_transcode.exists() and temp_transcode.stat().st_size > 0:
                if dst.exists():
                    os.remove(dst)
                shutil.move(str(temp_transcode), str(dst))
                if src != dst and src.exists():
                    os.remove(src)
                return True
        except subprocess.CalledProcessError as ffmpeg_err:
            print(f"Warning: FFmpeg transcoding failed: {ffmpeg_err}. Using fallback (copy).")
            if temp_transcode.exists():
                try:
                    os.remove(temp_transcode)
                except OSError:
                    pass
        except Exception as general_err:
            print(f"Warning: FFmpeg transcoding error: {general_err}. Using fallback (copy).")
            if temp_transcode.exists():
                try:
                    os.remove(temp_transcode)
                except OSError:
                    pass

    if src != dst and src.exists():
        shutil.copyfile(src, dst)
    return True


def process_video_file(
    input_path,
    output_path,
    model_name_or_path: str = MODEL_PATH,
    confidence: float = 0.4,
    inbound_ratio: float = 0.40,
    outbound_ratio: float = 0.68,
    line_ratio: float = None,
    corridor_gap: float = None,
    zone_buffer: int = 25,
    class_preset: str = "auto",
    frame_stride: int = 1,
    imgsz: int = 640,
    progress_callback=None,
    stop_check=None,
    **kwargs,
):
    """
    Runs the full detection + tracking + extended dual-boundary corridor counting pipeline on a video file end to end.
    - frame_stride: 1 = all frames, 2 = 2x speed (process every 2nd frame), 3 = 3x speed (process every 3rd frame).
    - imgsz: inference resolution (default 640).
    Backward compatibility: older callers may still pass extra kwargs, which are normalized here.
    """
    # Backward compatibility for older callers or stale wrappers that inject frame_stride/imgsz via kwargs.
    if "frame_stride" in kwargs:
        frame_stride = kwargs.pop("frame_stride")
    if "imgsz" in kwargs:
        imgsz = kwargs.pop("imgsz")
    if kwargs:
        # Ignore legacy or unknown parameters instead of failing with unexpected keyword errors.
        pass

    model = load_model(model_name_or_path)
    vehicle_classes = get_vehicle_classes(model, preset=class_preset)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output FPS proportional to stride to preserve natural playback speed
    output_fps = max(10.0, float(fps) / max(1, frame_stride))

    # Write temporary intermediate video with OpenCV
    temp_raw_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_raw_output.close()
    raw_path = temp_raw_output.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, output_fps, (width, height))

    boundaries, counts, counted_ids, track_history, track_state = make_tracker_state(
        height,
        vehicle_classes,
        inbound_ratio=inbound_ratio,
        outbound_ratio=outbound_ratio,
        line_ratio=line_ratio,
        corridor_gap=corridor_gap,
        zone_buffer=zone_buffer,
    )

    frame_num = 0
    inference_times = []
    pipeline_start = time.time()

    try:
        while cap.isOpened():
            if stop_check and stop_check():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            infer_start = time.time()

            annotated_frame = process_frame(
                model=model,
                frame=frame,
                vehicle_classes=vehicle_classes,
                conf_threshold=confidence,
                boundaries=boundaries,
                counts=counts,
                counted_ids=counted_ids,
                track_history=track_history,
                track_state=track_state,
                draw_trails=True,
                show_hud=True,
                imgsz=imgsz,
            )
            infer_duration = time.time() - infer_start
            inference_times.append(infer_duration)

            writer.write(annotated_frame)

            if progress_callback:
                current_fps = (1.0 / infer_duration) if infer_duration > 0 else 0.0
                try:
                    progress_callback(frame_num, total_frames, annotated_frame, counts, current_fps)
                except TypeError:
                    try:
                        progress_callback(frame_num, total_frames)
                    except Exception as callback_err:
                        print(f"Warning: Progress callback failed: {callback_err}")

            # Fast frame stride skipping
            if frame_stride > 1:
                for _ in range(frame_stride - 1):
                    if not cap.grab():
                        break
                    frame_num += 1

    finally:
        cap.release()
        writer.release()

    # Transcode to H.264 web-compatible MP4
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_web_compatible_video(raw_path, str(output_path))
    if os.path.exists(raw_path):
        try:
            os.remove(raw_path)
        except OSError:
            pass

    total_time = time.time() - pipeline_start
    avg_inference = (sum(inference_times) / len(inference_times)) if inference_times else 0.0
    stats = {
        "total_frames": frame_num,
        "total_time_sec": round(total_time, 2),
        "avg_inference_ms": round(avg_inference * 1000, 1),
    }

    clean_counts = {k: dict(v) for k, v in counts.items()}
    return clean_counts, stats
