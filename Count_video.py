import csv
import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from model import MODEL_PATH, get_available_models, load_model, get_vehicle_classes
from processing import process_video_file, make_tracker_state, process_frame


# Streamlit resource caching to prevent redundant model loading on reruns
@st.cache_resource
def get_cached_model(model_name: str):
    """Cache model loading to avoid redundant GPU allocation during Streamlit reruns."""
    return load_model(model_name)


st.set_page_config(
    page_title="Vehicle Detection & Counting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ultra-Aesthetic OLED Dark Interface CSS with High-Contrast Buttons
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Dark Theme */
    html, body, [class*="css"], .stApp {
        background-color: #030407 !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #06080e !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    .app-header {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-bottom: 0.15rem;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 1.4rem;
    }

    .metric-card {
        background: #080b12;
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.8), 0 0 15px -3px rgba(56, 189, 248, 0.1);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.55);
        box-shadow: 0 12px 35px -8px rgba(0, 0, 0, 0.9), 0 0 20px -2px rgba(56, 189, 248, 0.2);
        transform: translateY(-2px);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-badge-in {
        color: #4ade80;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-badge-out {
        color: #fb923c;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* High Visibility Buttons */
    .stButton button, [data-testid="baseButton-primary"], [data-testid="baseButton-secondary"], .stDownloadButton button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.6) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button *, [data-testid="baseButton-primary"] *, .stDownloadButton button * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Dropdowns / Selectboxes */
    div[data-baseweb="select"] {
        background-color: #0c1017 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
        background-color: #0c1017 !important;
    }
    div[data-baseweb="popover"] * {
        color: #f8fafc !important;
        background-color: #0c1017 !important;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] section {
        background-color: #080b12 !important;
        border: 1px dashed rgba(56, 189, 248, 0.4) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section * {
        color: #cbd5e1 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #0284c7 !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploader"] button * {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header">🚗 Vehicle Detection & Traffic Counting</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Real-time YOLO vehicle detection, ByteTrack tracking & bidirectional traffic flow counting</div>', unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    available_models = get_available_models()
    selected_model = st.selectbox("📦 YOLO Model Weights", available_models, index=0)

    # Display detected vehicle classes directly from model
    try:
        preview_model = get_cached_model(selected_model)
        detected_vtypes = list(get_vehicle_classes(preview_model).values())
        vtype_pills = " • ".join([v.capitalize() for v in detected_vtypes])
        st.markdown(f"<div style='font-size:0.83rem; color:#38bdf8; margin-bottom:12px;'>🚗 <b>Target Vehicles:</b><br><span style='color:#cbd5e1;'>{vtype_pills}</span></div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='font-size:0.83rem; color:#38bdf8; margin-bottom:12px;'>🚗 <b>Target Vehicles:</b><br><span style='color:#cbd5e1;'>Car • Truck • Bus • Motorcycle</span></div>", unsafe_allow_html=True)

    confidence = st.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

    speed_mode = st.select_slider(
        "⚡ Processing Speed",
        options=["1x Precision", "2x Turbo (Recommended)", "3x Ultra"],
        value="2x Turbo (Recommended)",
        help="Higher speed uses fewer processed frames while keeping the output smooth enough for counting.",
    )
    stride_map = {"1x Precision": 1, "2x Turbo (Recommended)": 2, "3x Ultra": 3}
    frame_stride = stride_map.get(speed_mode, 2)

    st.markdown("---")
    st.markdown("#### 🛣️ Counting Boundaries")
    inbound_ratio = st.slider(
        "Inbound Line (Incoming %)",
        10, 80, 40, 2,
        help="Top boundary line where incoming vehicles enter.",
    ) / 100.0

    outbound_ratio = st.slider(
        "Outbound Line (Departing %)",
        20, 95, 70, 2,
        help="Bottom boundary line where departing vehicles enter.",
    ) / 100.0

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5;">
            <b>🧭 Directional Flow:</b><br>
            • <span style="color:#4ade80;"><b>Incoming (Inbound)</b></span>: Top ➔ Bottom<br>
            • <span style="color:#fb923c;"><b>Departing (Outbound)</b></span>: Bottom ➔ Top
        </div>
        """,
        unsafe_allow_html=True,
    )

source = st.radio("Select Video Input", ["Upload Video File", "Live Local Camera"], horizontal=True)


def build_report_csv(counts: dict, stats: dict, model_name: str) -> str:
    buffer = io.StringIO()
    writer_csv = csv.writer(buffer)
    writer_csv.writerow(["timestamp", datetime.now().isoformat()])
    writer_csv.writerow(["model", model_name])
    writer_csv.writerow(["total_frames", stats["total_frames"]])
    writer_csv.writerow(["total_processing_time_sec", stats["total_time_sec"]])
    writer_csv.writerow(["avg_inference_time_ms", stats["avg_inference_ms"]])
    writer_csv.writerow([])
    writer_csv.writerow(["class", "in", "out", "total"])
    for class_name, class_counts in counts.items():
        total = class_counts["in"] + class_counts["out"]
        writer_csv.writerow([class_name, class_counts["in"], class_counts["out"], total])
    return buffer.getvalue()


# ---------------- Upload Video mode ----------------
if source == "Upload Video File":
    uploaded_file = st.file_uploader("Upload traffic video file", type=["mp4", "mov", "avi", "mkv"])
    start_button = st.button("🚀 Start Video Analysis", type="primary", disabled=uploaded_file is None)

    if start_button and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name

        output_path = "outputs/app_output.mp4"
        Path("outputs").mkdir(exist_ok=True)

        st.info("🎬 Processing video through extended dual-boundary corridor. Live detection preview below:")

        # Real-time metrics
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            kpi_tot = st.empty()
        with kpi_c2:
            kpi_in = st.empty()
        with kpi_c3:
            kpi_out = st.empty()
        with kpi_c4:
            kpi_spd = st.empty()

        progress_bar = st.progress(0, text="Initializing YOLO ByteTrack Corridor Engine...")

        v_col, t_col = st.columns([3, 2])
        with v_col:
            st.markdown("##### 📹 Live Corridor Stream")
            frame_placeholder = st.empty()
        with t_col:
            st.markdown("##### 📊 Live Vehicle Breakdown")
            table_placeholder = st.empty()

        last_update = [0.0]
        last_img_update = [0.0]

        def update_progress(current_frame, total_frames, annotated_frame, current_counts, current_fps):
            now = time.time()
            total_in = sum(c["in"] for c in current_counts.values())
            total_out = sum(c["out"] for c in current_counts.values())
            grand_total = total_in + total_out

            pct = min(current_frame / total_frames, 1.0) if total_frames > 0 else 0.0

            # Throttle WebSocket image rendering to max ~20 FPS (every 0.05s) to eliminate Streamlit frontend lag
            if now - last_img_update[0] > 0.05 or current_frame == total_frames:
                last_img_update[0] = now
                frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                progress_bar.progress(pct, text=f"Analyzing frame {current_frame}/{total_frames} ({pct * 100:.1f}%) — {current_fps:.1f} FPS")

            if now - last_update[0] > 0.10 or current_frame == total_frames:
                last_update[0] = now
                kpi_tot.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Traffic Flow</div>
                        <div class="metric-value">{grand_total}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_in.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Incoming (Inbound)</div>
                        <div class="metric-value metric-badge-in">⬇ {total_in}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_out.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Departing (Outbound)</div>
                        <div class="metric-value metric-badge-out">⬆ {total_out}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_spd.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Speed</div>
                        <div class="metric-value" style="color: #38bdf8;">{current_fps:.1f} <span style="font-size:0.9rem;">FPS</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                df_live = pd.DataFrame([
                    {
                        "Vehicle Type": name.upper(),
                        "Incoming (Inbound)": c["in"],
                        "Departing (Outbound)": c["out"],
                        "Total Count": c["in"] + c["out"],
                    }
                    for name, c in current_counts.items()
                    if (c["in"] + c["out"]) > 0 or len(current_counts) <= 6
                ])
                table_placeholder.dataframe(df_live, use_container_width=True, hide_index=True)

        counts, stats = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model_name_or_path=selected_model,
            confidence=confidence,
            inbound_ratio=inbound_ratio,
            outbound_ratio=outbound_ratio,
            frame_stride=frame_stride,
            progress_callback=update_progress,
        )

        progress_bar.empty()
        st.success("🎉 Video Surveillance Completed!")

        # Clean up temp input
        if os.path.exists(input_path):
            try:
                os.remove(input_path)
            except OSError:
                pass

        # Final video and download
        st.markdown("---")
        res_v, res_t = st.columns([3, 2])
        with res_v:
            st.subheader("📼 Annotated Replay Video")
            st.video(output_path)

        with res_t:
            st.subheader("📥 Final Breakdown & CSV")
            df_counts = pd.DataFrame([
                {
                    "Vehicle Type": name.upper(),
                    "Incoming (Inbound)": c["in"],
                    "Departing (Outbound)": c["out"],
                    "Total Count": c["in"] + c["out"],
                }
                for name, c in counts.items()
            ])
            st.dataframe(df_counts, use_container_width=True, hide_index=True)

            report_csv = build_report_csv(counts, stats, selected_model)
            st.download_button(
                "📥 Download CSV Report",
                data=report_csv,
                file_name="vehicle_traffic_report.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )


# ---------------- Webcam mode ----------------
else:
    st.info("🎥 Live camera telemetry streams through your machine's default video input (Device 0).")
    
    # Create session state for camera control
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False
    
    # Button layout for camera control
    col_cam_start, col_cam_stop = st.columns(2)
    with col_cam_start:
        if st.button("▶️ Start Live Camera Feed", type="primary", use_container_width=True):
            st.session_state.camera_running = True
    with col_cam_stop:
        if st.button("⏹️ Stop Live Camera Feed", type="secondary", use_container_width=True):
            st.session_state.camera_running = False
    
    frame_placeholder = st.empty()
    counts_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.camera_running:
        try:
            model = load_model(selected_model)
            vehicle_classes = get_vehicle_classes(model)
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not access camera device 0. Ensure camera is connected and not in use by another application.")
            else:
                status_placeholder.info("✅ Camera feed active. Press 'Stop' button above to exit.")
                
                ret, first_frame = cap.read()
                if ret:
                    height = first_frame.shape[0]
                    boundaries, counts, counted_ids, track_history, track_state = make_tracker_state(
                        height, vehicle_classes, inbound_ratio=inbound_ratio, outbound_ratio=outbound_ratio
                    )
                    
                    frame_count = 0
                    start_time = time.time()

                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            status_placeholder.warning("⚠️ Lost camera feed connection. Check camera availability.")
                            break

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
                        )
                        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                        df_webcam = pd.DataFrame([
                            {"Vehicle Type": name.upper(), "IN": c["in"], "OUT": c["out"], "Total": c["in"] + c["out"]}
                            for name, c in counts.items()
                            if (c["in"] + c["out"]) > 0 or len(counts) <= 6
                        ])
                        counts_placeholder.dataframe(df_webcam, use_container_width=True, hide_index=True)
                        
                        frame_count += 1
                        if frame_count % 30 == 0:  # Update status every 30 frames
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed if elapsed > 0 else 0
                            status_placeholder.info(f"✅ Camera feed active · Frames: {frame_count} · FPS: {fps:.1f}")

                cap.release()
                status_placeholder.success("✅ Camera feed stopped cleanly.")
        except Exception as camera_err:
            st.error(f"❌ Camera feed error: {camera_err}")
            st.session_state.camera_running = False
    else:
        status_placeholder.write("Camera feed is not active. Click 'Start' to begin.")