import csv
import io
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import requests
import streamlit as st
from sqlalchemy import func as sqla_func

from model import MODEL_PATH, get_available_models, load_model, get_vehicle_classes
from processing import process_video_file, make_tracker_state, process_frame
from database import SessionLocal
from db_models import Run, ClassCount

API_BASE = "http://localhost:8000"


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

# Custom High-End Modern Styling with Theme Responsive Button Text Colors
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Base Theme */
    html, body, [class*="css"], .stApp {
        background-color: #030407 !important;
        color: #f8fafc !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* Sidebar Dark Glass */
    [data-testid="stSidebar"] {
        background-color: #06080e !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    }
    [data-testid="stSidebar"] * {
        color: #f1f5f9 !important;
    }

    /* Gradient Title */
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

    /* OLED Black Metric Cards with Neon Accents */
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

    /* Primary Action Buttons - White Writing on Dark Theme */
    .stButton button, [data-testid="baseButton-primary"] {
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
    .stButton button *, [data-testid="baseButton-primary"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton button:hover, [data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border-color: #7dd3fc !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Download Buttons - White Writing on Emerald Theme */
    .stDownloadButton button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
        color: #ffffff !important;
        border: 1px solid #10b981 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.55rem 1.3rem !important;
    }
    .stDownloadButton button * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border-color: #34d399 !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5) !important;
    }

    /* Danger / Delete Action Buttons - Distinct Red Text / Border */
    [data-testid="baseButton-secondary"] {
        background: rgba(220, 38, 38, 0.18) !important;
        border: 1px solid #ef4444 !important;
        color: #f87171 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    [data-testid="baseButton-secondary"] * {
        color: #f87171 !important;
        font-weight: 700 !important;
    }
    [data-testid="baseButton-secondary"]:hover {
        background: #ef4444 !important;
        color: #ffffff !important;
        border-color: #f87171 !important;
        box-shadow: 0 0 16px rgba(239, 68, 68, 0.5) !important;
    }
    [data-testid="baseButton-secondary"]:hover * {
        color: #ffffff !important;
    }

    /* Selectbox, Inputs & Popovers */
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

    /* Status Pills */
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 9999px;
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-pill-local {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 14px;
        border-radius: 9999px;
        background: rgba(234, 179, 8, 0.12);
        color: #facc15;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Aesthetic Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #080b12;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px 20px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(129, 140, 248, 0.12) 100%) !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        box-shadow: 0 0 15px -3px rgba(56, 189, 248, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Banner
st.markdown('<div class="app-header">🚗 Vehicle Detection & Traffic Counting</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Real-time multi-lane vehicle detection, ByteTrack tracking & bidirectional traffic counting</div>', unsafe_allow_html=True)

# ----------------- Sidebar Controls -----------------
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")

    # Backend Connection Check
    backend_online = False
    try:
        r = requests.get(f"{API_BASE}/models", timeout=1)
        if r.status_code == 200:
            backend_online = True
            available_models = r.json().get("models", get_available_models())
        else:
            available_models = get_available_models()
    except Exception:
        available_models = get_available_models()

    if backend_online:
        st.markdown('<div class="status-pill-online">● Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill-local">● Direct Engine Mode</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    selected_model = st.selectbox("📦 YOLO Model Weights", available_models, index=0)

    # Display detected vehicle classes directly from model
    try:
        preview_model = get_cached_model(selected_model)
        detected_vtypes = list(get_vehicle_classes(preview_model).values())
        vtype_pills = " • ".join([v.capitalize() for v in detected_vtypes])
        st.markdown(f"<div style='font-size:0.83rem; color:#38bdf8; margin-bottom:12px;'>🚗 <b>Target Vehicles:</b><br><span style='color:#cbd5e1;'>{vtype_pills}</span></div>", unsafe_allow_html=True)
    except Exception:
        st.markdown("<div style='font-size:0.83rem; color:#38bdf8; margin-bottom:12px;'>🚗 <b>Target Vehicles:</b><br><span style='color:#cbd5e1;'>Car • Truck • Bus • Motorcycle</span></div>", unsafe_allow_html=True)

    confidence = st.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.4, 0.05, help="Minimum detection confidence threshold.")

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
        help="Top boundary line where incoming/southbound vehicles first enter the detection zone.",
    ) / 100.0

    outbound_ratio = st.slider(
        "Outbound Line (Departing %)",
        20, 95, 70, 2,
        help="Bottom boundary line where departing/northbound vehicles enter.",
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


# ----------------- Main Navigation Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Video Analysis",
    "📂 Past Runs",
    "📊 Traffic Analytics",
    "🎥 Live Camera",
])


# ================= Tab 1: Video Processing =================
with tab1:
    col_top1, col_top2 = st.columns([3, 1])
    with col_top1:
        uploaded_file = st.file_uploader(
            "Upload Traffic Video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Select an MP4, MOV, or AVI video to perform vehicle detection and traffic counting.",
        )
    with col_top2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_button = st.button("🚀 Start Video Analysis", type="primary", use_container_width=True, disabled=(uploaded_file is None))

    if start_button and uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(uploaded_file.read())
            input_path = tmp_in.name

        output_dir = Path("outputs").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        unique_id = int(time.time())
        output_path = output_dir / f"run_{unique_id}_output.mp4"

        # Real-time KPI Dash
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            kpi_total_box = st.empty()
        with kpi_c2:
            kpi_in_box = st.empty()
        with kpi_c3:
            kpi_out_box = st.empty()
        with kpi_c4:
            kpi_fps_box = st.empty()

        progress_bar = st.progress(0, text="Initializing YOLO ByteTrack Extended Corridor Engine...")

        v_col, t_col = st.columns([3, 2])
        with v_col:
            st.markdown("##### 📹 Live Surveillance Stream")
            frame_placeholder = st.empty()
        with t_col:
            st.markdown("##### 📊 Live Vehicle Tally")
            table_placeholder = st.empty()

        last_ui_update = [0.0]
        last_img_update = [0.0]

        def live_frame_callback(frame_num, total_frames, annotated_frame, current_counts, current_fps):
            now = time.time()
            total_in = sum(c["in"] for c in current_counts.values())
            total_out = sum(c["out"] for c in current_counts.values())
            grand_total = total_in + total_out

            pct = min(frame_num / total_frames, 1.0) if total_frames > 0 else 0.0

            # Throttle WebSocket frame streaming to max ~20 FPS (every 0.05s) to eliminate Streamlit frontend lag
            if now - last_img_update[0] > 0.05 or frame_num == total_frames:
                last_img_update[0] = now
                frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                progress_bar.progress(pct, text=f"Analyzing frame {frame_num}/{total_frames} ({pct * 100:.1f}%) — {current_fps:.1f} FPS")

            if now - last_ui_update[0] > 0.10 or frame_num == total_frames:
                last_ui_update[0] = now
                kpi_total_box.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Total Traffic Flow</div>
                        <div class="metric-value">{grand_total}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_in_box.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Incoming (Inbound)</div>
                        <div class="metric-value metric-badge-in">⬇ {total_in}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_out_box.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Departing (Outbound)</div>
                        <div class="metric-value metric-badge-out">⬆ {total_out}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_fps_box.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Processing Speed</div>
                        <div class="metric-value" style="color: #38bdf8;">{current_fps:.1f} <span style="font-size:0.9rem;">FPS</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                df_live = pd.DataFrame([
                    {
                        "Vehicle Type": k.upper(),
                        "Incoming (Inbound)": v["in"],
                        "Departing (Outbound)": v["out"],
                        "Total Count": v["in"] + v["out"],
                    }
                    for k, v in current_counts.items()
                    if (v["in"] + v["out"]) > 0 or len(current_counts) <= 6
                ])
                table_placeholder.dataframe(df_live, use_container_width=True, hide_index=True)

        try:
            final_counts, final_stats = process_video_file(
                input_path=input_path,
                output_path=output_path,
                model_name_or_path=selected_model,
                confidence=confidence,
                inbound_ratio=inbound_ratio,
                outbound_ratio=outbound_ratio,
                frame_stride=frame_stride,
                progress_callback=live_frame_callback,
            )

            progress_bar.empty()
            st.success("🎉 Video Surveillance Analysis Completed Successfully!")

            # Save run to SQLite database with explicit session handling
            run_db_id = None
            db = SessionLocal()
            try:
                db_run = Run(
                    filename=uploaded_file.name,
                    status="complete",
                    model_path=selected_model,
                    confidence_threshold=confidence,
                    total_frames=final_stats["total_frames"],
                    total_time_sec=final_stats["total_time_sec"],
                    avg_inference_ms=final_stats["avg_inference_ms"],
                    output_video_path=str(output_path),
                )
                db.add(db_run)
                db.commit()
                db.refresh(db_run)
                run_db_id = db_run.id

                for class_name, c in final_counts.items():
                    db.add(ClassCount(
                        run_id=db_run.id,
                        class_name=class_name,
                        in_count=c["in"],
                        out_count=c["out"],
                    ))
                db.commit()
            except Exception as db_err:
                st.warning(f"Note: Saved results locally. Database sync info: {db_err}")
            finally:
                db.close()

            # Clean up temp input
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except OSError:
                    pass

            # Final summary & CSV download
            st.markdown("---")
            res_vcol, res_tcol = st.columns([3, 2])
            with res_vcol:
                st.subheader("📼 Annotated Replay Stream")
                if output_path.exists():
                    st.video(str(output_path))

            with res_tcol:
                st.subheader("📥 Final Analytics Breakdown")
                df_final = pd.DataFrame([
                    {
                        "Vehicle Type": k.upper(),
                        "Incoming (Inbound)": v["in"],
                        "Departing (Outbound)": v["out"],
                        "Total Count": v["in"] + v["out"],
                    }
                    for k, v in final_counts.items()
                ])
                st.dataframe(df_final, use_container_width=True, hide_index=True)

                buf = io.StringIO()
                cw = csv.writer(buf)
                cw.writerow(["filename", uploaded_file.name])
                cw.writerow(["model", selected_model])
                cw.writerow(["total_frames", final_stats["total_frames"]])
                cw.writerow(["total_time_sec", final_stats["total_time_sec"]])
                cw.writerow(["avg_inference_ms", final_stats["avg_inference_ms"]])
                cw.writerow([])
                cw.writerow(["class", "in", "out", "total"])
                for k, v in final_counts.items():
                    cw.writerow([k, v["in"], v["out"], v["in"] + v["out"]])

                report_filename = f"run_{run_db_id or unique_id}_traffic_report.csv"
                st.download_button(
                    "📥 Export Run Report (.csv)",
                    data=buf.getvalue(),
                    file_name=report_filename,
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error during video surveillance processing: {e}")


# ================= Tab 2: Past Runs Inspector =================
with tab2:
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.subheader("Historical Video Runs & Reports")
    with col_h2:
        if st.button("🗑️ Clear All Past Runs", type="secondary", help="Deletes all run records and output videos"):
            cleared = False
            if backend_online:
                try:
                    r_del = requests.delete(f"{API_BASE}/runs", timeout=2)
                    if r_del.status_code == 200:
                        cleared = True
                except Exception:
                    pass
            if not cleared:
                db = SessionLocal()
                try:
                    runs = db.query(Run).all()
                    for r in runs:
                        if r.output_video_path and os.path.exists(r.output_video_path):
                            try:
                                os.remove(r.output_video_path)
                            except OSError:
                                pass
                    db.query(ClassCount).delete()
                    db.query(Run).delete()
                    db.commit()
                    cleared = True
                finally:
                    db.close()
            if cleared:
                st.success("All past runs and output files deleted successfully!")
                st.rerun()

    # Load runs directly from SQLite with API fallback for highest reliability
    db = SessionLocal()
    runs_list = []
    try:
        db_runs = db.query(Run).order_by(Run.created_at.desc()).all()
        runs_list = [
            {
                "id": r.id,
                "filename": r.filename,
                "status": r.status,
                "model_path": r.model_path,
                "confidence_threshold": r.confidence_threshold,
                "created_at": str(r.created_at) if r.created_at else "",
                "total_frames": r.total_frames,
                "total_time_sec": r.total_time_sec,
                "avg_inference_ms": r.avg_inference_ms,
                "output_video_path": r.output_video_path,
                "class_counts": [
                    {"class_name": cc.class_name, "in_count": cc.in_count, "out_count": cc.out_count}
                    for cc in r.class_counts
                    if cc.class_name.lower() != "bicycle"
                ],
            }
            for r in db_runs
        ]
    except Exception:
        if backend_online:
            try:
                r_api = requests.get(f"{API_BASE}/runs", timeout=2)
                if r_api.status_code == 200:
                    runs_list = r_api.json()
            except Exception:
                runs_list = []
    finally:
        db.close()

    if not runs_list:
        st.info("No past runs found in database. Process a video in Tab 1 to see records here.")
    else:
        run_options = {}
        for r in runs_list:
            dt_str = str(r.get("created_at", ""))[:19].replace("T", " ")
            run_options[f"Run #{r['id']} — {r['filename']} ({r['status'].upper()} | {dt_str})"] = r['id']

        selected_run_label = st.selectbox("Select a Run to Inspect", list(run_options.keys()))
        selected_run_id = run_options[selected_run_label]
        run_detail = next((r for r in runs_list if r['id'] == selected_run_id), runs_list[0])

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Model Used</div>
                    <div class="metric-value" style="font-size:1.1rem; color:#38bdf8;">{run_detail.get('model_path', 'default')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Processed Frames</div>
                    <div class="metric-value">{run_detail.get('total_frames') or 'N/A'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Execution Time</div>
                    <div class="metric-value">{run_detail.get('total_time_sec') or 'N/A'} <span style="font-size:0.9rem;">sec</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Latency</div>
                    <div class="metric-value">{run_detail.get('avg_inference_ms') or 'N/A'} <span style="font-size:0.9rem;">ms</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        r_vcol, r_tcol = st.columns([3, 2])
        with r_vcol:
            st.subheader("📼 Video Playback")
            video_path = run_detail.get("output_video_path") or f"outputs/run_{selected_run_id}_output.mp4"
            if video_path and os.path.exists(video_path):
                st.video(video_path)
            elif backend_online and run_detail.get("status") == "complete":
                st.video(f"{API_BASE}/runs/{selected_run_id}/video")
            else:
                st.info("Video file not available on disk for this run.")

        with r_tcol:
            st.subheader("📊 Class Tallies")
            if run_detail.get("class_counts"):
                df_past = pd.DataFrame([
                    {
                        "Type": cc["class_name"].upper(),
                        "IN (South)": cc["in_count"],
                        "OUT (North)": cc["out_count"],
                        "Total": cc["in_count"] + cc["out_count"],
                    }
                    for cc in run_detail["class_counts"]
                ])
                st.dataframe(df_past, use_container_width=True, hide_index=True)

                buf = io.StringIO()
                cw = csv.writer(buf)
                cw.writerow(["run_id", selected_run_id])
                cw.writerow(["filename", run_detail.get("filename")])
                cw.writerow(["model", run_detail.get("model_path")])
                cw.writerow(["total_frames", run_detail.get("total_frames")])
                cw.writerow(["total_time_sec", run_detail.get("total_time_sec")])
                cw.writerow([])
                cw.writerow(["class", "in", "out", "total"])
                for cc in run_detail["class_counts"]:
                    cw.writerow([cc["class_name"], cc["in_count"], cc["out_count"], cc["in_count"] + cc["out_count"]])

                col_btn_dl, col_btn_del = st.columns([3, 2])
                with col_btn_dl:
                    st.download_button(
                        "📥 Export Run CSV",
                        data=buf.getvalue(),
                        file_name=f"run_{selected_run_id}_report.csv",
                        mime="text/csv",
                        key=f"dl_csv_{selected_run_id}",
                        use_container_width=True,
                    )
                with col_btn_del:
                    if st.button("🗑️ Delete Run", key=f"del_single_{selected_run_id}", type="secondary", use_container_width=True):
                        db = SessionLocal()
                        try:
                            r_to_del = db.get(Run, selected_run_id)
                            if r_to_del:
                                if r_to_del.output_video_path and os.path.exists(r_to_del.output_video_path):
                                    try:
                                        os.remove(r_to_del.output_video_path)
                                    except OSError:
                                        pass
                                db.delete(r_to_del)
                                db.commit()
                                st.success(f"Run #{selected_run_id} deleted!")
                                st.rerun()
                        finally:
                            db.close()


# ================= Tab 3: Global Analytics =================
with tab3:
    st.subheader("Aggregated Traffic Intelligence Across All Runs")

    # Load summary directly from SQLite with API fallback for 100% reliability
    db = SessionLocal()
    summary_data = []
    total_completed_runs = 0
    try:
        total_completed_runs = db.query(Run).filter(Run.status == "complete").count()
        rows = (
            db.query(ClassCount.class_name, sqla_func.sum(ClassCount.in_count), sqla_func.sum(ClassCount.out_count))
            .join(Run)
            .filter(Run.status == "complete")
            .filter(sqla_func.lower(ClassCount.class_name) != "bicycle")
            .group_by(ClassCount.class_name)
            .all()
        )
        summary_data = [
            {"class_name": name, "total_in": int(in_sum or 0), "total_out": int(out_sum or 0)}
            for name, in_sum, out_sum in rows
        ]
    except Exception:
        if backend_online:
            try:
                r_sum = requests.get(f"{API_BASE}/analytics/summary", timeout=2)
                if r_sum.status_code == 200:
                    summary_data = r_sum.json()
            except Exception:
                summary_data = []
    finally:
        db.close()

    if not summary_data or total_completed_runs == 0:
        st.info("No aggregated traffic telemetry recorded yet. Process video runs in Tab 1 to see global analytics here.")
    else:
        total_inflow = sum(item["total_in"] for item in summary_data)
        total_outflow = sum(item["total_out"] for item in summary_data)
        grand_total = total_inflow + total_outflow

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Completed Runs</div>
                    <div class="metric-value" style="color: #38bdf8;">{total_completed_runs}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Traffic Flow</div>
                    <div class="metric-value">{grand_total}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Incoming (Inbound)</div>
                    <div class="metric-value metric-badge-in">⬇ {total_inflow}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Total Departing (Outbound)</div>
                    <div class="metric-value metric-badge-out">⬆ {total_outflow}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        df_summary = pd.DataFrame([
            {
                "Vehicle Type": item["class_name"].upper(),
                "Incoming (Inbound)": item["total_in"],
                "Departing (Outbound)": item["total_out"],
                "Total Traffic": item["total_in"] + item["total_out"],
            }
            for item in summary_data
        ])

        ch1, ch2 = st.columns(2)
        with ch1:
            st.subheader("Vehicle Breakdown Table")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

        with ch2:
            st.subheader("Directional Distribution Chart")
            chart_df = df_summary.set_index("Vehicle Type")[["Incoming (Inbound)", "Departing (Outbound)"]]
            st.bar_chart(chart_df)


# ================= Tab 4: Live Local Webcam =================
with tab4:
    st.subheader("Live Dual-Boundary Camera Telemetry")
    st.caption("Streams directly from your connected video input (Camera 0) with real-time corridor gate counting.")

    cam_run = st.checkbox("🔌 Activate Camera Stream", key="webcam_toggle")
    cam_placeholder = st.empty()
    cam_stats_placeholder = st.empty()

    if cam_run:
        model = load_model(selected_model)
        v_classes = get_vehicle_classes(model)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access camera device 0. Please verify device permissions.")
        else:
            ret, first_frame = cap.read()
            if ret:
                height = first_frame.shape[0]
                boundaries, counts, counted_ids, track_history, track_state = make_tracker_state(
                    height, v_classes, inbound_ratio=inbound_ratio, outbound_ratio=outbound_ratio
                )

                while cam_run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera stream interrupted.")
                        break

                    annotated = process_frame(
                        model=model,
                        frame=frame,
                        vehicle_classes=v_classes,
                        conf_threshold=confidence,
                        boundaries=boundaries,
                        counts=counts,
                        counted_ids=counted_ids,
                        track_history=track_history,
                        track_state=track_state,
                        draw_trails=True,
                        show_hud=True,
                    )

                    cam_placeholder.image(annotated, channels="BGR", use_container_width=True)

                    cam_stats_df = pd.DataFrame([
                        {"Type": k.upper(), "IN": v["in"], "OUT": v["out"], "Total": v["in"] + v["out"]}
                        for k, v in counts.items()
                        if (v["in"] + v["out"]) > 0 or len(counts) <= 6
                    ])
                    cam_stats_placeholder.dataframe(cam_stats_df, use_container_width=True, hide_index=True)

            cap.release()
    else:
        st.info("Camera stream is currently inactive. Toggle the checkbox above to begin live feed.")
