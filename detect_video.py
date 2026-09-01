import argparse
import sys
import time
from pathlib import Path

from model import MODEL_PATH, CONFIDENCE_THRESHOLD, get_available_models
from processing import process_video_file


def print_progress(current_frame: int, total_frames: int, *args):
    pct = (current_frame / total_frames * 100) if total_frames > 0 else 0
    bar_len = 30
    filled_len = int(bar_len * current_frame // total_frames) if total_frames > 0 else 0
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    sys.stdout.write(f"\rProgress: [{bar}] {current_frame}/{total_frames} frames ({pct:.1f}%)")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection & extended dual-boundary corridor counting on video.")
    parser.add_argument("--input", "-i", default="data/vehicles_video.mp4", help="Path to input video file")
    parser.add_argument("--output", "-o", default="outputs/cli_video_output.mp4", help="Path to output video file")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help="Path or name of YOLO model weights")
    parser.add_argument("--conf", "-c", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold (0.1 - 0.9)")
    parser.add_argument("--inbound", type=float, default=0.40, help="Inbound boundary line position (0.1 to 0.9, default 0.40)")
    parser.add_argument("--outbound", type=float, default=0.68, help="Outbound boundary line position (0.2 to 0.95, default 0.68)")
    parser.add_argument("--stride", "-s", type=int, default=2, help="Frame stride / speed multiplier (1 = every frame, 2 = 2x speed [default], 3 = 3x speed)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input video not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("       VEHICLE DETECTION & TRAFFIC COUNTING (CLI)")
    print("=" * 65)
    print(f"Input video       : {input_path.resolve()}")
    print(f"Output video      : {output_path.resolve()}")
    print(f"Model             : {args.model}")
    print(f"Confidence        : {args.conf}")
    print(f"Speed Stride      : {args.stride}x ({'Process every frame' if args.stride == 1 else f'Process 1 in {args.stride} frames'})")
    print(f"Inbound Boundary  : {args.inbound * 100:.0f}% height")
    print(f"Outbound Boundary : {args.outbound * 100:.0f}% height")
    print("-" * 65)

    try:
        counts, stats = process_video_file(
            input_path=input_path,
            output_path=output_path,
            model_name_or_path=args.model,
            confidence=args.conf,
            inbound_ratio=args.inbound,
            outbound_ratio=args.outbound,
            frame_stride=args.stride,
            progress_callback=print_progress,
        )
        print("\n" + "-" * 65)
        print("Processing Complete!")
        print(f"Total Frames Processed : {stats['total_frames']}")
        print(f"Total Time             : {stats['total_time_sec']}s")
        print(f"Avg Inference Speed    : {stats['avg_inference_ms']}ms / frame")
        print("\nVehicle Count Breakdown:")
        print(f"{'Class':<15} {'Incoming (In)':<16} {'Departing (Out)':<16} {'Total':<10}")
        print("-" * 59)
        total_all = 0
        for cname, c in counts.items():
            total_cls = c["in"] + c["out"]
            total_all += total_cls
            print(f"{cname.capitalize():<15} {c['in']:<16} {c['out']:<16} {total_cls:<10}")
        print("-" * 59)
        print(f"{'TOTAL VEHICLES':<15} {'':<16} {'':<16} {total_all:<10}")
        print(f"\nAnnotated web-compatible video saved to:\n  {output_path.resolve()}\n")

    except Exception as e:
        print(f"\nError during video processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
