import argparse
from pathlib import Path

from model import load_model, get_vehicle_classes, MODEL_PATH, CONFIDENCE_THRESHOLD


def run_image_detection(image_path: str, model_path: str = MODEL_PATH, conf: float = CONFIDENCE_THRESHOLD, output_path: str = "outputs/output.jpg"):
    model = load_model(model_path)
    vehicle_classes = get_vehicle_classes(model)
    class_ids = list(vehicle_classes.keys())

    print(f"Loading model: {model_path}")
    print(f"Target vehicle classes: {vehicle_classes}")
    print(f"Running detection on {image_path} (confidence: {conf})...")

    results = model(
        image_path,
        classes=class_ids if class_ids else None,
        conf=conf,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    detected_counts = {}
    for r in results:
        r.save(filename=str(out_file))
        if r.boxes is not None:
            for cls_id, conf_val in zip(r.boxes.cls.int().cpu().tolist(), r.boxes.conf.float().cpu().tolist()):
                cname = vehicle_classes.get(cls_id, model.names.get(cls_id, f"class_{cls_id}"))
                detected_counts[cname] = detected_counts.get(cname, 0) + 1

    print("\n--- Detection Summary ---")
    if detected_counts:
        for cname, count in detected_counts.items():
            print(f"  {cname.capitalize()}: {count}")
    else:
        print("  No vehicles detected matching criteria.")
    print(f"Annotated output saved to: {out_file.resolve()}\n")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO vehicle detection on a single image.")
    parser.add_argument("--image", default="data/vehicle1.jpg", help="Path to input image")
    parser.add_argument("--model", default=MODEL_PATH, help="Path or name of YOLO model weights")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--output", default="outputs/output.jpg", help="Output path for annotated image")
    args = parser.parse_args()

    run_image_detection(args.image, args.model, args.conf, args.output)


if __name__ == "__main__":
    main()