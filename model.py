import os
from pathlib import Path
from ultralytics import YOLO

# Default model path (YOLOv8 Nano for fastest inference & accuracy)
MODEL_PATH = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.4

# Standard 4 motorized vehicle classes detected by pretrained YOLO models (COCO)
# Explicitly excludes non-motorized / unexpected classes like bicycle
STANDARD_VEHICLE_NAMES = {"car", "truck", "bus", "motorcycle"}

# COCO benchmark IDs for the 4 motorized vehicle types
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Standard 6 vehicle classes for UVH-26 custom dataset taxonomy
UVH26_DEFAULT_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "motorcycle",
    5: "rikshaw",
}


def get_available_models(directory: str = ".") -> list[str]:
    """
    Returns a prioritized list of all .pt model weight files found in the directory.
    Places default yolov8n.pt and custom vehicle models at the top.
    """
    dir_path = Path(directory)
    pt_files = [f.name for f in dir_path.glob("*.pt")]

    # Sort with primary recommended models first
    priority = ["yolov8n.pt", "uvh26_vehicle_model.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"]
    sorted_files = []
    for p in priority:
        if p in pt_files and p not in sorted_files:
            sorted_files.append(p)
    for f in sorted(pt_files):
        if f not in sorted_files:
            sorted_files.append(f)

    return sorted_files if sorted_files else [MODEL_PATH]


def load_model(model_name_or_path: str = MODEL_PATH) -> YOLO:
    """
    Loads a YOLO model from the given path or filename.
    Supports backward-compatibility aliases (e.g. mapping 'best.pt' to 'uvh26_vehicle_model.pt').
    """
    path = str(model_name_or_path).strip()

    # Backward compatibility alias
    if path in ("best.pt", "./best.pt") and not os.path.exists(path):
        if os.path.exists("uvh26_vehicle_model.pt"):
            path = "uvh26_vehicle_model.pt"

    if not os.path.exists(path) and os.path.exists(os.path.join(".", path)):
        path = os.path.join(".", path)

    return YOLO(path)


def get_vehicle_classes(model: YOLO, preset: str = "auto", *args, **kwargs) -> dict[int, str]:
    """
    Extracts the active target vehicle classes for detection and tracking.
    
    Modes:
      - 'auto': 
          * If model has <= 10 classes (custom UVH-26 or custom vehicle dataset), uses all model classes directly.
          * If model has > 10 classes (e.g., 80-class COCO pretrained YOLOv8/YOLO11), strictly filters to the 
            4 motorized vehicle classes: Car, Motorcycle, Bus, Truck (excluding bicycle and other non-vehicle objects).
      - 'coco_4_class': Strictly returns the 4 standard motorized vehicle classes.
      - 'uvh26_6_class' or 'custom': Returns all custom model classes directly.
    """
    model_names = model.names
    if not isinstance(model_names, dict):
        model_names = {i: name for i, name in enumerate(model_names)}

    preset_clean = str(preset).strip().lower()

    if preset_clean == "coco_4_class":
        vehicle_classes = {}
        for cls_id, cls_name in model_names.items():
            name_clean = str(cls_name).strip().lower().replace("-", "_").replace(" ", "_")
            if name_clean in STANDARD_VEHICLE_NAMES:
                vehicle_classes[int(cls_id)] = str(cls_name)
        return vehicle_classes if vehicle_classes else dict(COCO_VEHICLE_CLASSES)

    if preset_clean in ("uvh26_6_class", "custom"):
        return {int(k): str(v) for k, v in model_names.items()}

    # Preset 'auto'
    if len(model_names) <= 10:
        # Custom-trained vehicle model (e.g. UVH-26 6-class model)
        return {int(k): str(v) for k, v in model_names.items()}

    # Standard COCO pretrained model (> 10 classes) -> Strictly match 4 motorized classes
    vehicle_classes = {}
    for cls_id, cls_name in model_names.items():
        name_clean = str(cls_name).strip().lower().replace("-", "_").replace(" ", "_")
        if name_clean in STANDARD_VEHICLE_NAMES:
            vehicle_classes[int(cls_id)] = str(cls_name)

    return vehicle_classes if vehicle_classes else dict(COCO_VEHICLE_CLASSES)


# Default fallback vehicle classes dictionary for backward compatibility (Strict 4 classes)
try:
    _default_model = load_model(MODEL_PATH)
    VEHICLE_CLASSES = get_vehicle_classes(_default_model)
except Exception:
    VEHICLE_CLASSES = dict(COCO_VEHICLE_CLASSES)


# Model Cache: Prevents redundant model loading in Streamlit rerun cycles
_model_cache = {}


def load_model_cached(model_name_or_path: str = MODEL_PATH) -> YOLO:
    """
    Loads a YOLO model with caching to prevent redundant loading in Streamlit.
    
    In Streamlit apps, functions are re-executed on every interaction. This function
    caches loaded models by path to avoid unnecessary GPU memory allocation and initialization.
    
    Args:
        model_name_or_path: Model file path or name
        
    Returns:
        Cached YOLO model instance
    """
    if model_name_or_path not in _model_cache:
        _model_cache[model_name_or_path] = load_model(model_name_or_path)
    return _model_cache[model_name_or_path]