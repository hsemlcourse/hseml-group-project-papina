from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "final_model.pkl"

model = joblib.load(MODEL_PATH)
feature_names: List[str] = list(model.feature_names_in_)
class_map = {0: "Hike", 1: "Hold", 2: "Cut"}


def validate_features(data: Dict[str, float]) -> List[float]:
    missing = [name for name in feature_names if name not in data]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")
    extra = [name for name in data if name not in feature_names]
    if extra:
        raise ValueError(f"Unexpected features provided: {', '.join(extra)}")
    return [float(data[name]) for name in feature_names]


def predict(data: Dict[str, float]) -> Dict[str, object]:
    features = validate_features(data)
    proba = model.predict_proba([features])[0].tolist()
    pred_idx = int(model.predict([features])[0])
    return {
        "predicted_class_index": pred_idx,
        "predicted_class": class_map[pred_idx],
        "probabilities": {
            class_map[idx]: float(prob)
            for idx, prob in enumerate(proba)
        },
        "feature_names": feature_names,
        "feature_values": [float(value) for value in features],
    }


def get_model_info() -> Dict[str, object]:
    return {
        "model_type": type(model).__name__,
        "features": feature_names,
        "classes": [class_map[idx] for idx in sorted(class_map.keys())],
    }
