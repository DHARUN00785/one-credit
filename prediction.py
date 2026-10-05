from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np


class AgeRange(str, Enum):
    AGE_0_10 = "[0-10)"
    AGE_10_20 = "[10-20)"
    AGE_20_30 = "[20-30)"
    AGE_30_40 = "[30-40)"
    AGE_40_50 = "[40-50)"
    AGE_50_60 = "[50-60)"
    AGE_60_70 = "[60-70)"
    AGE_70_80 = "[70-80)"
    AGE_80_90 = "[80-90)"
    AGE_90_100 = "[90-100)"


AGE_OPTIONS = [age_range.value for age_range in AgeRange]
FEATURE_MEAN = np.array(
    [0.462, 6.088, 4.396, 43.096, 1.340, 16.022, 0.369, 0.198, 0.636, 7.423],
    dtype=np.float32,
)
FEATURE_STD = np.array(
    [0.499, 1.604, 2.985, 19.671, 1.705, 8.128, 1.267, 0.931, 1.263, 1.938],
    dtype=np.float32,
)


def scale_features(features: Sequence[int | float]) -> np.ndarray:
    if len(features) != len(FEATURE_MEAN):
        raise ValueError(f"Expected {len(FEATURE_MEAN)} features, received {len(features)}")

    raw = np.asarray([features], dtype=np.float32)
    return (raw - FEATURE_MEAN) / FEATURE_STD


def predict_readmission(model: Any, features: Sequence[int | float]) -> float:
    scaled_features = scale_features(features)
    return float(model.predict(scaled_features, verbose=0)[0][0])


def classify_risk(probability: float) -> tuple[str, str, str]:
    if probability >= 0.15:
        return "🔴 High Risk", "risk-high", (
            "This patient has a significantly elevated readmission risk. "
            "Consider enhanced discharge planning and close follow-up."
        )
    if probability >= 0.06:
        return "🟠 Moderate Risk", "risk-moderate", (
            "This patient shows moderate readmission risk indicators. "
            "Ensure a clear follow-up plan before discharge."
        )
    return "🟢 Low Risk", "risk-low", (
        "This patient currently shows low readmission risk indicators. "
        "Standard discharge protocols are appropriate."
    )