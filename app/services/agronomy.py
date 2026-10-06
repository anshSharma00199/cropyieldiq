"""Dataset-range comparison only; does not provide agronomic prescriptions."""

LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}

UNITS = {
    "temperature": " °C",
    "humidity": "%",
    "rainfall": " mm",
}


def explain(crop: str, inputs: dict, crop_ranges: dict) -> dict:
    ranges = crop_ranges.get(crop, {})
    status = []
    advice = []

    for feature, value in inputs.items():
        feature_range = ranges.get(feature)
        if not feature_range:
            continue

        low = feature_range["p10"]
        high = feature_range["p90"]

        state = "low" if value < low else "high" if value > high else "ok"

        status.append(
            {
                "feature": feature,
                "label": LABELS[feature],
                "value": value,
                "status": state,
                "typical_low": low,
                "typical_median": feature_range["p50"],
                "typical_high": high,
            }
        )

        label = LABELS[feature]
        unit = UNITS.get(feature, "")

        if state == "low":
            advice.append(f"{label} is below the range observed for {crop} in this project's training data ({low}-{high}{unit}).")
        elif state == "high":
            advice.append(f"{label} is above the range observed for {crop} in this project's training data ({low}-{high}{unit}).")
        else:
            advice.append(f"{label} is within the range observed for {crop} in this project's training data ({low}-{high}{unit}).")

    advice.append(
        "These are training-data comparisons only. They do not establish "
        "a nutrient deficiency, diagnose a crop problem, or determine "
        "whether a treatment is needed."
    )

    return {
        "feature_status": status,
        "advice": advice,
    }
