"""Generate a reproducible demonstration dataset for crop-yield modelling."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "crop_yield_sample.csv"

CROPS = {"maize": 3.0, "rice": 3.7, "cassava": 11.0, "soybean": 1.8, "sorghum": 2.1}
STATES = {"Benue": 0.35, "Kaduna": 0.15, "Kano": -0.20, "Kebbi": 0.10, "Niger": 0.30, "Oyo": 0.05, "Plateau": 0.20, "Taraba": 0.25}


def main() -> None:
    rng = np.random.default_rng(2026)
    OUTPUT.parent.mkdir(exist_ok=True)
    fields = ["crop", "state", "season", "rainfall_mm", "avg_temperature_c", "soil_ph", "soil_organic_matter_pct", "nitrogen_kg_ha", "farm_size_ha", "irrigated", "yield_tonnes_ha"]
    with open(OUTPUT, "w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for _ in range(600):
            crop = rng.choice(list(CROPS))
            state = rng.choice(list(STATES))
            season = rng.choice(["wet", "dry"], p=[0.73, 0.27])
            rainfall = np.clip(rng.normal(1150 if season == "wet" else 470, 210), 0, 2500)
            temperature = rng.normal(27.0, 2.3)
            soil_ph = np.clip(rng.normal(6.1, 0.65), 4.5, 8.0)
            organic_matter = np.clip(rng.normal(2.5, 0.85), 0.6, 5.8)
            nitrogen = np.clip(rng.normal(78, 28), 0, 160)
            farm_size = np.clip(rng.lognormal(0.55, 0.65), 0.4, 18)
            irrigated = int(season == "dry" and rng.random() < 0.58 or season == "wet" and rng.random() < 0.15)
            rain_effect = max(0, 1 - abs(rainfall - 1150) / 1450)
            heat_effect = max(0, 1 - abs(temperature - 27) / 10)
            fertility = 0.34 * organic_matter + 0.006 * nitrogen + 0.26 * max(0, 1 - abs(soil_ph - 6.3) / 2.2)
            yield_value = CROPS[crop] * (0.38 + 0.31 * rain_effect + 0.18 * heat_effect + fertility / 2.4 + 0.15 * irrigated + STATES[state] / 8)
            yield_value += rng.normal(0, CROPS[crop] * 0.075)
            writer.writerow({
                "crop": crop, "state": state, "season": season,
                "rainfall_mm": round(float(rainfall), 1), "avg_temperature_c": round(float(temperature), 1),
                "soil_ph": round(float(soil_ph), 2), "soil_organic_matter_pct": round(float(organic_matter), 2),
                "nitrogen_kg_ha": round(float(nitrogen), 1), "farm_size_ha": round(float(farm_size), 2),
                "irrigated": irrigated, "yield_tonnes_ha": round(float(max(0.2, yield_value)), 2),
            })
    print(f"Created {OUTPUT} with 600 rows.")


if __name__ == "__main__":
    main()

