from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "machine_load",
    "spindle_speed",
    "temperature",
    "vibration",
    "energy_consumption",
    "material_hardness",
    "queue_length",
    "wip",
    "operator_efficiency",
    "maintenance_index",
    "order_pressure",
    "environment_humidity",
]

TARGET_COLUMNS = [
    "production_time",
    "equipment_utilization",
    "quality_fluctuation",
    "unit_cost",
    "delivery_capability",
]


def generate_manufacturing_data(
    n_samples: int = 12000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    daily = np.sin(2 * np.pi * t / 288)
    weekly = np.sin(2 * np.pi * t / (288 * 7))
    slow_trend = t / n_samples

    machine_load = np.clip(
        0.68 + 0.14 * daily + 0.08 * weekly
        + 0.07 * rng.normal(size=n_samples),
        0.2,
        1.0,
    )
    spindle_speed = (
        1450 + 220 * machine_load + 70 * daily
        + 45 * rng.normal(size=n_samples)
    )
    temperature = (
        34 + 16 * machine_load + 2.5 * daily
        + 1.3 * rng.normal(size=n_samples)
    )
    vibration = np.clip(
        1.6 + 2.1 * machine_load + 0.012 * temperature
        + 0.35 * rng.normal(size=n_samples),
        0,
        None,
    )
    energy = (
        55 + 38 * machine_load + 0.006 * spindle_speed
        + 3.0 * rng.normal(size=n_samples)
    )
    material_hardness = (
        58 + 4 * np.sin(2 * np.pi * t / 900)
        + 2 * rng.normal(size=n_samples)
    )
    queue_length = np.clip(
        9 + 14 * machine_load + 4 * order_wave(t)
        + 3 * rng.normal(size=n_samples),
        0,
        None,
    )
    wip = np.clip(
        18 + 1.7 * queue_length + 5 * weekly
        + 4 * rng.normal(size=n_samples),
        0,
        None,
    )
    operator_efficiency = np.clip(
        0.87 - 0.05 * daily - 0.04 * slow_trend
        + 0.025 * rng.normal(size=n_samples),
        0.55,
        1.0,
    )
    maintenance_index = np.clip(
        0.95 - 0.55 * slow_trend
        + 0.07 * np.sin(2 * np.pi * t / 1600)
        + 0.035 * rng.normal(size=n_samples),
        0.15,
        1.0,
    )
    order_pressure = np.clip(
        0.55 + 0.20 * weekly + 0.12 * order_wave(t)
        + 0.06 * rng.normal(size=n_samples),
        0,
        1,
    )
    humidity = np.clip(
        52 + 13 * np.sin(2 * np.pi * t / 288 + 0.8)
        + 4 * rng.normal(size=n_samples),
        20,
        90,
    )

    production_time = (
        42
        + 11 * machine_load
        + 0.17 * queue_length
        + 0.13 * material_hardness
        + 5 * vibration
        - 13 * operator_efficiency
        + 1.2 * rng.normal(size=n_samples)
    )
    utilization = np.clip(
        0.30
        + 0.65 * machine_load
        - 0.004 * queue_length
        + 0.12 * maintenance_index
        + 0.025 * rng.normal(size=n_samples),
        0,
        1,
    )
    quality_fluctuation = np.clip(
        0.8
        + 0.30 * vibration
        + 0.035 * np.abs(temperature - 43)
        + 0.018 * np.abs(humidity - 50)
        + 0.25 * (1 - maintenance_index)
        + 0.10 * rng.normal(size=n_samples),
        0,
        None,
    )
    unit_cost = (
        18
        + 0.16 * energy
        + 0.18 * production_time
        + 4.0 * quality_fluctuation
        + 0.7 * rng.normal(size=n_samples)
    )
    delivery_capability = np.clip(
        1.05
        - 0.010 * production_time
        - 0.006 * queue_length
        - 0.12 * order_pressure
        + 0.28 * utilization
        + 0.02 * rng.normal(size=n_samples),
        0,
        1,
    )

    frame = pd.DataFrame({
        "timestamp": pd.date_range(
            "2024-01-01",
            periods=n_samples,
            freq="5min",
        ),
        "machine_load": machine_load,
        "spindle_speed": spindle_speed,
        "temperature": temperature,
        "vibration": vibration,
        "energy_consumption": energy,
        "material_hardness": material_hardness,
        "queue_length": queue_length,
        "wip": wip,
        "operator_efficiency": operator_efficiency,
        "maintenance_index": maintenance_index,
        "order_pressure": order_pressure,
        "environment_humidity": humidity,
        "production_time": production_time,
        "equipment_utilization": utilization,
        "quality_fluctuation": quality_fluctuation,
        "unit_cost": unit_cost,
        "delivery_capability": delivery_capability,
    })
    return frame


def order_wave(t: np.ndarray) -> np.ndarray:
    return (np.sin(2 * np.pi * t / 600) > 0.65).astype(float)


if __name__ == "__main__":
    output = Path("data/manufacturing.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    data = generate_manufacturing_data()
    data.to_csv(output, index=False)
    print(f"Generated {len(data)} rows: {output}")
