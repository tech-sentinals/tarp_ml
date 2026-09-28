"""
simulate.py - Source-informed four-zone intervention dataset generator.

Each row is one decision cycle: pre-intervention state + proposed micro-intervention
+ generated post-intervention response + reference outcome label.

Post-intervention columns (dC_protected, command_achievement_pct) are used ONLY
for labelling and validation. They are never given to the classifier.
"""
import numpy as np
import pandas as pd

ACTUATORS = ["fan", "damper", "valve", "pump", "door"]
# relative ability of each actuator type to move the hazard (per unit command)
ACT_GAIN = {"fan": 1.00, "damper": 0.85, "valve": 0.70, "pump": 0.65, "door": 0.90}

LABELS = ["Protective", "Harmful", "Ineffective", "Inconsistent"]


def label_record(dC, airflow_toward, sensor_fault, achievement):
    """Reference-label rules (prototype thresholds, configurable)."""
    if sensor_fault == 1 or achievement < 45:
        return "Inconsistent"
    if dC > 6 or (airflow_toward == 1 and dC > 1):
        return "Harmful"
    if dC < -5:
        return "Protective"
    return "Ineffective"


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    source_zone = rng.integers(1, 5, n)
    offset = rng.integers(1, 4, n)                       # guarantees protected != source
    protected_zone = (source_zone - 1 + offset) % 4 + 1
    hop_distance = np.abs(source_zone - protected_zone)  # zones laid out as a corridor 1-2-3-4

    initial_conc = rng.uniform(5, 60, n)
    airflow_toward = rng.binomial(1, 0.55, n)
    spatial_gradient = rng.normal(0.9 * airflow_toward - 0.1, 1.0, n)
    temporal_gradient = rng.normal(0.5 * airflow_toward - 0.1, 1.0, n)
    airflow_speed = rng.uniform(0.2, 3.0, n)
    pressure_diff = rng.normal(0, 15, n)
    oxygen_pct = rng.uniform(17.0, 21.0, n)
    door_open = rng.binomial(1, 0.35, n)
    blocked_path = rng.binomial(1, 0.20, n)
    sensor_fault = rng.binomial(1, 0.07, n)
    actuator_fault = rng.binomial(1, 0.12, n)
    comm_delay_ms = np.clip(rng.exponential(130, n) + 20, 20, 900)
    actuator = rng.choice(ACTUATORS, n)
    magnitude_pct = rng.uniform(10, 60, n)
    duration_s = rng.uniform(2, 20, n)
    actuator_trust = rng.beta(5, 2, n)
    occupancy = rng.binomial(1, 0.6, n)
    route_exposure = rng.binomial(1, 0.4, n)

    # ---- generated physical response (hidden from the classifier) ----
    achievement = (100
                   - actuator_fault * rng.uniform(35, 80, n)
                   - (1 - actuator_trust) * 25
                   - comm_delay_ms / 45
                   + rng.normal(0, 7, n))
    achievement = np.clip(achievement, 0, 100)

    gain = np.array([ACT_GAIN[a] for a in actuator])
    effort = (magnitude_pct / 100) * (achievement / 100) * np.sqrt(duration_s / 10) * gain
    effort *= np.where(blocked_path == 1, 0.5, 1.0)
    proximity = 1.0 / hop_distance

    # direction the action actually pushes the hazard relative to the protected zone
    # (+1 toward, -1 away). Unknown open doors can flip the real path.
    flip_p = 0.04 + 0.10 * door_open + 0.04 * (pressure_diff > 15)
    direction = np.where(airflow_toward == 1, 1, -1)
    direction = np.where(rng.random(n) < flip_p, -direction, direction)

    dC_true = (direction * effort * 30 * proximity * (0.6 + initial_conc / 60)
               + 1.4 * spatial_gradient + 1.0 * temporal_gradient
               + rng.normal(0, 3.0, n))
    # a faulty sensor reports a biased measurement
    dC_measured = dC_true + sensor_fault * rng.normal(0, 8, n)

    df = pd.DataFrame({
        "source_zone": source_zone, "protected_zone": protected_zone,
        "hop_distance": hop_distance,
        "initial_hazard_concentration": initial_conc.round(3),
        "spatial_gradient": spatial_gradient.round(4),
        "temporal_gradient": temporal_gradient.round(4),
        "airflow_speed_mps": airflow_speed.round(3),
        "airflow_direction_to_protected": airflow_toward,
        "pressure_diff_pa": pressure_diff.round(3),
        "oxygen_pct": oxygen_pct.round(3),
        "door_open": door_open, "blocked_path": blocked_path,
        "sensor_fault": sensor_fault, "actuator_fault": actuator_fault,
        "communication_delay_ms": comm_delay_ms.round(1),
        "actuator_type": actuator,
        "intervention_magnitude_pct": magnitude_pct.round(2),
        "intervention_duration_s": duration_s.round(2),
        "actuator_trust": actuator_trust.round(4),
        "occupancy": occupancy, "evacuation_route_exposure": route_exposure,
        # ---- post-intervention (label/validation only) ----
        "dC_protected": dC_measured.round(3),
        "command_achievement_pct": achievement.round(2),
    })
    df["outcome"] = [label_record(d, a, s, c) for d, a, s, c in
                     zip(df.dC_protected, df.airflow_direction_to_protected,
                         df.sensor_fault, df.command_achievement_pct)]
    df.insert(0, "record_id", [f"R{i:05d}" for i in range(n)])
    return df


if __name__ == "__main__":
    d = generate()
    d.to_csv("data/intervention_records.csv", index=False)
    print(d.outcome.value_counts())
