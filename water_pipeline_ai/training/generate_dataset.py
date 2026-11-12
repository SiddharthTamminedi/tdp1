"""
==============================================================================
Project: AI-Based Water Pipeline Fault Detection System
File: training/generate_dataset.py
Purpose: Generates a realistic synthetic time-series dataset for training and
         evaluating machine learning fault detection models.
==============================================================================

IMPORTANT NOTICE:
This dataset is SYNTHETIC training data created through physics-guided stochastic
simulation (autoregressive Ornstein-Uhlenbeck processes, cavitation perturbations,
and constriction modeling). While engineered to replicate real fluid dynamic
signatures (turbulence, throttling, line surge, zero-flow states), ACTUAL FAULT
DETECTION IN PRODUCTION PIPELINES MUST BE RIGOROUSLY VALIDATED USING EMPIRICAL,
PHYSICAL TEST RIG EXPERIMENTS WITH GROUND-TRUTH FLOW & PRESSURE LOGGERS.
==============================================================================
"""

import os
import numpy as np
import pandas as pd

def generate_synthetic_pipeline_data(
    target_samples=15000,
    random_seed=42,
    output_path="dataset/water_pipeline_dataset.csv"
):
    """
    Generates realistic time-series sensor sequences for 5 pipeline states:
      1. NORMAL: Stable nominal flow with natural fluid turbulence.
      2. LEAKAGE: Elevated flow rate representing uncontrolled downstream discharge/rupture.
      3. BLOCKAGE: Choked/throttled flow with partial or near-complete constriction.
      4. NO_FLOW: Stagnant pipeline, 0 L/min, pump off or line isolated.
      5. ABNORMAL_FLOW: Water hammer, pump cavitation, air entrainment, surge oscillations.
    """
    rng = np.random.default_rng(random_seed)
    
    # 5 target classes
    classes = ["NORMAL", "LEAKAGE", "BLOCKAGE", "NO_FLOW", "ABNORMAL_FLOW"]
    samples_per_class = target_samples // len(classes)
    run_duration_sec = 100  # Each experiment run lasts 100 seconds
    runs_per_class = max(1, samples_per_class // run_duration_sec)
    
    all_runs = []
    global_run_id = 0

    print(f"[*] Generating synthetic pipeline time-series dataset...")
    print(f"[*] Target samples: ~{target_samples} across {len(classes)} classes.")
    print(f"[*] Runs per class: {runs_per_class} (Duration: {run_duration_sec}s per run)")

    for label in classes:
        for _ in range(runs_per_class):
            global_run_id += 1
            flow_series = np.zeros(run_duration_sec, dtype=float)

            if label == "NORMAL":
                # Domestic / commercial baseline flow: 2.8 - 4.5 L/min
                base_flow = rng.uniform(2.8, 4.2)
                noise_std = rng.uniform(0.08, 0.18)
                theta = 0.45  # Mean reversion rate
                
                current = base_flow + rng.normal(0, 0.1)
                for t in range(run_duration_sec):
                    # Ornstein-Uhlenbeck mean-reverting process
                    drift = theta * (base_flow - current)
                    shock = rng.normal(0, noise_std)
                    current = max(0.2, current + drift + shock)
                    flow_series[t] = current

            elif label == "LEAKAGE":
                # Pipeline breach / burst orifice: elevated flow rate (4.8 - 7.5 L/min)
                # with persistent high discharge and higher jitter
                leak_severity = rng.uniform(1.4, 2.5)
                nominal_base = rng.uniform(3.0, 4.0)
                leak_base = nominal_base * leak_severity
                noise_std = rng.uniform(0.25, 0.45)
                theta = 0.35
                
                # Start at leak flow or transition into leak
                current = leak_base + rng.normal(0, 0.2)
                for t in range(run_duration_sec):
                    # Slight pressure drop causing micro-wobbles
                    drift = theta * (leak_base - current)
                    shock = rng.normal(0, noise_std)
                    current = max(0.5, current + drift + shock)
                    flow_series[t] = current

            elif label == "BLOCKAGE":
                # Partial or severe constriction: flow drops to 0.1 - 1.4 L/min
                is_severe = rng.choice([True, False], p=[0.35, 0.65])
                target_flow = rng.uniform(0.02, 0.35) if is_severe else rng.uniform(0.5, 1.4)
                noise_std = rng.uniform(0.05, 0.18)
                theta = 0.50
                
                current = target_flow + rng.normal(0, 0.05)
                for t in range(run_duration_sec):
                    # Flutter due to throttled orifice
                    drift = theta * (target_flow - current)
                    shock = rng.normal(0, noise_std)
                    current = max(0.0, current + drift + shock)
                    # Occasional intermittent zero stall in severe blockage
                    if is_severe and rng.random() < 0.15:
                        current = 0.0
                    flow_series[t] = current

            elif label == "NO_FLOW":
                # Closed valve, dry pipe, pump turned off: 0 L/min
                for t in range(run_duration_sec):
                    # 98% exact 0.0, 2% electronic/hall sensor micro-noise (0.01 - 0.03 L/min)
                    flow_series[t] = 0.0 if rng.random() > 0.02 else rng.uniform(0.0, 0.03)

            elif label == "ABNORMAL_FLOW":
                # Water hammer, air slug cavitation, violent surge fluctuations
                mean_center = rng.uniform(2.5, 4.5)
                current = mean_center
                for t in range(run_duration_sec):
                    # Rapid oscillations and chaotic jumps
                    surge = rng.choice([-2.2, -1.5, 0.0, 1.8, 2.8], p=[0.15, 0.2, 0.3, 0.2, 0.15])
                    current = mean_center + surge + rng.normal(0, 0.8)
                    current = max(0.0, min(current, 9.5))  # Physical bounds of YF-S201
                    flow_series[t] = current

            # Construct DataFrame for this run
            run_df = pd.DataFrame({
                "run_id": global_run_id,
                "time_sec": np.arange(run_duration_sec),
                "flow_rate": np.round(flow_series, 3),
                "label": label
            })
            all_runs.append(run_df)

    # Concatenate all runs
    df_raw = pd.concat(all_runs, ignore_index=True)

    print("[*] Computing physical and rolling statistical features...")
    
    # Feature Engineering per run (prevent boundary bleeding across runs)
    feature_dfs = []
    for run_id, group in df_raw.groupby("run_id"):
        g = group.copy()
        
        # Calibration constant: YF-S201 produces ~7.5 pulses per second per 1 L/min
        g["pulse_count"] = np.round(g["flow_rate"] * 7.5).astype(int)
        
        # Cumulative volume in Liters: (flow_rate / 60) * dt (dt = 1 sec)
        g["total_volume"] = np.round((g["flow_rate"] / 60.0).cumsum(), 4)
        
        # Prior flow rate (lag 1)
        g["prev_flow_rate"] = g["flow_rate"].shift(1).bfill()
        
        # Flow rate delta: flow_rate - prev_flow_rate
        g["flow_rate_diff"] = np.round(g["flow_rate"] - g["prev_flow_rate"], 3)
        
        # Rolling window features (Window = 5 samples)
        rolling_w5 = g["flow_rate"].rolling(window=5, min_periods=1)
        g["avg_flow_rate"] = np.round(rolling_w5.mean(), 3)
        g["min_flow_rate"] = np.round(rolling_w5.min(), 3)
        g["max_flow_rate"] = np.round(rolling_w5.max(), 3)
        g["std_flow_rate"] = np.round(rolling_w5.std().fillna(0.0), 3)
        g["flow_rate_var"] = np.round(rolling_w5.var().fillna(0.0), 3)
        
        # Rolling window features (Window = 10 samples for broader baseline)
        rolling_w10 = g["flow_rate"].rolling(window=10, min_periods=1)
        g["rolling_avg_flow_rate"] = np.round(rolling_w10.mean(), 3)
        
        feature_dfs.append(g)

    df_final = pd.concat(feature_dfs, ignore_index=True)

    # Reorder columns logically
    ordered_columns = [
        "run_id",
        "time_sec",
        "flow_rate",
        "avg_flow_rate",
        "min_flow_rate",
        "max_flow_rate",
        "std_flow_rate",
        "flow_rate_var",
        "pulse_count",
        "total_volume",
        "flow_rate_diff",
        "prev_flow_rate",
        "rolling_avg_flow_rate",
        "label"
    ]
    df_final = df_final[ordered_columns]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"\n[+] Synthetic dataset successfully generated!")
    print(f"    - Output file: {output_path}")
    print(f"    - Total samples: {len(df_final):,} rows")
    print(f"    - Distinct experiment runs: {global_run_id}")
    print(f"    - Class distribution:")
    for lbl, count in df_final["label"].value_counts().items():
        print(f"      * {lbl:15s}: {count:,} samples ({count/len(df_final):.1%})")

    return df_final


if __name__ == "__main__":
    import sys
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_file = os.path.join(base_dir, "dataset", "water_pipeline_dataset.csv")
    generate_synthetic_pipeline_data(target_samples=15000, output_path=out_file)

