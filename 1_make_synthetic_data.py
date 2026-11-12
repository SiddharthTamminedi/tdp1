# STEP 1: Make a SYNTHETIC (made-up) dataset for practice.
# These numbers are my ASSUMPTIONS, not real sensor readings.
# Replace with your own measured CSV before your college demo.
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (typical flow in L/min, how much the flow wobbles)
conditions = {
    "NORMAL":   (2.0, 0.15),
    "LEAKAGE":  (3.2, 0.25),   # extra water leaving through the hole
    "BLOCKAGE": (0.9, 0.15),   # squeezed pipe -> less flow
    "ABNORMAL": (2.0, 1.00),   # unstable, jumping flow
}

rows = []
run_id = 0
for label, (base, wobble) in conditions.items():
    for _ in range(20):                      # 20 separate experiment runs
        run_id += 1
        run_base = base + rng.normal(0, 0.4)
        run_wobble = abs(wobble + rng.normal(0, 0.05))
        flow = np.clip(rng.normal(run_base, run_wobble, 60), 0, None)  # 60 seconds
        for start in range(0, 60, 5):        # 5-second windows
            w = flow[start:start + 5]
            rows.append([run_id, round(w.mean(), 3), round(w.std(), 3),
                         round(w.min(), 3), round(w.max(), 3), label])

df = pd.DataFrame(rows, columns=["run_id", "mean_flow", "std_flow",
                                 "min_flow", "max_flow", "label"])
df.to_csv("synthetic_pipeline_data.csv", index=False)
print(df.head())
print(df["label"].value_counts())
print("Saved synthetic_pipeline_data.csv")
