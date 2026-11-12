# STEP 3: Live display on the laptop.
# Reads "FLOW,x.xx" lines from the ESP32, groups 5 seconds into a window,
# then shows the fault prediction AND the anomaly check.
#
# Real ESP32:   python 3_live_monitor.py COM3      (pip install pyserial first)
# No hardware:  python 3_live_monitor.py --simulate
import sys, time
import numpy as np
import pandas as pd
import joblib

WINDOW = 5
bundle = joblib.load("pipeline_models.joblib")
clf, iso, FEATURES = bundle["classifier"], bundle["anomaly"], bundle["features"]

def readings_from_esp32(port):
    import serial                                   # pip install pyserial
    with serial.Serial(port, 115200, timeout=2) as s:
        while True:
            line = s.readline().decode(errors="ignore").strip()
            if line.startswith("FLOW,"):
                try:
                    yield float(line.split(",")[1])
                except ValueError:
                    pass

def readings_simulated():
    rng = np.random.default_rng(7)
    plan = [("NORMAL", 2.0, 0.15), ("LEAKAGE", 3.2, 0.25),
            ("BLOCKAGE", 0.9, 0.15), ("ABNORMAL", 2.0, 1.0)]
    for name, base, wob in plan:
        print(f"[simulator now creating: {name}]")
        for _ in range(15):
            time.sleep(0.2)
            yield max(0.0, rng.normal(base, wob))

source = readings_simulated() if "--simulate" in sys.argv else \
         readings_from_esp32(sys.argv[1] if len(sys.argv) > 1 else "COM3")

buf = []
for flow in source:
    buf.append(flow)
    if len(buf) < WINDOW:
        continue
    w = np.array(buf); buf = []
    x = pd.DataFrame([[w.mean(), w.std(), w.min(), w.max()]], columns=FEATURES)
    fault = clf.predict(x)[0]
    odd = iso.predict(x)[0] == -1
    print("===================================")
    print("IOT PIPELINE FAULT DETECTION\n")
    print(f"Flow Rate: {w.mean():.2f} L/min")
    print(f"AI Prediction: {fault}")
    print(f"Anomaly Check: {'UNUSUAL PATTERN' if odd else 'looks normal'}")
    if fault == "NORMAL" and not odd:
        print("Pipeline Status: HEALTHY")
    else:
        print(f"Warning: Possible {fault} condition!")
    print("===================================\n")
