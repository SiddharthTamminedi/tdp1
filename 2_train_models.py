# STEP 2: Train TWO models from a CSV file.
#   Model A: Decision Tree  -> names the fault (NORMAL/LEAKAGE/BLOCKAGE/ABNORMAL)
#   Model B: Isolation Forest -> anomaly detector. Learns ONLY what NORMAL looks
#            like, then flags anything that looks different ("something is odd").
# Change CSV_FILE to your real data file later.
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

CSV_FILE = "synthetic_pipeline_data.csv"
FEATURES = ["mean_flow", "std_flow", "min_flow", "max_flow"]
LABELS = ["NORMAL", "LEAKAGE", "BLOCKAGE", "ABNORMAL"]

df = pd.read_csv(CSV_FILE)
X, y = df[FEATURES], df["label"]

# Split by whole experiment RUN so test runs are never seen in training
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
tr, te = next(splitter.split(X, y, groups=df["run_id"]))
X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

# ---- Model A: fault classifier ----
clf = DecisionTreeClassifier(max_depth=4, random_state=1)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
print("MODEL A - Decision Tree accuracy:", round(accuracy_score(y_te, pred), 3))
print("Rows = true, columns = predicted, order:", LABELS)
print(confusion_matrix(y_te, pred, labels=LABELS))

# ---- Model B: anomaly detector (trained on NORMAL only) ----
normal_train = X_tr[y_tr == "NORMAL"]
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=1)
iso.fit(normal_train)

flag = iso.predict(X_te) == -1            # True = anomaly
is_normal = (y_te == "NORMAL").values
print("\nMODEL B - Isolation Forest (anomaly detector)")
print("Normal windows wrongly flagged :", f"{flag[is_normal].mean():.1%}")
print("Faulty windows correctly flagged:", f"{flag[~is_normal].mean():.1%}")

joblib.dump({"classifier": clf, "anomaly": iso, "features": FEATURES},
            "pipeline_models.joblib")
print("\nSaved pipeline_models.joblib")
