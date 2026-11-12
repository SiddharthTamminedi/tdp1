"""
==============================================================================
Project: AI-Based Water Pipeline Fault Detection System
File: training/train_model.py
Purpose: Trains and benchmarks multiple classification algorithms (Random Forest,
         Decision Tree, Support Vector Machine, and XGBoost if available)
         using leakage-free group-aware cross-validation, saves the best model,
         scaler, and publication-ready evaluation charts.
==============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Optional XGBoost import with safe fallback
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False


def train_and_evaluate_models(
    dataset_path="dataset/water_pipeline_dataset.csv",
    models_dir="models",
    random_seed=42
):
    """
    Loads the synthetic dataset, splits without data leakage using GroupShuffleSplit,
    trains and evaluates candidate ML models, saves the champion model and preprocessors,
    and produces evaluation plots.
    """
    print("=" * 70)
    print("AI WATER PIPELINE FAULT DETECTION - MODEL BENCHMARKING & TRAINING")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}. Run generate_dataset.py first!")

    os.makedirs(models_dir, exist_ok=True)

    print(f"[*] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"    - Total records: {len(df):,}")
    print(f"    - Columns: {list(df.columns)}")

    # Define exact feature set required by system specifications
    features = [
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
        "rolling_avg_flow_rate"
    ]
    target_col = "label"
    group_col = "run_id"

    # Verify features exist
    for f in features:
        if f not in df.columns:
            raise KeyError(f"Required feature '{f}' missing from dataset!")

    X = df[features]
    y = df[target_col]
    groups = df[group_col]

    # Label encoding for integer mapping (essential for XGBoost and standardized metrics)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print(f"[*] Classes ({len(class_names)}): {class_names}")

    # =========================================================================
    # LEAKAGE-FREE TIME-SERIES GROUP SPLIT
    # =========================================================================
    # We split by whole simulation run (run_id) rather than random rows.
    # This prevents temporal autocorrelation / adjacent-sample data leakage.
    print(f"[*] Performing GroupShuffleSplit (75% Train, 25% Test on run_id)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))

    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    train_runs = groups.iloc[train_idx].nunique()
    test_runs = groups.iloc[test_idx].nunique()

    print(f"    - Train set: {len(X_train_raw):,} samples across {train_runs} independent runs")
    print(f"    - Test set : {len(X_test_raw):,} samples across {test_runs} independent runs")

    # =========================================================================
    # FEATURE SCALING
    # =========================================================================
    print(f"[*] Fitting StandardScaler on training set only (preventing scale leakage)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Save scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[+] Saved scaler to: {scaler_path}")

    # =========================================================================
    # MODEL DEFINITIONS
    # =========================================================================
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=120,
                max_depth=12,
                min_samples_split=4,
                random_state=random_seed,
                n_jobs=-1
            ),
            "use_scaled": False  # Tree models do not strictly require scaling
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=6,
                random_state=random_seed
            ),
            "use_scaled": False
        },
        "Support Vector Machine": {
            "model": SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                random_state=random_seed
            ),
            "use_scaled": True  # SVM requires scaled data
        }
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = {
            "model": XGBClassifier(
                n_estimators=120,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="mlogloss",
                random_state=random_seed
            ),
            "use_scaled": False
        }
        print("[+] XGBoost detected and added to candidate benchmark suite.")
    else:
        print("[i] XGBoost not available in environment; benchmarking Random Forest, Decision Tree, and SVM.")

    # =========================================================================
    # TRAINING & EVALUATION LOOP
    # =========================================================================
    results = []
    trained_models = {}
    test_predictions = {}
    test_probabilities = {}

    for name, config in models.items():
        print(f"\n---> Training: {name}...")
        clf = config["model"]
        use_scaled = config["use_scaled"]

        X_tr = X_train_scaled if use_scaled else X_train_raw.values
        X_te = X_test_scaled if use_scaled else X_test_raw.values

        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        
        # Probabilities
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_te)
        else:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        trained_models[name] = clf
        test_predictions[name] = y_pred
        test_probabilities[name] = y_proba

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Uses Scaled": use_scaled
        })

        print(f"     Accuracy : {acc:.4f}")
        print(f"     Precision: {prec:.4f}")
        print(f"     Recall   : {rec:.4f}")
        print(f"     F1-Score : {f1:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY TABLE:")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # Champion Selection
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    best_config = models[best_model_name]
    print(f"\n[★] Champion Model Selected: {best_model_name} (F1: {results_df.iloc[0]['F1-Score']:.4f})")

    # Detailed Classification Report for Champion Model
    best_pred = test_predictions[best_model_name]
    report_str = classification_report(y_test, best_pred, target_names=class_names, digits=4)
    print(f"\n--- Classification Report ({best_model_name}) ---")
    print(report_str)

    report_path = os.path.join(models_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Test Samples: {len(y_test)}\n\n")
        f.write(report_str)
    print(f"[+] Saved classification report to: {report_path}")

    # =========================================================================
    # SERIALIZATION / EXPORT OF CHAMPION MODEL BUNDLE
    # =========================================================================
    # We bundle the model with metadata for clean, robust inference in Streamlit
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "features": features,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "use_scaled": best_config["use_scaled"],
        "metrics": results_df.iloc[0].to_dict()
    }
    model_save_path = os.path.join(models_dir, "water_pipeline_model.pkl")
    joblib.dump(model_bundle, model_save_path)
    print(f"[+] Champion model package saved to: {model_save_path}")

    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("[*] Generating publication-quality evaluation visualizations...")
    sns.set_theme(style="whitegrid")

    # 1. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    df_plot = pd.melt(results_df, id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"])
    ax = sns.barplot(data=df_plot, x="Model", y="value", hue="variable", palette="viridis")
    plt.title("Water Pipeline Fault Detection - Model Performance Comparison", fontsize=14, weight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylim(0.85, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comparison_chart_path = os.path.join(models_dir, "model_comparison.png")
    plt.savefig(comparison_chart_path, dpi=300)
    plt.close()
    print(f"    - Saved: {comparison_chart_path}")

    # 2. Confusion Matrix Heatmap for Champion Model
    cm = confusion_matrix(y_test, best_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title(f"Confusion Matrix: {best_model_name} (Counts)", fontsize=13, weight="bold")
    plt.ylabel("True Physical Condition", fontsize=11)
    plt.xlabel("Predicted Condition", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(models_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"    - Saved: {cm_path}")

    # 3. Feature Importance Plot (if tree-based champion or Random Forest)
    feat_model = best_model if hasattr(best_model, "feature_importances_") else trained_models.get("Random Forest")
    if feat_model and hasattr(feat_model, "feature_importances_"):
        importances = feat_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_feats = [features[i] for i in indices]
        sorted_scores = importances[indices]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_scores, y=sorted_feats, hue=sorted_feats, palette="mako", legend=False)
        plt.title(f"Feature Importance Analysis ({feat_model.__class__.__name__})", fontsize=13, weight="bold")
        plt.xlabel("Gini Relative Importance", fontsize=11)
        plt.ylabel("Telemetry Feature", fontsize=11)
        plt.tight_layout()
        feat_imp_path = os.path.join(models_dir, "feature_importance.png")
        plt.savefig(feat_imp_path, dpi=300)
        plt.close()
        print(f"    - Saved: {feat_imp_path}")

    print("\n[✓] All training, evaluation, and visualization tasks completed successfully!")
    return results_df, model_save_path


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ds_file = os.path.join(base_dir, "dataset", "water_pipeline_dataset.csv")
    md_dir = os.path.join(base_dir, "models")
    train_and_evaluate_models(dataset_path=ds_file, models_dir=md_dir)
