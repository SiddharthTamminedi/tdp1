"""
==============================================================================
Project: AI-Based Water Pipeline Fault Detection System
File: dashboard/app.py
Purpose: Real-time interactive Streamlit dashboard for IoT water pipeline
         monitoring, ESP32 serial communication, rolling feature extraction,
         and machine learning fault inference with alert debouncing.
==============================================================================
"""

import os
import sys
import time
import json
import re
from datetime import datetime
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Serial communication
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ==============================================================================
# PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Water Pipeline Fault Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-visibility industrial telemetry cards
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .status-badge {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 12px 20px;
        border-radius: 8px;
        text-align: center;
        letter-spacing: 0.5px;
        color: white;
        margin-bottom: 15px;
    }
    .badge-normal {
        background-color: #28a745; /* GREEN */
    }
    .badge-leakage {
        background-color: #dc3545; /* RED */
    }
    .badge-blockage {
        background-color: #fd7e14; /* ORANGE */
    }
    .badge-noflow {
        background-color: #6c757d; /* GREY */
    }
    .badge-abnormal {
        background-color: #e0a800; /* YELLOW / DARK GOLD */
        color: #212529 !important;
    }
    .alert-box {
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# PATH RESOLUTION & MODEL LOADING
# ==============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "water_pipeline_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
LOG_CSV_FILE = os.path.join(LOGS_DIR, "pipeline_logs.csv")

os.makedirs(LOGS_DIR, exist_ok=True)

@st.cache_resource
def load_ml_pipeline():
    """Loads trained model bundle and scaler with error verification."""
    if not os.path.exists(MODEL_FILE):
        return None, None, f"Model file not found at {MODEL_FILE}. Run train_model.py first."
    if not os.path.exists(SCALER_FILE):
        return None, None, f"Scaler file not found at {SCALER_FILE}. Run train_model.py first."

    try:
        bundle = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return bundle, scaler, None
    except Exception as e:
        return None, None, f"Error loading model assets: {str(e)}"

model_bundle, scaler_obj, load_error = load_ml_pipeline()


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "reading_buffer" not in st.session_state:
    st.session_state.reading_buffer = deque(maxlen=25)  # Recent raw flow readings

if "history_df" not in st.session_state:
    st.session_state.history_df = pd.DataFrame(columns=[
        "timestamp", "flow_rate", "total_volume", "pulse_count",
        "predicted_condition", "confidence", "status_display"
    ])

if "active_fault" not in st.session_state:
    st.session_state.active_fault = "NORMAL"

if "consecutive_candidate" not in st.session_state:
    st.session_state.consecutive_candidate = None

if "consecutive_count" not in st.session_state:
    st.session_state.consecutive_count = 0

if "latest_confidence" not in st.session_state:
    st.session_state.latest_confidence = 100.0

if "latest_reading" not in st.session_state:
    st.session_state.latest_reading = {
        "flow_rate": 0.0,
        "total_volume": 0.0,
        "pulse_count": 0,
        "elapsed_ms": 1000
    }

if "alerts_history" not in st.session_state:
    st.session_state.alerts_history = []

if "sim_volume" not in st.session_state:
    st.session_state.sim_volume = 0.0


# ==============================================================================
# SERIAL COMMUNICATION HELPERS
# ==============================================================================
def get_available_com_ports():
    """Detects available serial ports on the system."""
    if not SERIAL_AVAILABLE:
        return []
    ports = serial.tools.list_ports.comports()
    return [(p.device, f"{p.device} - {p.description}") for p in ports]

def parse_incoming_serial_line(line_str, current_total_vol=0.0):
    """
    Parses incoming raw line from ESP32.
    Supports standard JSON format:
      {"flow_rate": 2.50, "total_volume": 10.50, "pulse_count": 125, "elapsed_ms": 1000}
    Also supports fallback regex for legacy formats:
      'Pulses: 18  Flow Rate: 2.40 L/min' or 'FLOW,2.40'
    """
    line_clean = line_str.strip()
    if not line_clean:
        return None

    # 1. Primary: JSON parser
    if line_clean.startswith("{") and line_clean.endswith("}"):
        try:
            data = json.loads(line_clean)
            flow = float(data.get("flow_rate", 0.0))
            pulses = int(data.get("pulse_count", 0))
            vol = float(data.get("total_volume", current_total_vol + (flow / 60.0)))
            elapsed = int(data.get("elapsed_ms", 1000))
            return {
                "flow_rate": max(0.0, flow),
                "total_volume": max(0.0, vol),
                "pulse_count": pulses,
                "elapsed_ms": elapsed
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Fallback: 'Pulses: <n>  Flow Rate: <f> L/min'
    match_pulse_flow = re.search(r"Pulses:\s*(\d+).*?Flow Rate:\s*([0-9.]+)", line_clean, re.IGNORECASE)
    if match_pulse_flow:
        try:
            pulses = int(match_pulse_flow.group(1))
            flow = float(match_pulse_flow.group(2))
            vol = current_total_vol + (flow / 60.0)
            return {
                "flow_rate": max(0.0, flow),
                "total_volume": round(vol, 4),
                "pulse_count": pulses,
                "elapsed_ms": 1000
            }
        except ValueError:
            pass

    # 3. Fallback: 'FLOW,<float>'
    if line_clean.startswith("FLOW,"):
        try:
            parts = line_clean.split(",")
            flow = float(parts[1])
            pulses = int(round(flow * 7.5))
            vol = current_total_vol + (flow / 60.0)
            return {
                "flow_rate": max(0.0, flow),
                "total_volume": round(vol, 4),
                "pulse_count": pulses,
                "elapsed_ms": 1000
            }
        except (ValueError, IndexError):
            pass

    return None


# ==============================================================================
# SYNTHETIC TEST RIG SIMULATOR
# ==============================================================================
def generate_simulated_packet(condition_mode):
    """
    Simulates realistic ESP32 physical sensor output with fluid inertia,
    enabling live test runs without hardware.
    """
    rng = np.random.default_rng()
    st.session_state.sim_volume = getattr(st.session_state, "sim_volume", 0.0)
    
    if condition_mode == "NORMAL":
        flow = max(0.2, rng.normal(3.5, 0.12))
    elif condition_mode == "LEAKAGE":
        flow = max(0.5, rng.normal(5.8, 0.35))
    elif condition_mode == "BLOCKAGE":
        flow = max(0.0, rng.normal(0.7, 0.15))
    elif condition_mode == "NO_FLOW":
        flow = 0.0 if rng.random() > 0.03 else 0.02
    elif condition_mode == "ABNORMAL_FLOW":
        flow = max(0.0, min(9.5, rng.choice([0.6, 2.0, 5.5, 7.8]) + rng.normal(0, 0.6)))
    else:
        flow = 3.2

    pulses = int(round(flow * 7.5))
    st.session_state.sim_volume += (flow / 60.0)
    
    return {
        "flow_rate": round(flow, 3),
        "total_volume": round(st.session_state.sim_volume, 3),
        "pulse_count": pulses,
        "elapsed_ms": 1000
    }


# ==============================================================================
# FEATURE ENGINEERING & INFERENCE PIPELINE
# ==============================================================================
def compute_telemetry_features(reading_buffer, current_reading):
    """
    Calculates exact 11 rolling features matching the training dataset pipeline:
      - flow_rate, avg_flow_rate, min_flow_rate, max_flow_rate, std_flow_rate,
        flow_rate_var, pulse_count, total_volume, flow_rate_diff,
        prev_flow_rate, rolling_avg_flow_rate
    """
    current_flow = current_reading["flow_rate"]
    
    # Prior reading
    prev_flow = reading_buffer[-1]["flow_rate"] if len(reading_buffer) > 0 else current_flow
    flow_diff = current_flow - prev_flow

    # Window of recent flows (up to last 5 samples)
    recent_flows = [item["flow_rate"] for item in reading_buffer] + [current_flow]
    w5_flows = recent_flows[-5:]
    w10_flows = recent_flows[-10:]

    avg_w5 = float(np.mean(w5_flows))
    min_w5 = float(np.min(w5_flows))
    max_w5 = float(np.max(w5_flows))
    std_w5 = float(np.std(w5_flows)) if len(w5_flows) > 1 else 0.0
    var_w5 = float(np.var(w5_flows)) if len(w5_flows) > 1 else 0.0

    rolling_avg_w10 = float(np.mean(w10_flows))

    feat_dict = {
        "flow_rate": round(current_flow, 3),
        "avg_flow_rate": round(avg_w5, 3),
        "min_flow_rate": round(min_w5, 3),
        "max_flow_rate": round(max_w5, 3),
        "std_flow_rate": round(std_w5, 3),
        "flow_rate_var": round(var_w5, 3),
        "pulse_count": int(current_reading["pulse_count"]),
        "total_volume": round(current_reading["total_volume"], 3),
        "flow_rate_diff": round(flow_diff, 3),
        "prev_flow_rate": round(prev_flow, 3),
        "rolling_avg_flow_rate": round(rolling_avg_w10, 3)
    }

    return feat_dict


def predict_pipeline_state(feature_dict, model_bundle, scaler_obj):
    """
    Executes ML inference on the calculated rolling feature vector.
    Returns: (predicted_class, confidence_percent, class_probabilities_dict)
    """
    if model_bundle is None:
        return "NORMAL", 100.0, {}

    model = model_bundle["model"]
    feature_cols = model_bundle["features"]
    use_scaled = model_bundle.get("use_scaled", False)
    class_names = model_bundle.get("class_names", [])

    # Assemble 1-row DataFrame with strict column order
    input_df = pd.DataFrame([[feature_dict[col] for col in feature_cols]], columns=feature_cols)

    # Feature transformation if model requires scaling
    X_eval = scaler_obj.transform(input_df) if use_scaled else input_df.values

    # Inference
    raw_pred = model.predict(X_eval)[0]
    
    # Resolve class label
    if hasattr(model_bundle.get("label_encoder"), "inverse_transform"):
        pred_label = model_bundle["label_encoder"].inverse_transform([raw_pred])[0]
    elif isinstance(raw_pred, (int, np.integer)) and class_names:
        pred_label = class_names[raw_pred]
    else:
        pred_label = str(raw_pred)

    # Confidence score calculation
    probs_dict = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_eval)[0]
        max_prob = float(np.max(probs))
        confidence = round(max_prob * 100.0, 1)
        for i, cname in enumerate(class_names):
            if i < len(probs):
                probs_dict[cname] = round(float(probs[i]) * 100.0, 1)
    else:
        confidence = 95.0

    return pred_label, confidence, probs_dict


def update_debounced_fault_state(raw_prediction, debounce_threshold=3):
    """
    Hysteresis debouncer: requires 'debounce_threshold' consecutive identical
    readings before committing a state transition, preventing single-packet
    turbulence blips from triggering false emergency alerts.
    """
    if raw_prediction == st.session_state.consecutive_candidate:
        st.session_state.consecutive_count += 1
    else:
        st.session_state.consecutive_candidate = raw_prediction
        st.session_state.consecutive_count = 1

    # Only flip persistent state once threshold is satisfied
    if st.session_state.consecutive_count >= debounce_threshold:
        old_state = st.session_state.active_fault
        st.session_state.active_fault = raw_prediction
        
        # Log alert event on transition to/between fault states
        if raw_prediction != old_state and raw_prediction != "NORMAL":
            alert_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": f"FAULT TRIGGERED: {raw_prediction}",
                "severity": "CRITICAL" if raw_prediction in ["LEAKAGE", "ABNORMAL_FLOW"] else "WARNING"
            }
            st.session_state.alerts_history.insert(0, alert_entry)
        elif raw_prediction == "NORMAL" and old_state != "NORMAL":
            recovery_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": f"RECOVERY: Flow stabilized back to NORMAL",
                "severity": "INFO"
            }
            st.session_state.alerts_history.insert(0, recovery_entry)

    return st.session_state.active_fault


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/water-pipe.png", width=64)
st.sidebar.title("Pipeline AI Control")

if load_error:
    st.sidebar.error(load_error)

# Operation Mode Selection
data_source = st.sidebar.radio(
    "Data Source",
    options=["Live ESP32 (Hardware Serial)", "Test Rig Simulator (Demo Mode)"],
    index=0
)

# Serial Port Settings
selected_com_port = None
baud_rate = 115200

if data_source == "Live ESP32 (Hardware Serial)":
    available_ports = get_available_com_ports()
    if available_ports:
        port_options = [p[0] for p in available_ports]
        port_labels = {p[0]: p[1] for p in available_ports}
        # Default to COM4 if present
        default_index = port_options.index("COM4") if "COM4" in port_options else 0
        selected_com_port = st.sidebar.selectbox(
            "ESP32 Serial Port",
            options=port_options,
            format_func=lambda x: port_labels[x],
            index=default_index
        )
    else:
        st.sidebar.warning("No active USB COM ports detected. Plug in ESP32 or switch to Simulator Mode.")
        selected_com_port = st.sidebar.text_input("Manual Port Entry (e.g. COM4)", value="COM4")
    
    baud_rate = st.sidebar.selectbox("Baud Rate", options=[115200, 9600, 57600], index=0)

else:
    st.sidebar.markdown("### Simulated Pipeline Rig")
    simulated_condition = st.sidebar.selectbox(
        "Inject Pipeline State",
        options=["NORMAL", "LEAKAGE", "BLOCKAGE", "NO_FLOW", "ABNORMAL_FLOW"],
        index=0
    )
    st.sidebar.caption("Change this selection during runtime to verify real-time fault detection!")

# Sensitivity / Debounce Control
st.sidebar.markdown("---")
st.sidebar.markdown("### Detection Settings")
debounce_n = st.sidebar.slider("Debounce Window (Consecutive Checks)", min_value=1, max_value=6, value=3)
window_len = st.sidebar.slider("Rolling History Length", min_value=10, max_value=60, value=30)

# Control Buttons
st.sidebar.markdown("---")
col_start, col_stop = st.sidebar.columns(2)
with col_start:
    if st.button("▶ START", use_container_width=True, type="primary"):
        st.session_state.is_running = True
with col_stop:
    if st.button("⏹ STOP", use_container_width=True):
        st.session_state.is_running = False

if st.sidebar.button("🔄 Reset Telemetry & Logs", use_container_width=True):
    st.session_state.reading_buffer.clear()
    st.session_state.history_df = st.session_state.history_df.iloc[0:0]
    st.session_state.alerts_history = []
    st.session_state.sim_volume = 0.0
    st.session_state.active_fault = "NORMAL"
    st.session_state.consecutive_count = 0
    st.rerun()


# ==============================================================================
# MAIN DASHBOARD INTERFACE
# ==============================================================================
st.title("💧 AI-Based Water Pipeline Fault Detection System")
st.caption("Real-Time IoT Edge Telemetry & Machine Learning Condition Monitoring | ESP32 + YF-S201")

# IMPORTANT DOMAIN NOTICE: Single-sensor physical ambiguity explanation
with st.expander("ℹ️ Important Physical Measurement & Sensor Limitation Note (Click to expand)", expanded=False):
    st.info(
        r"""
        **Single-Point Sensor Limitation:**
        A single YF-S201 sensor measures fluid velocity at a single cross-section. An unexpected rise in flow rate 
        may indicate a catastrophic downstream rupture/leak, or a sudden demand surge (e.g. valve opened downstream). 
        Conversely, a drop in flow could mean pipe blockage, or upstream municipal pressure failure.
        
        **Industrial Design Principle:**
        In this system, elevated anomalous flow signatures are rigorously flagged as **SUSPECTED LEAKAGE**. 
        For commercial utility deployment, dual-point differential flow metering ($Q_{in} - Q_{out} > \epsilon$) 
        and acoustic/pressure-wave sensors are recommended for absolute physical localization.
        """
    )

# SECTION A: SYSTEM STATUS DISPLAY
condition_color_map = {
    "NORMAL": ("badge-normal", "NORMAL (HEALTHY)", "🟢", "#28a745"),
    "LEAKAGE": ("badge-leakage", "SUSPECTED LEAKAGE", "🔴", "#dc3545"),
    "BLOCKAGE": ("badge-blockage", "PIPELINE BLOCKAGE", "🟠", "#fd7e14"),
    "NO_FLOW": ("badge-noflow", "NO FLOW (STAGNANT)", "⚪", "#6c757d"),
    "ABNORMAL_FLOW": ("badge-abnormal", "ABNORMAL FLOW DYNAMICS", "🟡", "#ffc107")
}

badge_class, display_title, status_icon, hex_color = condition_color_map.get(
    st.session_state.active_fault,
    ("badge-normal", "NORMAL", "🟢", "#28a745")
)

st.markdown(
    f"""
    <div class="status-badge {badge_class}">
        {status_icon} CURRENT PIPELINE STATUS: {display_title}
    </div>
    """,
    unsafe_allow_html=True
)

# Header KPIs
col_a1, col_a2, col_a3, col_a4 = st.columns(4)

with col_a1:
    st.metric(
        label="AI Confidence Score",
        value=f"{st.session_state.latest_confidence:.1f}%",
        delta="High Certainty" if st.session_state.latest_confidence > 90 else "Reviewing"
    )

with col_a2:
    esp_status_str = "ACTIVE STREAMING" if st.session_state.is_running else "IDLE / PAUSED"
    st.metric(
        label="ESP32 Comm Status",
        value=esp_status_str,
        delta=data_source.split(" ")[0]
    )

with col_a3:
    st.metric(
        label="Debounce Confirmation",
        value=f"{min(st.session_state.consecutive_count, debounce_n)} / {debounce_n} Samples",
        delta="State Locked" if st.session_state.consecutive_count >= debounce_n else "Verifying"
    )

with col_a4:
    sensor_health = "PULSES OK" if st.session_state.latest_reading["pulse_count"] > 0 or st.session_state.active_fault == "NO_FLOW" else "ZERO DETECTED"
    st.metric(
        label="YF-S201 Sensor Integrity",
        value=sensor_health
    )

st.markdown("---")


# SECTION B: LIVE TELEMETRY METRICS
st.subheader("📊 Live Sensor Telemetry")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

cur_flow = st.session_state.latest_reading["flow_rate"]
cur_vol = st.session_state.latest_reading["total_volume"]
cur_pulses = st.session_state.latest_reading["pulse_count"]
elapsed = st.session_state.latest_reading.get("elapsed_ms", 1000)

kpi1.metric("Flow Rate", f"{cur_flow:.2f} L/min", delta=f"{cur_flow - 3.2:.2f} rel baseline")
kpi2.metric("Cumulative Volume", f"{cur_vol:.3f} Litres")
kpi3.metric("Pulse Frequency", f"{cur_pulses} pulses/s", delta=f"{(cur_pulses/7.5):.2f} Hz")
kpi4.metric("Sampling Window", f"{elapsed} ms")


# SECTION C: LIVE CHARTS & PROBABILITY SPECTRUM
st.subheader("📈 Real-Time Pipeline Dynamics")
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    if not st.session_state.history_df.empty:
        # Create dual plot using streamlit native line_chart
        plot_df = st.session_state.history_df.tail(window_len).copy()
        plot_df = plot_df.set_index("timestamp")
        st.write("**Flow Rate History (L/min)**")
        st.line_chart(plot_df[["flow_rate"]], color="#007bff", height=230)
        
        st.write("**Cumulative Consumption (Litres)**")
        st.line_chart(plot_df[["total_volume"]], color="#28a745", height=180)
    else:
        st.info("Awaiting sensor acquisition stream. Click 'START' in the sidebar to begin monitoring.")

with chart_col2:
    st.write("**AI Class Probability Distribution**")
    if "latest_probs" in st.session_state and st.session_state.latest_probs:
        prob_series = pd.Series(st.session_state.latest_probs).sort_values(ascending=True)
        st.bar_chart(prob_series, height=220, color="#17a2b8")
    else:
        dummy_probs = pd.Series({
            "NORMAL": 95.0, "LEAKAGE": 2.0, "BLOCKAGE": 1.0,
            "NO_FLOW": 1.0, "ABNORMAL_FLOW": 1.0
        })
        st.bar_chart(dummy_probs, height=220, color="#17a2b8")

    st.write("**Fault Alarm Log**")
    if st.session_state.alerts_history:
        for alert in st.session_state.alerts_history[:5]:
            box_color = "#f8d7da" if alert["severity"] == "CRITICAL" else "#fff3cd" if alert["severity"] == "WARNING" else "#d1ecf1"
            text_color = "#721c24" if alert["severity"] == "CRITICAL" else "#856404" if alert["severity"] == "WARNING" else "#0c5460"
            st.markdown(
                f"""
                <div style="background-color: {box_color}; color: {text_color}; padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem;">
                    <strong>{alert['timestamp']}</strong>: {alert['event']}
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.caption("No pipeline fault events recorded in active session.")


# SECTION D & E: SYSTEM AUDIT LOG & CSV EXPORT
st.markdown("---")
st.subheader("📑 System Telemetry & Event Audit Log")

if not st.session_state.history_df.empty:
    display_log_df = st.session_state.history_df.tail(15).iloc[::-1]
    st.dataframe(
        display_log_df[[
            "timestamp", "flow_rate", "total_volume", "pulse_count",
            "predicted_condition", "confidence"
        ]],
        use_container_width=True
    )

    csv_data = st.session_state.history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Session Audit Log (CSV)",
        data=csv_data,
        file_name=f"pipeline_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.caption("No log entries collected yet.")


# ==============================================================================
# BACKGROUND STREAMING & INFERENCE LOOP (WHEN RUNNING)
# ==============================================================================
if st.session_state.is_running:
    # 1. Acquire raw reading from Serial or Simulator
    new_reading = None

    if data_source == "Live ESP32 (Hardware Serial)":
        if not selected_com_port:
            st.error("No COM port selected.")
            st.session_state.is_running = False
            st.stop()

        try:
            # Short-timeout read
            with serial.Serial(selected_com_port, baud_rate, timeout=1.2) as ser:
                # Flush input buffer to grab freshest frame
                ser.reset_input_buffer()
                raw_line = ser.readline().decode("utf-8", errors="replace")
                cur_vol = st.session_state.latest_reading["total_volume"]
                new_reading = parse_incoming_serial_line(raw_line, cur_vol)
        except serial.SerialException as se:
            st.warning(f"ESP32 Serial Warning: {str(se)}. Retrying in next cycle...")
            time.sleep(1.0)
            st.rerun()
        except Exception as e:
            st.error(f"Unexpected Serial Error: {str(e)}")
            st.session_state.is_running = False
            st.stop()

    else:
        # Simulator Mode
        new_reading = generate_simulated_packet(simulated_condition)
        time.sleep(1.0)

    # 2. Process reading if valid
    if new_reading is not None:
        st.session_state.latest_reading = new_reading

        # Calculate exact 11 features matching ML model
        features = compute_telemetry_features(st.session_state.reading_buffer, new_reading)
        st.session_state.reading_buffer.append(new_reading)

        # ML Inference
        pred_condition, confidence, class_probs = predict_pipeline_state(
            features, model_bundle, scaler_obj
        )
        st.session_state.latest_confidence = confidence
        st.session_state.latest_probs = class_probs

        # Debounced State Filter
        stable_state = update_debounced_fault_state(pred_condition, debounce_threshold=debounce_n)

        # Append to live history
        now_str = datetime.now().strftime("%H:%M:%S")
        new_row = {
            "timestamp": now_str,
            "flow_rate": new_reading["flow_rate"],
            "total_volume": new_reading["total_volume"],
            "pulse_count": new_reading["pulse_count"],
            "predicted_condition": stable_state,
            "confidence": confidence,
            "status_display": f"{stable_state} ({confidence}%)"
        }
        st.session_state.history_df = pd.concat([
            st.session_state.history_df,
            pd.DataFrame([new_row])
        ], ignore_index=True)

        # Append to persistent disk log (pipeline_logs.csv)
        try:
            alert_lvl = "NORMAL" if stable_state == "NORMAL" else "CRITICAL" if stable_state in ["LEAKAGE", "ABNORMAL_FLOW"] else "WARNING"
            disk_log_row = f"{datetime.now().isoformat()},{new_reading['flow_rate']},{new_reading['total_volume']},{new_reading['pulse_count']},{features['avg_flow_rate']},{features['std_flow_rate']},{stable_state},{confidence},{alert_lvl},LOGGED\n"
            with open(LOG_CSV_FILE, "a", encoding="utf-8") as f:
                f.write(disk_log_row)
        except Exception:
            pass

    # Seamless sub-second loop trigger
    time.sleep(0.1)
    st.rerun()

