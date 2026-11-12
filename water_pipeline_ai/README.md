# 💧 AI-Based Water Pipeline Fault Detection System

An end-to-end Edge IoT and Machine Learning monitoring solution designed for real-time fault detection in pressurized water pipelines. The system combines an **ESP32 microcontroller**, a **YF-S201 Hall effect water flow sensor**, an **advanced feature-engineering and machine learning inference engine**, and an industrial **Streamlit web dashboard**.

---

## 📌 System Architecture

```
[ Water Pipeline ]
        │
        ▼ (Fluid flow rotates internal turbine)
[ YF-S201 Sensor ] (Hall-effect pulse train: 5V DC)
        │
        ▼ (1kΩ / 2kΩ Resistive Voltage Divider: 5V -> 3.3V Logic Protection)
[ ESP32 MCU (GPIO 27) ]
        │ (Hardware interrupt counts pulses; computes Flow Rate & Total Volume)
        │ (Formatted as structured JSON payload)
        ▼ (USB Serial / UART @ 115200 Baud)
[ Laptop / Host PC ]
        ├── PySerial Stream Engine (Robust line buffering & error recovery)
        ├── Rolling Window Telemetry Buffer (11 dynamic statistical features)
        ├── Machine Learning Inference Engine (Random Forest / Decision Tree / SVM)
        ├── Debounce / Hysteresis Filter (3-sample consecutive state confirmation)
        └── Streamlit Industrial Dashboard (Live KPIs, Charts, Alert Badges, CSV Audit Log)
```

---

## 📂 Project Repository Structure

```
water_pipeline_ai/
│
├── dataset/
│   └── water_pipeline_dataset.csv     # 15,000 synthetic time-series sensor samples
│
├── models/
│   ├── water_pipeline_model.pkl       # Champion trained ML model bundle + metadata
│   ├── scaler.pkl                     # StandardScaler fitted strictly on train partition
│   ├── classification_report.txt      # Precision, recall, and F1 metrics
│   ├── confusion_matrix.png           # Multi-class confusion matrix visualization
│   ├── feature_importance.png         # Gini feature importance ranking
│   └── model_comparison.png          # Comparison across candidate ML models
│
├── esp32/
│   └── esp32_flow_sensor.ino          # ESP32 firmware with atomic ISR & JSON serial
│
├── training/
│   ├── generate_dataset.py            # Physics-guided synthetic dataset generator
│   └── train_model.py                 # Leakage-free group-split ML training & benchmarking
│
├── dashboard/
│   └── app.py                         # Streamlit live telemetry & AI monitoring UI
│
├── logs/
│   └── pipeline_logs.csv              # Persistent telemetry and fault audit logger
│
├── requirements.txt                   # Python environment dependencies
└── README.md                          # Complete engineering documentation
```

---

## ⚡ 1. Hardware Setup & Electrical Wiring

### Components:
- **ESP32 Development Board** (NodeMCU / DOIT ESP32 DevKit V1)
- **YF-S201 Hall Effect Water Flow Sensor** (1/2" pipe thread)
- **$1\text{ k}\Omega$ and $2\text{ k}\Omega$ Resistors** (or $10\text{ k}\Omega$ and $20\text{ k}\Omega$) for 5V to 3.3V logic level shifting
- **Micro-USB / USB-C Cable** (data-capable)
- **Plumbing Rig** (1/2" tube/hose, mini pump or tap source, ball valves for fault testing)

### Electrical Safety Warning & Voltage Divider:
> [!CAUTION]
> The YF-S201 Hall sensor requires **5V VCC** to reliably switch its internal transistor. When powered with 5V, its output pulse train swings to **5V**. Connecting this 5V pulse directly to an ESP32 GPIO pin will degrade or permanently damage the ESP32 input gate oxide (maximum rated voltage is ~3.6V).

Connect a resistive voltage divider between the sensor signal pin and the ESP32:

```
 YF-S201 Signal (Yellow Wire, 5V Pulse)
                 │
                ┌┴┐
                │ │  R1 = 1.0 kΩ (or 10 kΩ)
                └┬┘
                 ├───► Connect to ESP32 GPIO 27 (Input voltage: ~3.33V Safe Logic)
                ┌┴┐
                │ │  R2 = 2.0 kΩ (or 20 kΩ)
                └┬┘
                 │
           Common Ground (GND)
```

### Pinout Connections:
| YF-S201 Wire | Function | Target Connection |
| :--- | :--- | :--- |
| **RED** | VCC (+5V Power) | ESP32 **VIN** / **5V** Pin (powered from USB) |
| **BLACK** | GND (0V Ground) | ESP32 **GND** Pin |
| **YELLOW** | Digital Pulse SIG | **R1 (1kΩ)** top leg $\rightarrow$ Junction to **GPIO 27** $\rightarrow$ **R2 (2kΩ)** to GND |

---

## 📐 2. Sensor Physics & Calibration Formula

The YF-S201 consists of a plastic valve body, a water rotor, and a Hall-effect magnetic sensor. When water passes through the rotor, the rotor rolls and magnetic flux transitions generate square-wave pulses.

- **Calibration Frequency Equation:**
  $$f (\text{Hz}) = 7.5 \times Q$$
  where $f$ is pulse frequency in pulses per second, and $Q$ is flow rate in Liters per minute ($\text{L/min}$).

- **Flow Rate Calculation:**
  $$Q (\text{L/min}) = \frac{\text{Pulses in window}}{7.5 \times \Delta t (\text{seconds})}$$

- **Volumetric Consumption Integration:**
  $$V_{\text{total}} = \sum \left( \frac{Q_i}{60} \times \Delta t_i \right) \quad \text{or} \quad V (\text{Litres}) = \frac{\text{Total Accumulated Pulses}}{450}$$

---

## 📊 3. Synthetic Dataset & Feature Engineering

Because raw physical pipeline fault data is challenging and hazardous to record at scale initially, `generate_dataset.py` synthesizes **15,000 continuous time-series samples** across **150 independent experiment runs** (30 runs per class, 100 seconds per run) using stochastic mean-reverting Ornstein-Uhlenbeck processes.

### 5 Pipeline States Modeled:
1. **NORMAL**: Steady nominal flow ($2.8 - 4.5\text{ L/min}$), low variance ($\sigma \approx 0.08 - 0.18$).
2. **LEAKAGE**: Elevated flow ($4.8 - 7.5\text{ L/min}$) simulating high discharge from an uncontrolled rupture.
3. **BLOCKAGE**: Severely constricted or choked flow ($0.0 - 1.4\text{ L/min}$) with hydraulic fluttering.
4. **NO_FLOW**: Complete standstill ($0.0\text{ L/min}$, 0 pulses), valves closed or pump offline.
5. **ABNORMAL_FLOW**: Surging, water hammer, entrained air slug flow, chaotic swings ($0.5 - 9.0\text{ L/min}$, $\sigma > 0.8$).

### 11 Computed Telemetry Features:
- `flow_rate`: Instantaneous flow (L/min)
- `avg_flow_rate`: Rolling average over window $W=5$
- `min_flow_rate`: Rolling minimum over $W=5$
- `max_flow_rate`: Rolling maximum over $W=5$
- `std_flow_rate`: Rolling standard deviation over $W=5$
- `flow_rate_var`: Rolling variance over $W=5$
- `pulse_count`: Pulses recorded in sampling interval
- `total_volume`: Cumulative volume passed (Litres)
- `flow_rate_diff`: First-difference delta ($flow_t - flow_{t-1}$)
- `prev_flow_rate`: Lag-1 flow reading ($flow_{t-1}$)
- `rolling_avg_flow_rate`: Longer baseline rolling average ($W=10$)

> [!IMPORTANT]
> **Data Leakage Prevention:** The dataset uses `GroupShuffleSplit` on `run_id` (75% train, 25% test). No contiguous time-series points from the same physical run appear in both train and test partitions.
>
> **Physical Limitation Notice:** A single flow sensor measures flow at a single point. Elevated flow cannot be conclusively differentiated from intentional downstream usage without differential metering ($Q_{in} - Q_{out}$) or line pressure correlation. Therefore, all elevated leak signatures are classified as **SUSPECTED LEAKAGE**.

---

## 🤖 4. Model Training & Benchmark Results

The training suite (`training/train_model.py`) evaluates multiple classifiers:

| Model Architecture | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score | Data Scaling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Champion)** | **99.53%** | **99.53%** | **99.53%** | **0.9953** | Raw |
| **Decision Tree** | 99.29% | 99.29% | 99.29% | 0.9929 | Raw |
| **Support Vector Machine (SVM)** | 99.03% | 99.06% | 99.03% | 0.9903 | StandardScaler |

Artifacts saved:
- `models/water_pipeline_model.pkl`: Best model bundle with label encoders & feature lists.
- `models/scaler.pkl`: StandardScaler fit strictly on the train partition.
- `models/confusion_matrix.png`: High-resolution confusion matrix heatmap.
- `models/feature_importance.png`: Feature importance breakdown.
- `models/model_comparison.png`: Metric comparison bar chart.

---

## 🖥️ 5. Real-Time Streamlit Dashboard

Run the industrial monitoring dashboard with:
```bash
streamlit run dashboard/app.py
```

### Dashboard Capabilities:
- **Visual Status Banners:**
  - 🟢 **GREEN**: NORMAL (Healthy)
  - 🔴 **RED**: SUSPECTED LEAKAGE
  - 🟠 **ORANGE**: PIPELINE BLOCKAGE
  - ⚪ **GREY**: NO FLOW
  - 🟡 **YELLOW**: ABNORMAL FLOW
- **Hysteresis Debouncer:** Requires 3 consecutive identical fault predictions before flipping system alarm status to prevent false positives from transient bubbles.
- **Hardware & Simulator Modes:** Live USB serial monitoring of the ESP32 (e.g. `COM4`), or an interactive Test Rig Simulator to demonstrate all 5 fault scenarios on-demand.
- **Audit Logging & CSV Export:** Real-time logging to `logs/pipeline_logs.csv` and instant in-browser CSV download.

---

## 🚀 6. Step-by-Step Execution Guide

### STEP 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### STEP 2: Generate Synthetic Dataset
```bash
python training/generate_dataset.py
```
*Generates 15,000 samples and saves `dataset/water_pipeline_dataset.csv`.*

### STEP 3: Train and Evaluate AI Models
```bash
python training/train_model.py
```
*Trains candidate models, saves `models/water_pipeline_model.pkl`, `models/scaler.pkl`, and evaluation plots.*

### STEP 4: Wire Hardware & Connect ESP32
Connect the YF-S201 sensor to ESP32 using the voltage divider diagram above. Connect the ESP32 to your laptop via USB.

### STEP 5: Flash ESP32 Firmware
1. Open the Arduino IDE.
2. Under **Tools > Board**, select **ESP32 Dev Module**.
3. Under **Tools > Port**, select the active COM port (e.g. `COM4`).
4. Open `esp32/esp32_flow_sensor.ino` and click **Upload**.

### STEP 6: Launch Real-Time Dashboard
```bash
streamlit run dashboard/app.py
```

### STEP 7: Start Live Monitoring
1. In the dashboard sidebar, select your active COM port (`COM4`).
2. Click **▶ START**.
3. Watch the live telemetry, AI prediction, flow charts, and alert logs update in real time!

---

## 🧪 7. Physical Testing & Fault Simulation Guide

To test the physical system with water flow:

1. **NORMAL Condition:**
   - Turn on the water tap or mini-pump to a steady, medium flow rate (~$3.0 - 4.0\text{ L/min}$).
   - The dashboard displays **GREEN: NORMAL (HEALTHY)**.

2. **LEAKAGE Condition (Suspected Leakage):**
   - Open a downstream bypass valve or increase flow rate to maximum capacity ($> 5.5\text{ L/min}$).
   - The AI detects abnormal continuous volume discharge and alerts **RED: SUSPECTED LEAKAGE**.

3. **BLOCKAGE Condition:**
   - Partially close an inline valve or clamp the hose downstream to pinch flow down to $< 1.2\text{ L/min}$.
   - The AI identifies throttled flow and flags **ORANGE: PIPELINE BLOCKAGE**.

4. **NO FLOW Condition:**
   - Close the inlet valve completely ($0.0\text{ L/min}$, zero pulses).
   - The AI transitions to **GREY: NO FLOW (STAGNANT)**.

5. **ABNORMAL FLOW Condition:**
   - Rapidly open and close the tap valve, introduce air bubbles into the intake, or cycle the pump power erratically.
   - The AI detects turbulent surges and high variance, triggering **YELLOW: ABNORMAL FLOW DYNAMICS**.

