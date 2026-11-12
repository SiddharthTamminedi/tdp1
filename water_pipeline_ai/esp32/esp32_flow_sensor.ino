/*
 * ==============================================================================
 * Project: AI-Based Water Pipeline Fault Detection System
 * Hardware: ESP32 Development Board + YF-S201 Water Flow Sensor
 * Author: IoT & Embedded AI Engineering Team
 * ==============================================================================
 *
 * SENSOR SPECIFICATIONS (YF-S201):
 * - Sensor Type: Hall Effect Turbine Flow Meter
 * - Working Voltage: DC 4.5V to 18V (Nominal: 5V DC)
 * - Output High Level: > 4.5V (when powered from 5V)
 * - Output Low Level:  < 0.5V
 * - Pulse Characteristic: Frequency (Hz) = 7.5 * FlowRate (L/min)
 * - Calibration Factor: 7.5 pulses per second per 1 L/min (~450 pulses per Liter)
 *
 * CRITICAL ELECTRICAL / HARDWARE SAFETY NOTE:
 * - The YF-S201 Hall sensor requires 5V VCC to switch cleanly.
 * - However, ESP32 GPIOs operate strictly at 3.3V logic (Max rating: ~3.6V).
 * - Direct 5V pulse connection WILL damage or degrade the ESP32 GPIO over time!
 * - ALWAYS use a resistive voltage divider or bidirectional logic level shifter:
 *
 *   YF-S201 Signal (Yellow, 5V pulse)
 *             │
 *            ┌┴┐
 *            │ │ R1 (1.0 kΩ or 10 kΩ)
 *            └┬┘
 *             ├───> ESP32 GPIO 27 (reads ~3.33V peak safe logic)
 *            ┌┴┐
 *            │ │ R2 (2.0 kΩ or 20 kΩ)
 *            └┬┘
 *             │
 *            GND (ESP32 & Sensor common ground)
 *
 * WIRING SUMMARY:
 * - YF-S201 RED (VCC)    ---> ESP32 VIN / 5V pin (from USB)
 * - YF-S201 BLACK (GND)  ---> ESP32 GND
 * - YF-S201 YELLOW (SIG) ---> Voltage Divider Input ---> ESP32 GPIO 27
 * ==============================================================================
 */

#include <Arduino.h>

// Pin definition
const uint8_t SENSOR_PIN = 27; // Interrupt capable GPIO on ESP32

// Flow sensor calibration parameters
// YF-S201 formula: Flow Rate (L/min) = Frequency (Hz) / 7.5
const float CALIBRATION_FACTOR = 7.5f;

// Volatile counters for ISR (Interrupt Service Routine)
volatile unsigned long pulseCount = 0;
portMUX_TYPE isrMux = portMUX_INITIALIZER_UNLOCKED;

// Timing and calculation variables
unsigned long previousMillis = 0;
const unsigned long INTERVAL_MS = 1000; // Sample every 1000 ms (1 second)

float totalVolumeLitres = 0.0f;
float instantaneousFlowRate = 0.0f;

// Interrupt Service Routine (placed in IRAM for fast execution)
void IRAM_ATTR flowPulseISR() {
  portENTER_CRITICAL_ISR(&isrMux);
  pulseCount++;
  portEXIT_CRITICAL_ISR(&isrMux);
}

void setup() {
  // Initialize USB Serial communication at 115200 baud
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    // Wait up to 3 seconds for serial console ready
  }

  // Configure sensor pin as input with internal pull-up (or external divider)
  pinMode(SENSOR_PIN, INPUT_PULLUP);

  // Attach falling-edge interrupt
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), flowPulseISR, FALLING);

  previousMillis = millis();

  // Print startup metadata (in JSON-friendly diagnostic comment format)
  Serial.println();
  Serial.println("{\"status\":\"INITIALIZED\",\"device\":\"ESP32_FLOW_NODE\",\"pin\":27,\"baud\":115200}");
}

void loop() {
  unsigned long currentMillis = millis();

  // Non-blocking 1-second measurement window
  if (currentMillis - previousMillis >= INTERVAL_MS) {
    unsigned long elapsedMillis = currentMillis - previousMillis;
    previousMillis = currentMillis;

    // Atomically read and reset pulse counter
    unsigned long currentPulses = 0;
    portENTER_CRITICAL(&isrMux);
    currentPulses = pulseCount;
    pulseCount = 0;
    portEXIT_CRITICAL(&isrMux);

    // Calculate pulse frequency in Hz (pulses per second)
    float frequencyHz = ((float)currentPulses * 1000.0f) / (float)elapsedMillis;

    // Calculate flow rate in Liters per minute
    instantaneousFlowRate = frequencyHz / CALIBRATION_FACTOR;

    // Integrate volume: Liters = (L/min) / 60 * (elapsedSeconds)
    float volumeDeltaLitres = (instantaneousFlowRate / 60.0f) * ((float)elapsedMillis / 1000.0f);
    totalVolumeLitres += volumeDeltaLitres;

    // Guard against floating point drift or negative values
    if (instantaneousFlowRate < 0.02f) {
      instantaneousFlowRate = 0.0f;
    }

    /*
     * Emit clean, structured JSON string to USB Serial.
     * Schema:
     * {
     *   "flow_rate": <float in L/min>,
     *   "total_volume": <float in Litres>,
     *   "pulse_count": <int count in past window>,
     *   "elapsed_ms": <int duration in ms>
     * }
     */
    Serial.print("{\"flow_rate\": ");
    Serial.print(instantaneousFlowRate, 3);
    Serial.print(", \"total_volume\": ");
    Serial.print(totalVolumeLitres, 3);
    Serial.print(", \"pulse_count\": ");
    Serial.print(currentPulses);
    Serial.print(", \"elapsed_ms\": ");
    Serial.print(elapsedMillis);
    Serial.println("}");
  }
}

