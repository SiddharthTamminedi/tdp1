// YF-S201 -> ESP32 -> laptop (USB serial).
// Sends one line per second like:  FLOW,2.10
const int SENSOR_PIN = 27;                // D27
volatile unsigned long pulseCount = 0;
unsigned long lastTime = 0;

void IRAM_ATTR countPulse() { pulseCount++; }

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(SENSOR_PIN), countPulse, FALLING);
}

void loop() {
  if (millis() - lastTime >= 1000) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();
    lastTime = millis();
    float flowRate = pulses / 7.5;        // YF-S201: pulses per second / 7.5 = L/min
    Serial.print("FLOW,");
    Serial.println(flowRate, 2);
  }
}
