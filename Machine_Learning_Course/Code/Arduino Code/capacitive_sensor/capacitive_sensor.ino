#include <CapacitiveSensor.h>

int baseline = 1000;
int offset = 100;

// Initialize the sensor
// 1 is the Send pin, 2 is the Receive pin (where your foil is attached)
CapacitiveSensor keyA = CapacitiveSensor(A3, 12);
CapacitiveSensor keyB = CapacitiveSensor(A3, 11);
CapacitiveSensor keyC = CapacitiveSensor(A3, 10);
CapacitiveSensor keyD = CapacitiveSensor(A3, 9);
CapacitiveSensor keyE = CapacitiveSensor(A3, 8);
CapacitiveSensor keyF = CapacitiveSensor(A3, 7);
CapacitiveSensor keyG = CapacitiveSensor(A3, 6);
int LEDPin = 2;

int LEDPins [] = {5, 4, 3, 2, A0, A1, A2};

// Hysteresis: Set a high threshold to turn the LED ON, but require the signal to drop below a much lower threshold to turn it OFF.
int prevValues [] = {0, 0, 0, 0, 0, 0, 0};

void setup() {
  // Serial Monitoring
  Serial.begin(115200);
  for (int i = 0; i < 7; i++)
    pinMode(LEDPins[i], OUTPUT);
}

void loop() {
  long valueA = keyA.capacitiveSensor(5);
  handleValues(valueA, 0);
  // Serial.println(valueA);

  long valueB = keyB.capacitiveSensor(5);
  handleValues(valueB, 1);

  long valueC = keyC.capacitiveSensor(5);
  handleValues(valueC, 2);
  
  long valueD = keyD.capacitiveSensor(5);
  handleValues(valueD, 3);
  
  long valueE = keyE.capacitiveSensor(5);
  handleValues(valueE, 4);
  
  long valueF = keyF.capacitiveSensor(5);
  handleValues(valueF, 5);
  
  long valueG = keyG.capacitiveSensor(5);
  handleValues(valueG, 6);
}

void handleValues(int values, int i){
  // Handles the values read for each pin
    if (values > baseline + offset) {
      // Serial.println(String(values) + " " + String(i));
      digitalWrite(LEDPins[i], HIGH);
    } else {
      digitalWrite(LEDPins[i], LOW);
    }
}

