#define BUZZER_PIN 3

void setup() {
    pinMode(BUZZER_PIN, OUTPUT);
    Serial.begin(9600);  // Inicia comunicação serial
}

void loop() {
    if (Serial.available() > 0) {  // Verifica se há dados recebidos
        String command = Serial.readStringUntil('\n');  // Lê comando até o caractere de nova linha
        
        if (command == "ON") {
            digitalWrite(BUZZER_PIN, HIGH);
        } else if (command == "OFF") {
            digitalWrite(BUZZER_PIN, LOW);
        }
    }
}
