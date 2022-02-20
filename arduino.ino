#include <Wire.h>
#include <Servo.h>


// Настройки
#define SIZE 16
#define SERVO_PIN 9
#define LED_PIN A0
#define MOTORB 4
#define MOTORS 5

// Переменные
Servo servo;
byte volume = 45;
int SLAVE_ADDRESS = 0x8;
char data[6];
char data_speed[3];
char data_wheel[3];
int x = 0;
long speed_d, wheel;


void setup() {
  pinMode(MOTORB, OUTPUT);
  pinMode(MOTORS, OUTPUT);
  Wire.begin(SLAVE_ADDRESS);
  Wire.onReceive(processMessage);
  Serial.begin(9600);
  servo.attach(SERVO_PIN);
}

void loop() {
  
}

// Получение сообщения
void processMessage(int n) {
  char ch = Wire.read();
  if (ch != '*') {
    data[x] = ch;
    x += 1;
  }
  
  if (ch == '*'){
//    Serial.println(sizeof(data));
    long res = atol(data);
    speed_d = (res % 1000) - 100;
    wheel = (res - speed_d - 100100) / 1000;
    for (int i; i < sizeof(data); i++){
      data[i] = ' ';
    }
    x = 0;
  }

  writeServo(wheel);
  writeMotor(speed_d);
}

// Поворот серво
void writeServo(long val){
  Serial.println(val);
  servo.write(val);
}

// Мощность мотора
void writeMotor(int spe){
  analogWrite(MOTORS, spe);
  digitalWrite(MOTORB, 0);
}


// made by Tema and Denis
