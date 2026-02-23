#!/usr/bin/env python3
import time
import threading
import os
import math
import ctypes
import serial
import cv2
import numpy as np
import smbus2
import RPi.GPIO as GPIO
import tflite_runtime.interpreter as tflite

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit

# =========================
# CONFIGURATION
# =========================
ENABLE_CELLULAR_LOCATION = False

GPS_SERIAL_PORT = '/dev/serial0'
IMU_I2C_ADDRESS = 0x68

LIDAR_SAMPLE_INTERVAL = 0.25
IMU_GPS_SAMPLE_INTERVAL = 1
MAIN_LOOP_INTERVAL = 0.1
AI_PROCESS_EVERY_N_FRAMES = 3
MANUAL_COMMAND_TIMEOUT = 0.5

MOTOR_A_PWM_PIN = 12
MOTOR_A_DIR_PIN = 5
MOTOR_B_PWM_PIN = 13
MOTOR_B_DIR_PIN = 6
PWM_FREQ = 100

FORWARD_SPEED = 20
TURN_SPEED = 50

MODEL_DIR = "models"
MODEL_NAME = "model.tflite"
LABEL_NAME = "labels.txt"
MIN_CONF_THRESHOLD = 0.5
TARGET_OBJECT = "person"

LIDAR_OBSTACLE_DISTANCE = 0.4
LIDAR_FRONT_ANGLE_RANGE = (-20, 20)

# =========================
# GLOBAL STATE
# =========================
data_lock = threading.Lock()
shutdown_flag = threading.Event()

closest_lidar_distance = float("inf")
target_center_x = -1

imu_data = {"pitch": 0.0, "roll": 0.0}
gps_data = {"fix": False, "lat": 0.0, "lon": 0.0}

control_mode = "MANUAL"
last_manual_command_time = time.time()

# =========================
# MOTOR CONTROLLER
# =========================
class MotorController:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(MOTOR_A_PWM_PIN, GPIO.OUT)
        GPIO.setup(MOTOR_A_DIR_PIN, GPIO.OUT)
        GPIO.setup(MOTOR_B_PWM_PIN, GPIO.OUT)
        GPIO.setup(MOTOR_B_DIR_PIN, GPIO.OUT)

        self.pwm_a = GPIO.PWM(MOTOR_A_PWM_PIN, PWM_FREQ)
        self.pwm_b = GPIO.PWM(MOTOR_B_PWM_PIN, PWM_FREQ)

        self.pwm_a.start(0)
        self.pwm_b.start(0)

        print("MotorController initialized")

    def set_motors(self, left, right):
        GPIO.output(MOTOR_A_DIR_PIN, GPIO.HIGH if left >= 0 else GPIO.LOW)
        GPIO.output(MOTOR_B_DIR_PIN, GPIO.HIGH if right >= 0 else GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(abs(left))
        self.pwm_b.ChangeDutyCycle(abs(right))

    def forward(self, speed=FORWARD_SPEED):
        self.set_motors(-speed, -speed)

    def backward(self, speed=FORWARD_SPEED):
        self.set_motors(speed, speed)

    def turn_left(self, speed=TURN_SPEED):
        self.set_motors(-speed, speed)

    def turn_right(self, speed=TURN_SPEED):
        self.set_motors(speed, -speed)

    def stop(self):
        self.set_motors(0, 0)

    def cleanup(self):
        self.stop()
        GPIO.cleanup()
        print("GPIO cleaned up")

# =========================
# THREADS
# =========================
def lidar_thread():
    global closest_lidar_distance
    print("Lidar thread started")

    while not shutdown_flag.is_set():
        min_dist = 5.0
        for angle in range(-20, 21, 2):
            if time.time() % 10 > 5:
                min_dist = 0.3
        with data_lock:
            closest_lidar_distance = min_dist
        time.sleep(LIDAR_SAMPLE_INTERVAL)

def imu_thread():
    global imu_data
    try:
        bus = smbus2.SMBus(1)
        bus.write_byte_data(IMU_I2C_ADDRESS, 0x6B, 0)
        print("IMU thread started")

        while not shutdown_flag.is_set():
            imu_data["pitch"] = 0.0
            imu_data["roll"] = 0.0
            time.sleep(IMU_GPS_SAMPLE_INTERVAL)

    except Exception as e:
        print(f"IMU error: {e}")

def gps_thread():
    global gps_data
    try:
        ser = serial.Serial(GPS_SERIAL_PORT, 9600, timeout=1)
        print("GPS thread started")

        while not shutdown_flag.is_set():
            line = ser.readline().decode(errors="ignore")
            if line.startswith("$GPGGA"):
                gps_data["fix"] = True
            time.sleep(0.1)

    except Exception as e:
        print(f"GPS error: {e}")

# =========================
# WEB SERVER
# =========================
HTML_TEMPLATE = """
<!doctype html>
<html>
<body>
<h2>Robot Control</h2>
<button onclick="send('forward')">Forward</button>
<button onclick="send('left')">Left</button>
<button onclick="send('right')">Right</button>
<button onclick="send('backward')">Backward</button>
<button onclick="send('stop')">Stop</button>
<button onclick="socket.emit('switch_mode')">Switch Mode</button>

<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
<script>
const socket = io();
function send(a){ socket.emit('cmd',{action:a,speed:50}); }
</script>
</body>
</html>
"""

app = Flask(__name__)
socketio = SocketIO(app)

def web_server_thread():
    @app.route("/")
    def index():
        return render_template_string(HTML_TEMPLATE)

    @socketio.on("cmd")
    def on_cmd(data):
        global last_manual_command_time
        if control_mode != "MANUAL":
            return
        action = data["action"]
        if action == "forward": motors.forward()
        elif action == "backward": motors.backward()
        elif action == "left": motors.turn_left()
        elif action == "right": motors.turn_right()
        elif action == "stop": motors.stop()
        last_manual_command_time = time.time()

    @socketio.on("switch_mode")
    def switch_mode():
        global control_mode
        control_mode = "AUTO" if control_mode == "MANUAL" else "MANUAL"
        motors.stop()
        emit("mode_update", {"mode": control_mode}, broadcast=True)

    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)

# =========================
# MAIN
# =========================
def main():
    global motors
    motors = MotorController()

    threads = [
        threading.Thread(target=lidar_thread),
        threading.Thread(target=imu_thread),
        threading.Thread(target=gps_thread),
        threading.Thread(target=web_server_thread, daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while not shutdown_flag.is_set():
            with data_lock:
                obstacle = closest_lidar_distance
                mode = control_mode

            if mode == "AUTO":
                if obstacle < LIDAR_OBSTACLE_DISTANCE:
                    motors.stop()
                else:
                    motors.forward()

            elif mode == "MANUAL":
                if time.time() - last_manual_command_time > MANUAL_COMMAND_TIMEOUT:
                    motors.stop()

            time.sleep(MAIN_LOOP_INTERVAL)

    except KeyboardInterrupt:
        print("\nShutting down")

    finally:
        shutdown_flag.set()
        motors.cleanup()

if __name__ == "__main__":
    main()
