import time
import threading
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import os
import ctypes
import smbus2
import serial
import math
import requests
import json
from flask import Flask, Response, render_template_string
from flask_socketio import SocketIO, emit

# --- CONFIGURATION ---
ENABLE_CELLULAR_LOCATION = False
UNWIRED_API_KEY = "pk.d4733963ace6bd7a65a18ad7e5975b04"
CELLULAR_MODEM_PORT = '/dev/ttyUSB2'
GPS_SERIAL_PORT = '/dev/serial0'
IMU_I2C_ADDRESS = 0x68
LIDAR_PORT = '/dev/ttyUSB0'
CELLULAR_LOCATION_INTERVAL = 15
IMU_GPS_SAMPLE_INTERVAL = 1
LIDAR_SAMPLE_INTERVAL = 0.25
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
MODEL_DIR = 'models'
MODEL_NAME = 'model.tflite'
LABEL_NAME = 'labels.txt'
MIN_CONF_THRESHOLD = 0.5
TARGET_OBJECT = 'person'
LIDAR_OBSTACLE_DISTANCE = 0.4
LIDAR_FRONT_ANGLE_RANGE = (-20, 20)

# --- GLOBAL STATE & THREADING LOCKS ---
data_lock = threading.Lock()
shutdown_flag = threading.Event()
closest_lidar_distance = float('inf')
target_center_x = -1
imu_data = {'pitch': 0.0, 'roll': 0.0}
gps_data = {'fix': False, 'lat': 0.0, 'lon': 0.0}
cellular_location_data = {'fix': False, 'lat': 0.0, 'lon': 0.0, 'accuracy': 0}

# Web Control Data
control_mode = 'MANUAL'
last_manual_command_time = time.time()

# --- MOTOR CONTROLLER CLASS ---
class MotorController:
    """Handles all motor control functions."""
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(MOTOR_A_PWM_PIN, GPIO.OUT)
        GPIO.setup(MOTOR_A_DIR_PIN, GPIO.OUT)
        self.motor_a_pwm = GPIO.PWM(MOTOR_A_PWM_PIN, PWM_FREQ)
        self.motor_a_pwm.start(0)
        GPIO.setup(MOTOR_B_PWM_PIN, GPIO.OUT)
        GPIO.setup(MOTOR_B_DIR_PIN, GPIO.OUT)
        self.motor_b_pwm = GPIO.PWM(MOTOR_B_PWM_PIN, PWM_FREQ)
        self.motor_b_pwm.start(0)
        print("MotorController initialized.")

    def set_motors(self, left_speed, right_speed):
        GPIO.output(MOTOR_A_DIR_PIN, GPIO.HIGH if left_speed >= 0 else GPIO.LOW)
        self.motor_a_pwm.ChangeDutyCycle(abs(left_speed))
        GPIO.output(MOTOR_B_DIR_PIN, GPIO.HIGH if right_speed >= 0 else GPIO.LOW)
        self.motor_b_pwm.ChangeDutyCycle(abs(right_speed))

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
        print("Cleaning up GPIO.")
        self.stop()
        GPIO.cleanup()

# --- SENSOR & AI THREADS ---
def lidar_thread():
    """Continuously reads Lidar data in a background thread."""
    global closest_lidar_distance
    print("Lidar thread started.")
    try:
        class LaserPoint(ctypes.Structure):
            fields = [("angle", ctypes.c_float), ("range", ctypes.c_float), ("intensity", ctypes.c_float)]
        
        while not shutdown_flag.is_set():
            simulated_points = []
            for angle in range(-180, 180, 2):
                dist = 0.3 if -10 <= angle <= 10 and (time.time() % 10) > 5 else 5.0
                simulated_points.append(LaserPoint(float(angle), dist, 100.0))
            
            min_dist = float('inf')
            for point in simulated_points:
                if LIDAR_FRONT_ANGLE_RANGE[0] <= point.angle <= LIDAR_FRONT_ANGLE_RANGE[1]:
                    if 0.1 < point.range < min_dist:
                        min_dist = point.range
            
            with data_lock:
                closest_lidar_distance = min_dist
            
            time.sleep(LIDAR_SAMPLE_INTERVAL)
    except Exception as e:
        print(f"Lidar library error: {e}. Obstacle avoidance is disabled.")
    print("Lidar thread shutting down.")

def camera_ai_thread():
    """Captures video, runs AI inference, and updates target position."""
    global target_center_x
    
    time.sleep(2) # Stability delay
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    label_path = os.path.join(MODEL_DIR, LABEL_NAME)
    if not all(os.path.exists(p) for p in [model_path, label_path]):
        print("AI ERROR: AI model or label file not found. AI thread will not start.")
        return
    try:
        with open(label_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("AI ERROR: Cannot open camera. AI thread will not start.")
            return
            
        print("Camera & AI thread started successfully.")
        frame_counter = 0
        while not shutdown_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("AI WARNING: Dropped a camera frame.")
                time.sleep(0.1)
                continue
            frame_counter += 1
            if frame_counter % AI_PROCESS_EVERY_N_FRAMES != 0:
                continue
            imH, imW, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]
            classes = interpreter.get_tensor(output_details[1]['index'])[0]
            scores = interpreter.get_tensor(output_details[2]['index'])[0]
            
            found_target_in_frame = False
            
            if (scores > MIN_CONF_THRESHOLD) and (scores <= 1.0):
                object_name = labels[int(classes)]
                if object_name == TARGET_OBJECT:
                    ymin = int(max(1, boxes[0] * imH))
                    xmin = int(max(1, boxes[1] * imW))
                    ymax = int(min(imH, boxes[2] * imH))
                    xmax = int(min(imW, boxes[3] * imW))
                    center_pos = (xmin + xmax) // 2
                    with data_lock:
                        target_center_x = center_pos
                    found_target_in_frame = True
            
            if not found_target_in_frame:
                with data_lock:
                    target_center_x = -1
    except Exception as e:
        print(f"AI ERROR: An exception occurred in the AI thread: {e}")
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Camera & AI thread shutting down.")

def imu_thread():
    """Continuously reads MPU6050 data."""
    global imu_data
    PWR_MGMT_1, ACCEL_XOUT_H = 0x6B, 0x3B
    
    def read_word_2c(bus, reg):
        high = bus.read_byte_data(IMU_I2C_ADDRESS, reg)
        low = bus.read_byte_data(IMU_I2C_ADDRESS, reg + 1)
        val = (high << 8) + low
        return -((65535 - val) + 1) if val >= 0x8000 else val
        
    try:
        bus = smbus2.SMBus(1)
        bus.write_byte_data(IMU_I2C_ADDRESS, PWR_MGMT_1, 0)
        print("IMU thread started successfully.")
        while not shutdown_flag.is_set():
            x = read_word_2c(bus, ACCEL_XOUT_H) / 16384.0
            y = read_word_2c(bus, ACCEL_XOUT_H + 2) / 16384.0
            z = read_word_2c(bus, ACCEL_XOUT_H + 4) / 16384.0
            
            pitch = math.atan2(y, math.sqrt(x**2 + z**2)) * 180 / math.pi
            roll = math.atan2(-x, z) * 180 / math.pi
            
            with data_lock:
                imu_data['pitch'], imu_data['roll'] = pitch, roll
                
            time.sleep(IMU_GPS_SAMPLE_INTERVAL)
            
    except FileNotFoundError:
        print("IMU ERROR: I2C bus not found. Check if I2C is enabled in raspi-config.")
    except Exception as e:
        print(f"IMU ERROR: {e}")
    finally:
        print("IMU thread shutting down.")

def gps_thread():
    """Continuously reads NEO-6M data."""
    global gps_data
    
    def nmea_to_decimal(nmea_val, direction):
        degrees = float(nmea_val) // 100
        minutes = float(nmea_val) % 100
        decimal = degrees + minutes / 60.0
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal
        
    try:
        ser = serial.Serial(GPS_SERIAL_PORT, 9600, timeout=1)
        print("GPS thread started successfully.")
        while not shutdown_flag.is_set():
            line = ser.readline().decode('utf-8', errors='ignore')
            if line.startswith('$GPGGA'):
                parts = line.split(',')
                try:
                    if parts[6] != '0':
                        lat = nmea_to_decimal(parts[2], parts[3])
                        lon = nmea_to_decimal(parts[4], parts[5])
                        with data_lock:
                            gps_data['fix'] = True
                            gps_data['lat'] = lat
                            gps_data['lon'] = lon
                    else:
                        with data_lock:
                            gps_data['fix'] = False
                except (ValueError, IndexError):
                    with data_lock:
                        gps_data['fix'] = False
            # A small sleep to prevent this thread from hogging CPU if readline returns instantly
            time.sleep(0.1) 
    except serial.SerialException:
        print(f"GPS ERROR: Could not open port {GPS_SERIAL_PORT}. GPS data disabled.")
    except Exception as e:
        print(f"GPS ERROR: An exception occurred: {e}")
    finally:
        print("GPS thread shutting down.")

def cellular_location_thread():
    """Gets cell tower info."""
    pass

# --- WEB SERVER & REMOTE CONTROL THREAD ---
HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Robot Control</title>
  <style>
    body { font-family: sans-serif; display:flex; flex-direction:column; align-items:center; gap:12px; padding:12px; background-color: #f0f0f0; }
    h2 { margin: 0; }
    button { font-size:20px; padding:14px 18px; border-radius:8px; border: 1px solid #ccc; cursor: pointer; }
    .row { display:flex; gap:8px; }
    #status { font-size:14px; color:gray; }
    input[type=range] { width:240px; }
    #mode_button { background-color: #ffc107; font-weight: bold; width: 240px; }
    #stop { background-color: #dc3545; color: white; }
  </style>
</head>
<body>
  <h2>Robot Control Panel</h2>
  <button id="mode_button">Switch to Autonomous</button>
  <div class="row"><button id="fwd">▲</button></div>
  <div class="row"><button id="left">◄</button><button id="stop">Stop</button><button id="right">►</button></div>
  <div class="row"><button id="back">▼</button></div>
  <div>Speed: <span id="spdval">50</span><br/><input id="speed" type="range" min="20" max="100" value="50"/></div>
  <div id="status">Connecting...</div>
  <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
  <script>
    const socket = io();
    const status = document.getElementById('status');
    const spd = document.getElementById('speed');
    const spdval = document.getElementById('spdval');
    const modeBtn = document.getElementById('mode_button');
    let currentMode = 'MANUAL';

    function send(action) {
      if (currentMode !== 'MANUAL') return;
      const payload = { action: action, speed: parseInt(spd.value, 10) };
      socket.emit('cmd', payload);
    }
    
    function switchMode() { socket.emit('switch_mode'); }
    
    socket.on('connect', () => status.textContent = 'Connected');
    socket.on('disconnect', () => status.textContent = 'Disconnected');
    
    socket.on('mode_update', (data) => {
        currentMode = data.mode;
        status.textContent = `Connected | Mode: ${currentMode}`;
        if (currentMode === 'AUTO') {
            modeBtn.textContent = 'Switch to Manual';
            modeBtn.style.backgroundColor = '#28a745';
        } else {
            modeBtn.textContent = 'Switch to Autonomous';
            modeBtn.style.backgroundColor = '#ffc107';
        }
    });
    
    document.getElementById('fwd').onmousedown = () => send('forward');
    document.getElementById('back').onmousedown = () => send('backward');
    document.getElementById('left').onmousedown = () => send('left');
    document.getElementById('right').onmousedown = () => send('right');
    
    ['fwd', 'back', 'left', 'right'].forEach(id => {
        document.getElementById(id).onmouseup = () => send('stop');
        document.getElementById(id).ontouchend = () => send('stop');
    });
    
    document.getElementById('stop').onclick = () => send('stop');
    modeBtn.onclick = switchMode;
    spd.oninput = () => spdval.textContent = spd.value;
  </script>
</body>
</html>
"""

app = Flask(__name__)
socketio = SocketIO(app)

def web_server_thread():
    """Runs the Flask web server to handle remote control."""
    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE)

    @socketio.on('connect')
    def on_connect():
        with data_lock:
            current_mode = control_mode
        emit('mode_update', {'mode': current_mode})

    @socketio.on('cmd')
    def on_cmd(data):
        global last_manual_command_time, control_mode
        with data_lock:
            if control_mode != 'MANUAL':
                return
            action = data.get('action')
            speed = int(data.get('speed', 50))
            if action == 'forward': motors.forward(speed)
            elif action == 'backward': motors.backward(speed)
            elif action == 'left': motors.turn_left(speed)
            elif action == 'right': motors.turn_right(speed)
            elif action == 'stop': motors.stop()
            last_manual_command_time = time.time()

    @socketio.on('switch_mode')
    def on_switch_mode():
        global control_mode
        with data_lock:
            if control_mode == 'MANUAL':
                control_mode = 'AUTO'
                print("Switched to AUTO mode.")
            else:
                control_mode = 'MANUAL'
                print("Switched to MANUAL mode.")
                motors.stop()
        socketio.emit('mode_update', {'mode': control_mode})

    print("Starting web server on http://0.0.0.0:5000")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

# --- Main Control Script ---
def main():
    global motors
    motors = MotorController()
    
    threads = {
        "lidar": threading.Thread(target=lidar_thread),
        "camera_ai": threading.Thread(target=camera_ai_thread),
        "imu": threading.Thread(target=imu_thread),
        "gps": threading.Thread(target=gps_thread),
        "web_server": threading.Thread(target=web_server_thread)
    }
    
    if ENABLE_CELLULAR_LOCATION:
        threads["cellular"] = threading.Thread(target=cellular_location_thread)
    
    threads["web_server"].daemon = True
    
    for t in threads.values():
        t.start()
        
    print("\nStarting main control loop. Press Ctrl+C to exit.")
    
    frame_center, turn_threshold = 320, 50
    
    try:
        while not shutdown_flag.is_set():
            with data_lock:
                obstacle_dist = closest_lidar_distance
                target_pos = target_center_x
                current_imu = imu_data.copy()
                current_gps = gps_data.copy()
                current_cel = cellular_location_data.copy()
                mode = control_mode
                
            action = "Idle"
            
            if mode == 'AUTO':
                if obstacle_dist < LIDAR_OBSTACLE_DISTANCE:
                    action = "AUTO: Obstacle! STOPPING"
                    motors.stop()
                elif target_pos != -1:
                    action = "AUTO: Target Found! STOPPING"
                    motors.stop()
                else:
                    action = "AUTO: Searching... MOVING FORWARD"
                    motors.forward()
            
            elif mode == 'MANUAL':
                if time.time() - last_manual_command_time > MANUAL_COMMAND_TIMEOUT:
                    motors.stop()
                    action = "MANUAL: Idle (Timeout)"
                else:
                    action = "MANUAL: Awaiting command"
                    
            if current_gps['fix']:
                loc_str = f"GPS(Fix):{current_gps['lat']:.4f},{current_gps['lon']:.4f}"
            elif current_cel['fix'] and ENABLE_CELLULAR_LOCATION:
                loc_str = f"CEL(Est):{current_cel['lat']:.4f},{current_cel['lon']:.4f} (~{current_cel['accuracy']}m)"
            else:
                loc_str = "No Location Fix"
            
            imu_str = f"P:{current_imu['pitch']:.0f},R:{current_imu['roll']:.0f}"
            
            print(f"\rMODE: {mode} | Obst:{obstacle_dist:.2f}m | Target:{target_pos} | IMU:[{imu_str}] | LOC:[{loc_str}] | Action: {action.ljust(35)}", end="")
            time.sleep(MAIN_LOOP_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nShutdown signal received.")
    finally:
        print("\nInitiating shutdown...")
        shutdown_flag.set()
        for name, t in threads.items():
            if not t.daemon:
                t.join()
        motors.cleanup()
        print("Application shut down cleanly.")

if __name__ == '__main__':
    main()
