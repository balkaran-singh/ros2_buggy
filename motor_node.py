#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_node')
        
        # Subscribe to the standard ROS 2 velocity topic
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        # --- Hardware Setup (Cytron MDD20A) ---
        self.PWM_L = 12  # Left Motor Speed
        self.DIR_L = 5   # Left Motor Direction
        self.PWM_R = 13  # Right Motor Speed
        self.DIR_R = 6   # Right Motor Direction

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup([self.PWM_L, self.DIR_L, self.PWM_R, self.DIR_R], GPIO.OUT)

        # Set PWM frequency to 1000 Hz
        self.pwm_left = GPIO.PWM(self.PWM_L, 1000)
        self.pwm_right = GPIO.PWM(self.PWM_R, 1000)
        
        self.pwm_left.start(0)
        self.pwm_right.start(0)

        self.get_logger().info('Motor Node Started. Listening for /cmd_vel...')

    def cmd_vel_callback(self, msg):
        # Extract forward/back (linear) and rotation (angular) speeds
        linear = msg.linear.x
        angular = msg.angular.z

        # Differential Drive Math (Mixing steering and throttle)
        left_speed = linear - angular
        right_speed = linear + angular

        # Convert to 0-100% duty cycle for the PWM pins
        left_pwm = min(max(abs(left_speed) * 100, 0), 100)
        right_pwm = min(max(abs(right_speed) * 100, 0), 100)

        # Set Directions (HIGH = Forward, LOW = Backward)
        GPIO.output(self.DIR_L, GPIO.HIGH if left_speed >= 0 else GPIO.LOW)
        GPIO.output(self.DIR_R, GPIO.HIGH if right_speed >= 0 else GPIO.LOW)

        # Apply Speeds
        self.pwm_left.ChangeDutyCycle(left_pwm)
        self.pwm_right.ChangeDutyCycle(right_pwm)

    def destroy_node(self):
        # Safety catch: Stop motors if node is killed
        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)
        GPIO.cleanup()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
