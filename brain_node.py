#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
import math 

class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')
        
        # Store the latest data in memory
        self.user_cmd = Twist()
        self.closest_obstacle_distance = 999.0 
        self.is_braking = False
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
            
        self.web_sub = self.create_subscription(
            Twist, '/web_cmd_vel', self.web_callback, 10)
            
        self.motor_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # THE HEARTBEAT: Run the control loop 20 times every second (0.05s)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("V5 Brain Node: 20Hz Active Control Loop Online.")

    def scan_callback(self, msg):
        num_points = len(msg.ranges)
        if num_points == 0: return
            
        points_per_degree = num_points / 360.0
        arc_size = int(points_per_degree * 15) 
        front_ranges = msg.ranges[-arc_size:] + msg.ranges[:arc_size]
        
        crash_zone = [r for r in front_ranges if 0.01 < r <= 0.2]
        blinded_rays = [r for r in front_ranges if r <= 0.01 or math.isinf(r) or math.isnan(r)]
        
        if len(crash_zone) > 0 or len(blinded_rays) > (len(front_ranges) * 0.5):
            self.closest_obstacle_distance = 0.0 
            return
            
        valid_ranges = [r for r in front_ranges if 0.2 < r < 10.0]
        if len(valid_ranges) > 0:
            self.closest_obstacle_distance = min(valid_ranges)
        else:
            self.closest_obstacle_distance = 999.0

    def web_callback(self, msg):
        # Just save the user's command to memory. Do NOT publish it here!
        self.user_cmd = msg

    def control_loop(self):
        # This function runs continuously at 20Hz
        safe_cmd = Twist()
        safe_cmd.linear.x = self.user_cmd.linear.x
        safe_cmd.angular.z = self.user_cmd.angular.z
        
        # If the user is trying to drive forward
        if self.user_cmd.linear.x > 0.0:
            
            # Massive safety bubble for high speeds (0.5m base + up to 0.7m speed factor)
            dynamic_safe_distance = 0.5 + (self.user_cmd.linear.x * 0.7)
            
            # If a wall breaches the bubble, CUT POWER!
            if self.closest_obstacle_distance <= 0.0 or self.closest_obstacle_distance < dynamic_safe_distance:
                if not self.is_braking:
                    self.get_logger().warn(f'ACTIVE BRAKE TRIGGERED! Wall at {self.closest_obstacle_distance:.2f}m')
                    self.is_braking = True
                    
                safe_cmd.linear.x = 0.0  # Override forward speed to 0
            else:
                self.is_braking = False
                
        # Constantly push the approved command to the wheels
        self.motor_pub.publish(safe_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()