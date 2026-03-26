#!/usr/bin/python3

import rclpy
import numpy as np
from rclpy.node import Node
import tf2_geometry_msgs

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PointStamped, TransformStamped
from tf_transformations import euler_from_quaternion
from tf2_ros import Buffer, TransformListener


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller")

        # --- Parameters ---
        self.declare_parameter('ld_gain', 1.0)
        self.declare_parameter('min_ld', 0.5)
        self.declare_parameter('max_linear_speed', 0.2)
        self.declare_parameter('controller_freq', 20)
        self.declare_parameter('map_frame', 'base_link')
        self.declare_parameter('base_frame', 'chassis_link')
        self.declare_parameter('goal_tolerance', 0.2)

        self.ld_gain = self.get_parameter('ld_gain').value
        self.min_ld = self.get_parameter('min_ld').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.controller_freq = self.get_parameter('controller_freq').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value

        # Lookahead distance (adaptive based on speed)
        self.ld = self.min_ld
        
        # Path tracking
        self.path = []
        self.point_idx = 0
        self.got_path = False
        self.goal_reached = False
        self.is_loop = False

        # Robot state
        self.car_speed = 0.0
        self.odom_received = False

        # TF2 for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Subscribers ---
        self.create_subscription(Odometry, "/odometry/wheels", self.odom_callback, 10)
        self.create_subscription(Path, "/waypoints", self.path_callback, 10)

        # --- Publishers ---
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.lookahead_pub = self.create_publisher(PointStamped, "/pure_pursuit/lookahead_point", 10)

        # --- Timer for control loop ---
        timer_period = 1.0 / self.controller_freq
        self.create_timer(timer_period, self.control_loop)

        self.get_logger().info("Pure Pursuit controller started.")
        self.get_logger().info(f"  ld_gain: {self.ld_gain}")
        self.get_logger().info(f"  min_ld: {self.min_ld}")
        self.get_logger().info(f"  max_speed: {self.max_linear_speed}")

    # ---------------------------------------------------------
    # Odometry callback - updates adaptive lookahead distance
    # ---------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        self.get_logger().warn('odom data collected....')
        # Get current linear speed
        self.car_speed = abs(msg.twist.twist.linear.x)
        
        # Adaptive lookahead: increases with speed
        self.ld = max(self.ld_gain * self.car_speed, self.min_ld)
        
        if not self.odom_received:
            self.get_logger().info("Odometry received.")
            self.odom_received = True

    # ---------------------------------------------------------
    # Receive path waypoints
    # ---------------------------------------------------------
    def path_callback(self, msg: Path):
        self.path = msg.poses
        self.got_path = True
        self.goal_reached = False
        self.point_idx = 0
        
        self.get_logger().info(f"Received path with {len(self.path)} waypoints")
        
        # Check if this is a loop (start and end are close)
        if len(self.path) > 1:
            start = self.path[0].pose.position
            end = self.path[-1].pose.position
            start_end_dist = self.distance(start, end)
            self.get_logger().info(f"Start to End Distance: {start_end_dist:.3f}")
            
            if start_end_dist <= self.min_ld:
                self.is_loop = True
                self.get_logger().info("Path is a loop: True")
            else:
                self.is_loop = False

    # ---------------------------------------------------------
    # Calculate Euclidean distance
    # ---------------------------------------------------------
    def distance(self, pt1, pt2):
        return np.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2 + (pt1.z - pt2.z)**2)

    # ---------------------------------------------------------
    # Find lookahead point on path
    # ---------------------------------------------------------
    def get_lookahead_point(self, base_location):
        """Find first point on path that is >= lookahead distance away"""
        
        robot_pos = base_location.transform.translation
        
        # Search from current index forward
        for i in range(self.point_idx, len(self.path)):
            path_point = self.path[i].pose.position
            dist = self.distance(path_point, robot_pos)
            
            if dist >= self.ld:
                self.point_idx = i
                return self.path[i], dist
        
        # If no point found, return last point
        self.point_idx = len(self.path) - 1
        return self.path[-1], self.distance(self.path[-1].pose.position, robot_pos)

    # ---------------------------------------------------------
    # Pure pursuit control for differential drive
    # ---------------------------------------------------------
    def pure_pursuit_control(self, target_point_base_frame):
        """
        Calculate velocities for differential drive using pure pursuit.
        For diff drive: omega = v * (2 * y_target / ld^2)
        """
        
        # Get y-component of target in base_link frame
        y_t = target_point_base_frame.pose.position.y
        x_t = target_point_base_frame.pose.position.x
        
        # Pure pursuit formula adapted for differential drive
        ld_squared = self.ld * self.ld
        
        # Curvature calculation
        curvature = 2.0 * y_t / ld_squared
        
        # Linear velocity
        linear_vel = self.max_linear_speed
        
        # Angular velocity
        angular_vel = linear_vel * curvature
        
        return linear_vel, angular_vel

    # ---------------------------------------------------------
    # Main control loop
    # ---------------------------------------------------------
    def control_loop(self):
        
        if not self.odom_received or not self.got_path:
            return

        if not self.path:
            return

        try:
            # Get robot position in map frame using TF2
            base_location = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
            
            robot_pos = base_location.transform.translation
            
            # Check if goal reached
            goal_pos = self.path[-1].pose.position
            dist_to_goal = self.distance(goal_pos, robot_pos)
            
            if dist_to_goal < self.goal_tolerance:
                # Stop the robot
                twist = Twist()
                self.cmd_pub.publish(twist)
                
                if not self.goal_reached:
                    self.get_logger().info("Goal reached!")
                    self.goal_reached = True
                    
                    if self.is_loop:
                        # Reset for loop
                        self.point_idx = 0
                        self.goal_reached = False
                    else:
                        self.got_path = False
                return
            
            # Find lookahead point
            lookahead_pose, distance_to_point = self.get_lookahead_point(base_location)
            
            # Transform lookahead point to base_link frame
            lookahead_pose.header.stamp = self.get_clock().now().to_msg()
            target_point_base = self.tf_buffer.transform(
                lookahead_pose,
                self.base_frame,
                rclpy.duration.Duration(seconds=0.1)
            )
            
            # Calculate control commands
            linear_vel, angular_vel = self.pure_pursuit_control(target_point_base)
            
            # Publish velocity commands
            twist = Twist()
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)
            self.cmd_pub.publish(twist)
            
            # Publish lookahead point for visualization
            lookahead_point = PointStamped()
            lookahead_point.header = lookahead_pose.header
            lookahead_point.point = lookahead_pose.pose.position
            self.lookahead_pub.publish(lookahead_point)
            
            # Check if we've reached the end of path
            if self.point_idx >= len(self.path) - 1:
                if self.is_loop:
                    self.point_idx = 0
                else:
                    self.get_logger().info("Reached final waypoint")
            
        except Exception as e:
            self.get_logger().warn(f"Transform exception: {str(e)}")


def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()