#!/usr/bin/python3

import cv2
import math
import rclpy
import tf2_ros
import numpy as np
import tf2_geometry_msgs

from rclpy.node import Node
from nav_msgs.msg import Path
from cv_bridge import CvBridge
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped, TransformStamped, Point
from scipy.ndimage import gaussian_filter1d
from tf2_ros import TransformException
from visualization_msgs.msg import MarkerArray, Marker

WIDTH = 0.4445#1.2446

class SegToWaypoints(Node):
    def __init__(self):
        super().__init__('seg_to_waypoints')
        self.subscription = self.create_subscription(Image, '/segmentation', self.seg_callback, 10)
        
        self.camera_matrix = np.array([
            [957.8161392, 0.00000000e+00, 598.3205973],
            [0.00000000, 959.84135412, 349.77426678],
            [0.00000000, 0.00000000, 1.00000000]
        ])
        self.dist_coeffs = np.array([-0.6037583, 0.39257259, 0.00861181, -0.00144358, -0.11123289])
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publisher = self.create_publisher(Path, '/waypoints', 10)
        self.img_publisher = self.create_publisher(Image, '/vis_trajectory',10)
        self.marker_publisher = self.create_publisher(MarkerArray, '/marker_array', 10)

        self.bridge = CvBridge()
        self.counter = 0
        self.image = np.zeros((352,672,3))

    def seg_callback(self, msg):
        print("Received segmentation image")
        cv_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        # print(self.image == cv_mask)
        self.image = cv_mask

        left_pts, right_pts, center_pts = self.getEdges(cv_mask)
        # self.get_logger().error(f'left_pts:: {left_pts}')
        # self.get_logger().error(f'left_pts:: {left_pts[0]}')

        self.get_logger().error(f'left_pts___ {left_pts}')
        self.get_logger().error(f'right_pts___ {right_pts}')
        self.get_logger().error(f'center_pts___ {center_pts}')
        # print('*' * 20)

        if len(left_pts[0]) == 0 or len(left_pts[1]) == 0:
            return
        
        left_pts = self.undistort_points(left_pts)
        right_pts = self.undistort_points(right_pts)
        center_pts = self.undistort_points(center_pts)
        self.get_logger().warn(f'left_pts___ {left_pts}')
        self.get_logger().warn(f'right_pts___ {right_pts}')
        self.get_logger().warn(f'center_pts___ {center_pts}')
        # print('*' * 20)

        # # print(len(center_pts))

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'base_link'

        marker_arr = MarkerArray()

        for i, (ux, uy) in enumerate(center_pts):
            
            if right_pts[i][0] == left_pts[i][0]:
                continue

            Z = (fy * WIDTH) / (abs(right_pts[i][0] - left_pts[i][0]))
            print('z::: ', Z)

            X = (ux - cy) / fy * Z
            Y = (uy - cx) / fx * Z

            # print('left-right: ', math.sqrt( (right_pts[i][0] - left_pts[i][0]) ** 2 + ((right_pts[i][1] - left_pts[i][1])) ** 2 ))
            # print(right_pts[i])
            # print(left_pts[i])
            # print(f'x, y, z: ({X},{Y},{Z})')
            u_x = 0
            u_y = 0
            ux, uy, uz = self.transform_point(X, Y, Z)
            self.get_logger().warn(f'i::::{i} ->>> ux: {float(ux)}, uy: {float(uy)}')
            ux = ux if not math.isnan(ux) or ux is not float('inf') else u_x
            uy = uy if not math.isnan(uy) or uy is not float('inf') else u_y
            # self.get_logger().warn(f'i::::{i} ->>> ux: {float(ux)}, uy: {float(uy)}')

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(ux)
            pose.pose.position.y = -float(uy)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "shapes"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position = Point(x=float(ux), y=-float(uy), z=0.0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            marker_arr.markers.append(marker)

            path_msg.poses.append(pose)
        print('*' * 20)
        xx = float('nan')

        def contains_nan_in_pose(pose):
            return (math.isnan(pose.pose.position.x) or 
                    math.isnan(pose.pose.position.y) or 
                    math.isnan(pose.pose.position.z))

        if any(contains_nan_in_pose(p) for p in path_msg.poses):
            print("Skipping publish because one or more positions are NaN")
        else:
            print(ux, uy, uz)
            self.publisher.publish(path_msg)
            self.marker_publisher.publish(marker_arr)

    def transform_point(self, X, Y, Z):
        # print(f'x, y, z: ({X},{Y},{Z})')
        try:
            requested_transform = TransformStamped()
            requested_transform = self.tf_buffer.lookup_transform('base_link', 'ir_camera_link', rclpy.time.Time())
        except TransformException as e:
            self.get_logger().error(f"error on transform: {e}")
            return 0, 0, 0

        
        p_cam = PointStamped()
        p_cam.header.frame_id = 'ir_camera_link'
        p_cam.header.stamp = self.get_clock().now().to_msg()
        p_cam.point.x = X
        p_cam.point.y = Y
        p_cam.point.z = Z

        p_robot = tf2_geometry_msgs.do_transform_point(p_cam, requested_transform)

        return p_robot.point.x, p_robot.point.y, p_robot.point.z
        
    
    def undistort_points(self, pts):

        pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        undist_pts = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)

        return undist_pts.reshape(-1, 2)

    def getEdges(self, mask, filter_window=8):

        def filteredROI(x_coords, y_coords, threshold=100):
            widths = []
            for rr in x_coords:
                widths.append(len(rr))

            filtered_xs = []
            filtered_ys = []

            for i in range(len(widths)):
                if widths[i] > threshold:
                    filtered_xs.append(x_coords[i])
                    filtered_ys.append(y_coords[i])

            filtered_xs = np.array([x for x in filtered_xs], dtype=object)
            filtered_ys = np.array([filtered_ys])

            return filtered_xs, filtered_ys
        
        def getROI(mask):
            x_coords, y_coords = [], []
            
            for i in range(mask[0].shape[0]):
                seg_arr = np.array(np.where(mask[i:i+1, :] == 255)[1])
                if len(seg_arr) != 0:
                    y_coords.append(i)
                    x_coords.append(seg_arr)

            x_coords = np.array([x for x in x_coords], dtype=object)
            y_coords = np.array(y_coords)

            return filteredROI(x_coords, y_coords)

        green_channel = mask[:,:,1:2]
        x_coords, y_coords = getROI(green_channel)

        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        y_coords = y_coords[0]
        
        for row_idx, xs in zip(y_coords, x_coords):
            #.astype(int)
            for i in range(len(xs)):
                new_mask[row_idx, xs[i]] = 1

        y_list, x_list, idx = np.where(new_mask == 1)

        avgs = []

        for i in range(len(x_coords)):
            avgs.append(np.mean(x_coords[i]))

        avgs = gaussian_filter1d(avgs, sigma=20)

        window = int(len(avgs)/filter_window)

        left_pts = []
        right_pts = []
        sampled_xs = []
        sampled_ys = []

        print('=' * 20)

        for i in range(len(avgs)):
            if window == 0:
                break
            if i % window == 0 and i != 0:
                sampled_xs.append(avgs[i])
                sampled_ys.append(y_coords[i])
                left_pts.append(np.min(x_coords[i]))
                right_pts.append(np.max(x_coords[i]))
                self.get_logger().error(f'{np.min(x_coords[i]) == np.max(x_coords[i])}')
                self.get_logger().warn(f'min: {np.min(x_coords[i])}, max: {np.max(x_coords[i])}')

        for x, y in zip(sampled_xs, sampled_ys):
            cv2.circle(mask, (int(x), int(y)), radius=10, color=(0,0,0), thickness=-1)
        for x, y in zip(left_pts, sampled_ys):
            cv2.circle(mask, (int(x), int(y)), radius=10, color=(255,255,0), thickness=-1)
        for x, y in zip(right_pts, sampled_ys):
            cv2.circle(mask, (int(x), int(y)), radius=10, color=(0,255,255), thickness=-1)

        self.img_publisher.publish(self.bridge.cv2_to_imgmsg(mask, encoding='rgb8'))

        return (np.array(left_pts), np.array(sampled_ys)), (np.array(right_pts), np.array(sampled_ys)), (np.array(sampled_xs), np.array(sampled_ys))
    

def main():
    print("....running")
    rclpy.init()
    node = SegToWaypoints()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()