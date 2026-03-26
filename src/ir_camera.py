#!/usr/bin/python3

import cv2
import glob
import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class IRCameraPublisher(Node):
    def __init__(self):
        super().__init__('ir_camera_publisher')
        self.devices = get_video_devices()

        self.publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.01, self.publish_frame)
        self.idx = 0
        self.create_pipeline()
        self.bridge = CvBridge()

    def create_pipeline(self):
        self.dev = self.devices[self.idx]
        self.cap = cv2.VideoCapture(self.dev)

    def publish_frame(self):
        
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error(f'stream loc: {self.dev}')
            self.get_logger().warn("Failed to read ir frame")
            self.idx = abs(self.idx - 1)
            self.create_pipeline()
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        self.publisher.publish(msg)


def get_video_devices():
    video_devices = glob.glob('/dev/video*')
    print(video_devices)
    return video_devices

def main():
    rclpy.init()
    node = IRCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()