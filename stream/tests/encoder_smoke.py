"""Run inside stream/Dockerfile image, without a live ROS graph or simulator."""

import importlib.util
import socket
import struct
import time

import rclpy
from sensor_msgs.msg import Image

spec = importlib.util.spec_from_file_location("publisher", "/opt/camera/ros_rtp_publisher.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rclpy.init()
node = module.CameraPublisher()
receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver.bind(("127.0.0.1", 5004))
receiver.settimeout(5)
try:
    # Padded ROS rows exercise both input and Gst row strides.
    for sequence in range(3):
        image = Image()
        image.width, image.height, image.encoding, image.step = 66, 48, "rgb8", 203
        image.data = bytes([30 + sequence * 50] * (image.step * image.height))
        image.header.frame_id = "head_camera_optical_frame"
        image.header.stamp.sec, image.header.stamp.nanosec = 100 + sequence, 123456789
        node.image(image)
        timestamps = set()
        while True:
            try:
                packet = receiver.recv(2048)
            except TimeoutError:
                node.check_bus()
                raise
            assert packet[0] >> 6 == 2
            timestamps.add(struct.unpack_from("!I", packet, 4)[0])
            if packet[1] & 128:
                break
        assert len(timestamps) == 1
        time.sleep(0.02)
        node.check_bus()
    print("PASS: standard H264 RTP, including padded ROS/Gst rows")
finally:
    node.destroy_node()
    rclpy.shutdown()
    receiver.close()
