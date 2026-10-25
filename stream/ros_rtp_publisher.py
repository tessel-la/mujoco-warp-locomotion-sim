"""Publish an ordinary ROS camera as H.264 RTP for MediaMTX WHEP clients."""

from __future__ import annotations

import os
import time

import gi
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

FORMATS = {
    "rgb8": ("RGB", 3),
    "bgr8": ("BGR", 3),
    "rgba8": ("RGBA", 4),
    "bgra8": ("BGRA", 4),
    "mono8": ("GRAY8", 1),
}


class CameraPublisher(Node):
    def __init__(self):
        super().__init__("microduck_webrtc_publisher")
        Gst.init(None)
        self.pipeline = None
        self.shape = None
        self.started_ns = time.monotonic_ns()
        topic = os.environ.get("STREAM_IMAGE_TOPIC", "/locomotion/camera/image_raw")
        self.create_subscription(Image, topic, self.image, qos_profile_sensor_data)
        self.create_timer(1.0, self.check_bus)

    def start(self, width, height, encoding):
        port = int(os.environ.get("STREAM_RTP_INPUT_PORT", "5004"))
        if not 1 <= port <= 65535:
            raise ValueError("Invalid RTP input port")
        # A nominal rate avoids x264's one-frame lookahead for unspecified-rate
        # inputs. Actual PTS follows monotonic acquisition here, including drops.
        self.pipeline = Gst.parse_launch(
            "appsrc name=input is-live=true format=time block=false "
            "max-buffers=1 max-bytes=0 leaky-type=downstream "
            f'caps="video/x-raw,format={FORMATS[encoding][0]},'
            f'width={width},height={height},framerate=15/1" '
            "! videoconvert ! video/x-raw,format=I420 "
            "! x264enc tune=zerolatency speed-preset=ultrafast "
            "bframes=0 key-int-max=15 bitrate=2500 "
            "! video/x-h264,profile=constrained-baseline,alignment=au "
            "! rtph264pay pt=96 config-interval=-1 mtu=1200 "
            f"! udpsink host=127.0.0.1 port={port} sync=false async=false"
        )
        self.input = self.pipeline.get_by_name("input")
        if self.pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Camera encoder could not start")
        self.shape = (width, height, encoding)

    def image(self, image):
        width, height, encoding = image.width, image.height, image.encoding
        channels = FORMATS.get(encoding, (None, 0))[1]
        if (
            not channels
            or width % 2
            or height % 2
            or not 0 < width <= 8192
            or not 0 < height <= 8192
            or width * height > 8388608
        ):
            self.get_logger().error("Unsupported camera dimensions/encoding")
            return
        row = width * channels
        if image.step < row or len(image.data) < image.step * height:
            return
        if self.shape != (width, height, encoding):
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self.start(width, height, encoding)
        # ROS rows can be padded; Gst RGB rows require four-byte-aligned stride.
        stride = (row + 3) & ~3
        pixels = bytearray(stride * height)
        for y in range(height):
            pixels[y * stride : y * stride + row] = image.data[
                y * image.step : y * image.step + row
            ]
        buffer = Gst.Buffer.new_allocate(None, len(pixels), None)
        buffer.fill(0, pixels)
        buffer.pts = time.monotonic_ns() - self.started_ns
        buffer.duration = Gst.SECOND // 15
        if self.input.emit("push-buffer", buffer) != Gst.FlowReturn.OK:
            raise RuntimeError("Camera encoder rejected a frame")

    def check_bus(self):
        if self.pipeline:
            message = self.pipeline.get_bus().pop_filtered(Gst.MessageType.ERROR)
            if message:
                raise RuntimeError(message.parse_error()[0].message)

    def destroy_node(self):
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
