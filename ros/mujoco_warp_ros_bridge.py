#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import TransformStamped, Twist
from microduck_interfaces.action import ExecuteMotion
from nav_msgs.msg import Odometry
from PIL import Image as PillowImage
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, SetBool, Trigger
from tf2_ros import TransformBroadcaster


class MujocoWarpRosBridge(Node):
    POLICY_MOTIONS = {
        "walk",
        "stand",
        "sit",
        "ground_pick",
        "kick_left",
        "kick_right",
        "roulade",
        "recover",
    }
    MOTIONS = POLICY_MOTIONS | {"jump"}

    def __init__(self) -> None:
        super().__init__("mujoco_warp_locomotion_bridge")
        self.simulator_url = os.environ.get("MJWARP_SIM_URL", "http://127.0.0.1:8010").rstrip("/")
        self.state_hz = float(os.environ.get("MJWARP_ROS_STATE_HZ", "50"))
        self.camera_hz = float(os.environ.get("MJWARP_ROS_CAMERA_HZ", "15"))
        self.cmd_topic = os.environ.get("MJWARP_CMD_TOPIC", "/cmd_vel")
        self.odom_topic = os.environ.get("MJWARP_ODOM_TOPIC", "/odom")
        self.imu_topic = os.environ.get("MJWARP_IMU_TOPIC", "/imu/data")
        self.camera_topic = os.environ.get("MJWARP_CAMERA_TOPIC", "/locomotion/camera/image_raw")
        self.odom_frame = os.environ.get("MJWARP_ODOM_FRAME", "odom")
        self.base_frame = os.environ.get("MJWARP_BASE_FRAME", "base_link")
        self.camera_frame = os.environ.get(
            "MJWARP_CAMERA_FRAME", "head_camera_optical_frame"
        )
        self.cmd_timeout = float(os.environ.get("MJWARP_CMD_TIMEOUT", "0.5"))
        self.camera_fov = math.radians(float(os.environ.get("MJWARP_CAMERA_FOV", "72")))
        self.ready_file = Path(os.environ.get("MJWARP_ROS_READY_FILE", "/tmp/mjwarp-ros-ready"))

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self.odom_publisher = self.create_publisher(Odometry, self.odom_topic, 10)
        self.imu_publisher = self.create_publisher(Imu, self.imu_topic, sensor_qos)
        self.clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self.image_publisher = self.create_publisher(Image, self.camera_topic, sensor_qos)
        self.compressed_publisher = self.create_publisher(
            CompressedImage, f"{self.camera_topic}/compressed", sensor_qos
        )
        self.camera_info_publisher = self.create_publisher(
            CameraInfo, f"{self.camera_topic.rsplit('/', 1)[0]}/camera_info", sensor_qos
        )
        self.status_publisher = self.create_publisher(String, "/locomotion/status", status_qos)
        self.diagnostics_publisher = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.cmd_subscription = self.create_subscription(
            Twist, self.cmd_topic, self._receive_command, 10
        )
        self.estop_subscription = self.create_subscription(
            Bool, "/emergency_stop", self._receive_estop, 10
        )
        self.policy_subscription = self.create_subscription(
            String, "/locomotion/policy_cmd", self._receive_policy, 10
        )
        self.reset_service = self.create_service(Empty, "/locomotion/reset", self._reset)
        self.enable_service = self.create_service(SetBool, "/locomotion/enable", self._enable)
        self.mouth_service = self.create_service(SetBool, "/locomotion/mouth", self._mouth)
        self.mouth_open_service = self.create_service(
            Trigger, "/locomotion/mouth/open", self._mouth_open
        )
        self.mouth_close_service = self.create_service(
            Trigger, "/locomotion/mouth/close", self._mouth_close
        )
        self.jump_service = self.create_service(Trigger, "/locomotion/jump", self._jump)
        self._motion_callback_group = ReentrantCallbackGroup()
        self.motion_action = ActionServer(
            self,
            ExecuteMotion,
            "/locomotion/execute_motion",
            self._execute_motion,
            goal_callback=self._motion_goal,
            cancel_callback=self._motion_cancel,
            callback_group=self._motion_callback_group,
        )

        self._last_cmd_at = 0.0
        self._watchdog_stopped = True
        self._last_warning = 0.0
        self._camera_lock = threading.Lock()
        self._camera_jpeg: bytes | None = None
        self._camera_sequence = 0
        self._published_camera_sequence = 0
        self._stop = threading.Event()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, name="mjwarp-camera", daemon=True
        )
        self._camera_thread.start()
        self.create_timer(1.0 / self.state_hz, self._publish_state)
        self.create_timer(1.0 / self.camera_hz, self._publish_camera)
        self.create_timer(0.1, self._command_watchdog)
        self.get_logger().info(
            f"Bridging {self.simulator_url} to ROS domain {os.environ.get('ROS_DOMAIN_ID', '0')}"
        )
        self.get_logger().info(
            f"Drive: {self.cmd_topic}; state: {self.odom_topic}, {self.imu_topic}, /joint_states"
        )

    def destroy_node(self) -> bool:
        self._stop.set()
        self._camera_thread.join(timeout=2.0)
        self.ready_file.unlink(missing_ok=True)
        self.motion_action.destroy()
        return super().destroy_node()

    def _receive_command(self, message: Twist) -> None:
        values = (float(message.linear.x), float(message.linear.y), float(message.angular.z))
        if not all(math.isfinite(value) for value in values):
            self._warn_throttled("Ignoring cmd_vel with non-finite values")
            return
        try:
            self._request_json(
                "/api/v1/command",
                method="POST",
                payload={"linear_x": values[0], "linear_y": values[1], "angular_z": values[2]},
            )
            self._last_cmd_at = time.monotonic()
            self._watchdog_stopped = max(abs(value) for value in values) <= 1e-9
        except Exception as error:
            self._warn_throttled(f"Could not forward cmd_vel: {error}")

    def _command_watchdog(self) -> None:
        if self._watchdog_stopped or time.monotonic() - self._last_cmd_at <= self.cmd_timeout:
            return
        try:
            self._request_json(
                "/api/v1/command",
                method="POST",
                payload={"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0},
            )
            self._watchdog_stopped = True
        except Exception as error:
            self._warn_throttled(f"Could not stop stale cmd_vel: {error}")

    def _receive_estop(self, message: Bool) -> None:
        try:
            self._request_json(
                "/api/v1/enable", method="POST", payload={"enabled": not message.data}
            )
            if message.data:
                self._watchdog_stopped = True
        except Exception as error:
            self._warn_throttled(f"Could not apply emergency stop: {error}")

    def _receive_policy(self, message: String) -> None:
        try:
            self._request_json(
                "/api/v1/policy", method="POST", payload={"mode": message.data.strip()}
            )
        except Exception as error:
            self._warn_throttled(f"Could not switch locomotion policy: {error}")

    def _reset(self, _: Empty.Request, response: Empty.Response) -> Empty.Response:
        try:
            self._request_json("/api/v1/reset", method="POST", payload={})
            self._watchdog_stopped = True
        except Exception as error:
            self._warn_throttled(f"Reset failed: {error}")
        return response

    def _enable(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        try:
            result = self._request_json(
                "/api/v1/enable", method="POST", payload={"enabled": bool(request.data)}
            )
            response.success = True
            response.message = f"locomotion enabled={result['enabled']}"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _set_mouth(self, opened: bool) -> str:
        result = self._request_json(
            "/api/v1/mouth", method="POST", payload={"opened": opened}
        )
        return f"mouth opened={bool(result['opened'])}"

    def _mouth(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        try:
            response.message = self._set_mouth(bool(request.data))
            response.success = True
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _mouth_open(self, _: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            response.message = self._set_mouth(True)
            response.success = True
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _mouth_close(self, _: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            response.message = self._set_mouth(False)
            response.success = True
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _jump(self, _: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self._request_json("/api/v1/jump", method="POST", payload={})
            response.success = True
            response.message = "jump requested"
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def _motion_goal(self, goal_request: ExecuteMotion.Goal) -> GoalResponse:
        motion = goal_request.motion.strip().lower()
        if motion not in self.MOTIONS:
            self.get_logger().warning(f"Rejecting unknown Microduck motion {motion!r}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    @staticmethod
    def _motion_cancel(_: Any) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_motion(self, goal_handle: Any) -> ExecuteMotion.Result:
        motion = goal_handle.request.motion.strip().lower()
        result = ExecuteMotion.Result()
        try:
            endpoint = "/api/v1/jump" if motion == "jump" else "/api/v1/policy"
            payload = {} if motion == "jump" else {"mode": motion}
            self._request_json(endpoint, method="POST", payload=payload)
            accepted_at = time.monotonic()
            started = False
            one_shot = motion in {
                "jump",
                "ground_pick",
                "kick_left",
                "kick_right",
                "roulade",
                "recover",
            }
            deadline = accepted_at + (8.0 if one_shot else 1.25)
            while time.monotonic() < deadline:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    result.message = f"{motion} canceled"
                    result.final_mode = "canceled"
                    return result
                state = self._request_json("/api/v1/state")
                active_mode = str(state.get("mode", "unknown"))
                policy_mode = str(state.get("policy", {}).get("mode", active_mode))
                feedback = ExecuteMotion.Feedback()
                feedback.phase = "running"
                feedback.active_mode = active_mode
                goal_handle.publish_feedback(feedback)
                if motion == "jump":
                    started = started or active_mode == "jumping"
                    complete = started and active_mode != "jumping"
                elif one_shot:
                    started = started or policy_mode == motion
                    complete = started and policy_mode == "walk"
                else:
                    expected_modes = {
                        motion,
                        "sitstand" if motion in {"sit", "stand"} else motion,
                    }
                    complete = (
                        time.monotonic() - accepted_at >= 0.75
                        and policy_mode in expected_modes
                    )
                if complete:
                    result.success = True
                    result.message = f"{motion} completed"
                    result.final_mode = active_mode
                    goal_handle.succeed()
                    return result
                time.sleep(0.05)
            if not one_shot:
                state = self._request_json("/api/v1/state")
                result.success = True
                result.message = f"{motion} accepted"
                result.final_mode = str(state.get("mode", motion))
                goal_handle.succeed()
                return result
            result.success = False
            result.message = f"{motion} did not complete before timeout"
            result.final_mode = "timeout"
            goal_handle.abort()
            return result
        except Exception as error:
            result.success = False
            result.message = str(error)
            result.final_mode = "error"
            goal_handle.abort()
            return result

    def _publish_state(self) -> None:
        try:
            state = self._request_json("/api/v1/state")
            stamp = self.get_clock().now().to_msg()
            self._publish_joint_state(state, stamp)
            self._publish_odometry_and_tf(state, stamp)
            self._publish_imu(state, stamp)
            self._publish_clock(float(state["simulation_time"]))
            self._publish_status(state, stamp)
            if not self.ready_file.exists():
                self.ready_file.touch()
        except Exception as error:
            self._warn_throttled(f"Could not publish simulator state: {error}")

    def _publish_joint_state(self, state: dict[str, Any], stamp: TimeMessage) -> None:
        message = JointState()
        message.header.stamp = stamp
        message.name = [str(name) for name in state["joint_names"]]
        message.position = [float(value) for value in state["joint_positions"]]
        message.velocity = [float(value) for value in state["joint_velocities"]]
        self.joint_publisher.publish(message)

    @staticmethod
    def _set_quaternion(target: Any, wxyz: list[float]) -> None:
        target.w, target.x, target.y, target.z = (
            float(wxyz[0]),
            float(wxyz[1]),
            float(wxyz[2]),
            float(wxyz[3]),
        )

    def _publish_odometry_and_tf(self, state: dict[str, Any], stamp: TimeMessage) -> None:
        position = state["base_pose"]["position"]
        quaternion = state["base_pose"]["quaternion_wxyz"]
        linear = state["base_twist"]["linear"]
        angular = state["base_twist"]["angular"]
        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self.odom_frame
        odometry.child_frame_id = self.base_frame
        (
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            odometry.pose.pose.position.z,
        ) = (float(position[0]), float(position[1]), float(position[2]))
        self._set_quaternion(odometry.pose.pose.orientation, quaternion)
        (
            odometry.twist.twist.linear.x,
            odometry.twist.twist.linear.y,
            odometry.twist.twist.linear.z,
        ) = (float(linear[0]), float(linear[1]), float(linear[2]))
        (
            odometry.twist.twist.angular.x,
            odometry.twist.twist.angular.y,
            odometry.twist.twist.angular.z,
        ) = (float(angular[0]), float(angular[1]), float(angular[2]))
        odometry.pose.covariance[0] = odometry.pose.covariance[7] = 0.0025
        odometry.pose.covariance[35] = 0.01
        odometry.twist.covariance[0] = odometry.twist.covariance[7] = 0.01
        odometry.twist.covariance[35] = 0.02
        self.odom_publisher.publish(odometry)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ) = (float(position[0]), float(position[1]), float(position[2]))
        self._set_quaternion(transform.transform.rotation, quaternion)
        self.tf_broadcaster.sendTransform(transform)

    def _publish_imu(self, state: dict[str, Any], stamp: TimeMessage) -> None:
        imu = state["imu"]
        message = Imu()
        message.header.stamp = stamp
        message.header.frame_id = str(imu.get("frame", "imu_link"))
        self._set_quaternion(message.orientation, imu["orientation_wxyz"])
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = (
            float(value) for value in imu["angular_velocity"]
        )
        (
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        ) = (float(value) for value in imu["linear_acceleration"])
        message.orientation_covariance[0] = message.orientation_covariance[
            4
        ] = message.orientation_covariance[8] = 0.0025
        message.angular_velocity_covariance[0] = message.angular_velocity_covariance[
            4
        ] = message.angular_velocity_covariance[8] = 0.01
        message.linear_acceleration_covariance[0] = message.linear_acceleration_covariance[
            4
        ] = message.linear_acceleration_covariance[8] = 0.04
        self.imu_publisher.publish(message)

    def _publish_clock(self, seconds: float) -> None:
        message = Clock()
        whole = int(seconds)
        message.clock.sec = whole
        message.clock.nanosec = int((seconds - whole) * 1_000_000_000)
        self.clock_publisher.publish(message)

    def _publish_status(self, state: dict[str, Any], stamp: TimeMessage) -> None:
        status = {
            "mode": state["mode"],
            "enabled": state["enabled"],
            "fallen": state["fallen"],
            "command_stale": state["command_stale"],
            "backend": state["backend"],
            "device": state["device"],
            "policy": state.get("policy", {}),
            "mouth": state.get("mouth", {}),
        }
        self.status_publisher.publish(String(data=json.dumps(status, separators=(",", ":"))))
        diagnostic = DiagnosticArray()
        diagnostic.header.stamp = stamp
        item = DiagnosticStatus()
        item.name = "mujoco_warp_locomotion_sim"
        item.hardware_id = "duckbot_sim"
        item.level = DiagnosticStatus.ERROR if state["fallen"] else DiagnosticStatus.OK
        item.message = str(state["mode"])
        item.values = [KeyValue(key=key, value=str(value)) for key, value in status.items()]
        diagnostic.status = [item]
        self.diagnostics_publisher.publish(diagnostic)

    def _camera_loop(self) -> None:
        interval = 1.0 / max(self.camera_hz, 1.0)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                with urllib.request.urlopen(
                    f"{self.simulator_url}/api/v1/camera/frame.jpg", timeout=2.0
                ) as response:
                    jpeg = response.read()
                with self._camera_lock:
                    self._camera_jpeg = jpeg
                    self._camera_sequence += 1
            except Exception as error:
                self._warn_throttled(f"Could not fetch camera frame: {error}")
            self._stop.wait(max(0.0, interval - (time.monotonic() - started)))

    def _publish_camera(self) -> None:
        with self._camera_lock:
            if (
                self._camera_jpeg is None
                or self._camera_sequence == self._published_camera_sequence
            ):
                return
            jpeg = self._camera_jpeg
            sequence = self._camera_sequence
        try:
            image = PillowImage.open(io.BytesIO(jpeg)).convert("RGB")
            width, height = image.size
            stamp = self.get_clock().now().to_msg()
            raw = Image()
            raw.header.stamp = stamp
            raw.header.frame_id = self.camera_frame
            raw.height, raw.width = height, width
            raw.encoding = "rgb8"
            raw.is_bigendian = 0
            raw.step = width * 3
            raw.data = image.tobytes()
            self.image_publisher.publish(raw)
            compressed = CompressedImage()
            compressed.header = raw.header
            compressed.format = "jpeg"
            compressed.data = jpeg
            self.compressed_publisher.publish(compressed)
            info = CameraInfo()
            info.header = raw.header
            info.width, info.height = width, height
            focal = 0.5 * width / math.tan(0.5 * self.camera_fov)
            info.k = [focal, 0.0, width / 2.0, 0.0, focal, height / 2.0, 0.0, 0.0, 1.0]
            info.p = [
                focal,
                0.0,
                width / 2.0,
                0.0,
                0.0,
                focal,
                height / 2.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
            info.distortion_model = "plumb_bob"
            info.d = [0.0] * 5
            self.camera_info_publisher.publish(info)
            self._published_camera_sequence = sequence
        except Exception as error:
            self._warn_throttled(f"Could not publish camera frame: {error}")

    def _request_json(
        self, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> Any:
        body = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{self.simulator_url}{path}",
            data=body,
            method=method,
            headers={"content-type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {detail}") from error

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= 5.0:
            self.get_logger().warning(message)
            self._last_warning = now


def main() -> None:
    rclpy.init()
    node = MujocoWarpRosBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
