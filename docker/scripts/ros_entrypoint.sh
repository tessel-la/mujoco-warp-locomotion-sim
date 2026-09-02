#!/bin/bash
set -eo pipefail

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export ROS_STATIC_PEERS="${ROS_STATIC_PEERS:-}"
if [ "${RMW_IMPLEMENTATION}" = "rmw_fastrtps_cpp" ]; then
  export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
fi
if [ "${RMW_IMPLEMENTATION}" = "rmw_cyclonedds_cpp" ] && [ -z "${CYCLONEDDS_URI:-}" ] && [ -f /etc/cyclonedds/config.xml ]; then
  export CYCLONEDDS_URI="file:///etc/cyclonedds/config.xml"
fi

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "/opt/tessella/ros_ws/install/setup.bash"
INTERFACE_EXPORT="${MJWARP_INTERFACE_EXPORT:-/overlay_export}"
if [ -d "${INTERFACE_EXPORT}" ]; then
  mkdir -p "${INTERFACE_EXPORT}"
  cp -a /opt/tessella/ros_ws/install/. "${INTERFACE_EXPORT}/"
fi
echo "DDS middleware: ${RMW_IMPLEMENTATION}; ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
ROBOT_URDF="${MJWARP_ROBOT_URDF:-/app/assets/microduck/microduck.urdf}"
MESH_ROOT="${MJWARP_MESH_ROOT:-/app/assets}"
MESH_PORT="${MJWARP_MESH_PORT:-8000}"

cleanup() {
  kill "${BRIDGE_PID:-}" "${STATE_PUBLISHER_PID:-}" "${MESH_SERVER_PID:-}" 2>/dev/null || true
  wait "${BRIDGE_PID:-}" "${STATE_PUBLISHER_PID:-}" "${MESH_SERVER_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

python3 -u /opt/tessella/cors_mesh_server.py "${MESH_PORT}" "${MESH_ROOT}" &
MESH_SERVER_PID=$!
ros2 run robot_state_publisher robot_state_publisher "${ROBOT_URDF}" &
STATE_PUBLISHER_PID=$!
python3 -u /opt/tessella/mujoco_warp_ros_bridge.py &
BRIDGE_PID=$!
wait -n "${BRIDGE_PID}" "${STATE_PUBLISHER_PID}" "${MESH_SERVER_PID}"
