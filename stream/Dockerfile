ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO}-ros-core
ARG ROS_DISTRO
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly \
    && rm -rf /var/lib/apt/lists/*
COPY stream/ros_rtp_publisher.py /opt/camera/ros_rtp_publisher.py
CMD ["python3", "/opt/camera/ros_rtp_publisher.py"]
