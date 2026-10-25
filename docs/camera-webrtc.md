# Optional camera WebRTC gateway

The `webrtc` Compose profile adds a standard MediaMTX WHEP stream named
`microduck_head_camera`. It is optional: existing simulator, ROS image and MJPEG
outputs work as before. No custom timestamp messages or metadata package is required.
The encoder subscribes to the standard `/locomotion/camera/image_raw` ROS Image
using sensor-data QoS, encodes H.264 once, and sends local RTP to MediaMTX. Browsers
receive ordinary WebRTC video; the gateway handles multiple readers.

After starting the simulator and ROS bridge using the existing GPU or CPU Compose
configuration, start just the optional services with the same Compose files:

```sh
docker compose --profile webrtc up -d --build --no-deps \
  microduck_stream_gateway microduck_camera_encoder
```

For a CPU deployment, include `-f docker-compose.yml -f docker-compose.cpu.yml`
before `--profile`. This profile is not enabled by default. Stopping these two
services does not stop the simulator or ROS bridge.

Default endpoints:

- WHEP: `http://ROBOT_HOST:8889/microduck_head_camera/whep`
- Discovery for a local Robo-Boy relay: `http://127.0.0.1:9997/v3/paths/list`
- ICE media: UDP/TCP port `8189` (required in addition to WHEP signaling).

Robo-Boy's existing WebRTC external panel can discover this stream through its
configured gateway relay. The built-in Camera/3D WebRTC source can also use its
WHEP URL directly. If a host already runs another gateway on these ports, configure
`STREAM_WHEP_PORT`, `STREAM_ICE_PORT`, `STREAM_API_PORT`, and
`STREAM_RTP_INPUT_PORT` to unused ports and update the Robo-Boy endpoint/relay
configuration to match. Do not run two gateways on the same ports. RTP input is
loopback only. The API permits management only from loopback; keep its callers on
the robot relay. Remote browsers should use discovery through that relay.

For LAN/VPN access, `STREAM_ADDITIONAL_HOSTS` accepts comma-separated reachable
hostnames/IPs for ICE advertisement. Configure firewall/NAT/TURN for the actual
network; working HTTP signaling alone does not prove media connectivity. Direct
HTTP is intended for a trusted LAN; public/HTTPS deployments should use the
existing authenticated Robo-Boy proxy and suitable ICE connectivity. This profile
does not change the proxy or other applications' gateway configuration.

## Projection and timestamps

WebRTC here carries plain video. Select the matching ROS CameraInfo and optical TF
frame in Robo-Boy for 3D projection. It uses Latest TF, so alignment during motion
is approximate and is labelled as not time-synchronized. Optional capture-time
WebRTC metadata and synchronization are deferred. Existing ROS Image headers and
non-WebRTC streamer metadata handling are unchanged.

The encoder accepts rgb8/bgr8/rgba8/bgra8/mono8, bounded even image dimensions, and
padded ROS rows. It uses a bounded newest-frame input queue and low-latency H.264
with no B frames. Encoding adds CPU work; it does not alter existing MJPEG streams.
If nobody needs WebRTC, leave the profile off. Future consumers use standard WHEP
without depending on Robo-Boy messages.

## Isolated encoder check

```sh
docker build -t microduck-webrtc-encoder:check -f stream/Dockerfile .
docker run --rm --network none -e ROS_DOMAIN_ID=103 \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  -v "$PWD/stream:/opt/camera:ro" microduck-webrtc-encoder:check \
  python3 /opt/camera/tests/encoder_smoke.py
```

This exercises the real GStreamer encoder and receives standard RTP, including
padded rows, without joining the robot network. Browser WHEP negotiation, playback
and Camera/3D sharing are checked in Robo-Boy's optional gateway integration test.
