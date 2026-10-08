#!/usr/bin/env python3
"""Task-local read-only G2 observation bridge with optimized JPEG Huffman tables.

This process deliberately exposes only two operations: ``info`` and
``snapshot``.  It contains no robot command, controller-mode, motion-plan, or
end-effector command API.  The listener is restricted to loopback so access is
expected through an SSH local-forward tunnel.

Run on the G2 control computer after sourcing ``/home/agi/app/env.sh``.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import signal
import socket
import struct
import time
from typing import Any

import cv2
import numpy as np


SCHEMA = "g2_groot_right_observation_v1"
PROTOCOL_VERSION = 2
RIGHT_EE_FRAME = "arm_r_end_link"
SOURCE_POSE_FRAME = "base_link"
TRAINING_POSE_FRAME_LABEL = "base_link_tf"
GRIPPER_TRAINING_OPEN = -0.785
GRIPPER_TRAINING_CLOSED = 0.0
GRIPPER_G2_RAW_CLOSED = 120.0
GRIPPER_FEEDBACK_ENCODINGS = ("raw_0_120", "native_radians")
MAX_REQUEST_BYTES = 4096
DEFAULT_TIMEOUT_MS = 500.0

RIGHT_ARM_JOINT_NAMES = (
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--camera-timeout-ms", type=float, default=DEFAULT_TIMEOUT_MS)
    parser.add_argument("--gdk-warmup-s", type=float, default=2.0)
    parser.add_argument("--max-camera-skew-ms", type=float, default=100.0)
    parser.add_argument("--max-state-camera-skew-ms", type=float, default=50.0)
    parser.add_argument("--gripper-joint-name", default="right_gripper_joint1")
    parser.add_argument(
        "--gripper-feedback-encoding",
        choices=GRIPPER_FEEDBACK_ENCODINGS,
        default="raw_0_120",
    )
    parser.add_argument(
        "--model-input-jpeg-quality",
        type=int,
        default=0,
        help="0 forwards source JPEGs; 80..100 letterboxes to 640x480 then encodes once",
    )
    return parser.parse_args()


def _require_loopback(host: str) -> None:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None)}
    except socket.gaierror as error:
        raise ValueError(f"cannot resolve bind host {host!r}: {error}") from error
    if not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses):
        raise ValueError("the read-only bridge must bind to a loopback address")


def _normalize_quaternion_xyzw(values: list[float]) -> list[float]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("right EEF quaternion is invalid")
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-8:
        raise ValueError("right EEF quaternion has zero norm")
    return [float(value / norm) for value in values]


def _gripper_feedback_to_training(position: float, encoding: str) -> float:
    if not math.isfinite(position):
        raise ValueError("right gripper feedback is non-finite")
    if encoding == "raw_0_120":
        fraction = (
            min(max(float(position), 0.0), GRIPPER_G2_RAW_CLOSED)
            / GRIPPER_G2_RAW_CLOSED
        )
        return float(
            GRIPPER_TRAINING_OPEN
            + fraction * (GRIPPER_TRAINING_CLOSED - GRIPPER_TRAINING_OPEN)
        )
    if encoding == "native_radians":
        if not GRIPPER_TRAINING_OPEN - 0.02 <= position <= GRIPPER_TRAINING_CLOSED + 0.02:
            raise ValueError("native right gripper feedback is out of range")
        return float(
            min(GRIPPER_TRAINING_CLOSED, max(GRIPPER_TRAINING_OPEN, position))
        )
    raise ValueError(f"unknown right gripper feedback encoding: {encoding}")


def _enum_name(value: Any) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _image_record(image: Any, name: str) -> tuple[dict[str, Any], bytes]:
    payload = bytes(image.data)
    if not payload:
        raise RuntimeError(f"{name} image payload is empty")
    record = {
        "name": name,
        "width": int(image.width),
        "height": int(image.height),
        "timestamp_ns": int(image.timestamp_ns),
        "encoding": _enum_name(image.encoding),
        "color_format": _enum_name(image.color_format),
        "bit_depth": int(image.bit_depth),
        "payload_bytes": len(payload),
    }
    return record, payload


def _model_input_image_record(
    source_record: dict[str, Any], source_payload: bytes, jpeg_quality: int
) -> tuple[dict[str, Any], bytes]:
    if jpeg_quality == 0:
        return source_record, source_payload
    name = str(source_record["name"])
    bgr = cv2.imdecode(
        np.frombuffer(source_payload, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if bgr is None:
        raise RuntimeError(f"failed to decode {name} source JPEG")
    source_height, source_width = bgr.shape[:2]
    target_width, target_height = 640, 480
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    if (resized_width, resized_height) == (source_width, source_height):
        resized = bgr
    else:
        interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(
            bgr, (resized_width, resized_height), interpolation=interpolation
        )
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x0 = (target_width - resized_width) // 2
    y0 = (target_height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    # Optimized Huffman tables reduce transfer bytes without changing the
    # quantized JPEG image or any decoded model-input pixel.  Quality and
    # letterboxing are identical to the pinned 194 observation bridge.
    ok, encoded = cv2.imencode(
        ".jpg", canvas,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    )
    if not ok:
        raise RuntimeError(f"failed to encode {name} model-input JPEG")
    payload = encoded.tobytes()
    record = {
        **source_record,
        "width": target_width,
        "height": target_height,
        "encoding": "JPEG",
        "color_format": "BGR",
        "bit_depth": 8,
        "payload_bytes": len(payload),
        "server_preprocessing": {
            "source_width": source_width,
            "source_height": source_height,
            "resize_mode": "letterbox",
            "target_width": target_width,
            "target_height": target_height,
            "letterbox_scale": scale,
            "letterbox_width": resized_width,
            "letterbox_height": resized_height,
            "letterbox_x0": x0,
            "letterbox_y0": y0,
            "jpeg_quality": jpeg_quality,
            "jpeg_huffman_optimized": True,
        },
    }
    return record, payload


class G2RightObservationSource:
    def __init__(
        self,
        gdk: Any,
        timeout_ms: float,
        max_camera_skew_ms: float,
        max_state_camera_skew_ms: float,
        model_input_jpeg_quality: int,
        gripper_joint_name: str,
        gripper_feedback_encoding: str,
    ):
        self.gdk = gdk
        self.timeout_ms = float(timeout_ms)
        self.max_camera_skew_ms = float(max_camera_skew_ms)
        self.max_state_camera_skew_ms = float(max_state_camera_skew_ms)
        self.model_input_jpeg_quality = int(model_input_jpeg_quality)
        self.gripper_joint_name = str(gripper_joint_name)
        self.gripper_feedback_encoding = str(gripper_feedback_encoding)
        self.robot = gdk.Robot()
        self.camera = gdk.Camera()
        self.tf = gdk.TF()
        self.snapshot_count = 0

    def info(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "mode": "right_arm_observation_only",
            "operations": ["info", "snapshot"],
            "control_api_exposed": False,
            "live_execution_enabled": False,
            "motor_commands_sent": 0,
            "cameras": ["head_color", "hand_right"],
            "state": ["right_eef", "right_gripper"],
            "action": [],
            "source_pose_frame": SOURCE_POSE_FRAME,
            "training_pose_frame_label": TRAINING_POSE_FRAME_LABEL,
            "right_ee_frame": RIGHT_EE_FRAME,
            "quaternion_order": "xyzw",
            "gripper_joint_name": self.gripper_joint_name,
            "gripper_feedback_encoding": self.gripper_feedback_encoding,
            "gripper_training_convention": [-0.785, 0.0],
            "max_camera_skew_ms": self.max_camera_skew_ms,
            "max_state_camera_skew_ms": self.max_state_camera_skew_ms,
            "server_model_input_letterbox": self.model_input_jpeg_quality > 0,
            "model_input_jpeg_quality": self.model_input_jpeg_quality or None,
            "jpeg_huffman_optimized": self.model_input_jpeg_quality > 0,
        }

    def _right_gripper(self) -> dict[str, Any]:
        end_state = self.robot.get_end_state()
        right = end_state.get("right_end_state", {})
        states = right.get("end_states") or []
        names = right.get("names") or []
        if not states:
            raise RuntimeError("right_end_state.end_states is empty")
        if names != [self.gripper_joint_name] or len(states) != 1:
            raise RuntimeError(
                "unexpected right omnipicker feedback schema: "
                f"names={names!r}, expected={[self.gripper_joint_name]!r}"
            )
        item = states[0]
        error_code = int(item.get("err_code", item.get("error_code", -1)))
        if error_code != 0:
            raise RuntimeError(f"right gripper error_code={error_code}")
        raw = float(item["position"])
        return {
            "name": str(names[0]),
            "raw_position": raw,
            "training_position": _gripper_feedback_to_training(
                raw, self.gripper_feedback_encoding
            ),
            "velocity": float(item.get("velocity", 0.0)),
            "effort": float(item.get("effort", 0.0)),
            "error_code": error_code,
        }

    def _right_joint_health(self) -> dict[str, Any]:
        joint_states = self.robot.get_joint_states()
        by_name = {str(item.get("name", "")): item for item in joint_states.get("states", [])}
        missing = [name for name in RIGHT_ARM_JOINT_NAMES if name not in by_name]
        if missing:
            raise RuntimeError(f"missing right arm joints: {missing}")
        records = []
        for name in RIGHT_ARM_JOINT_NAMES:
            item = by_name[name]
            records.append(
                {
                    "name": name,
                    "position": float(item.get("position", 0.0)),
                    "velocity": float(item.get("velocity", 0.0)),
                    "error_code": int(item.get("error_code", -1)),
                }
            )
        errors = [item for item in records if item["error_code"] != 0]
        if errors:
            raise RuntimeError(f"right arm joint errors: {errors}")
        return {
            "timestamp": int(joint_states.get("timestamp", 0)),
            "joints": records,
        }

    def _right_body_health(self) -> dict[str, Any]:
        status = self.robot.get_whole_body_status()
        return {
            "timestamp": int(status.get("timestamp", 0)),
            "right_arm_error": int(status.get("right_arm_error", -1)),
            "right_arm_control": bool(status.get("right_arm_control", False)),
            "right_arm_estop": bool(status.get("right_arm_estop", False)),
            "right_end_error": int(status.get("right_end_error", -1)),
            "right_end_model": str(status.get("right_end_model", "unknown")),
        }

    def snapshot(self, *, enforce_sync: bool = True) -> tuple[dict[str, Any], list[bytes]]:
        started_wall_ns = time.time_ns()
        started_monotonic_ns = time.monotonic_ns()
        head = self.camera.get_latest_image(
            self.gdk.CameraType.kHeadColor, self.timeout_ms
        )
        right = self.camera.get_nearest_image(
            self.gdk.CameraType.kHandRightColor,
            int(head.timestamp_ns),
            self.timeout_ms,
        )
        head_record, head_payload = _image_record(head, "head_color")
        right_record, right_payload = _image_record(right, "hand_right")
        camera_skew_ms = abs(head_record["timestamp_ns"] - right_record["timestamp_ns"]) / 1e6
        if camera_skew_ms > self.max_camera_skew_ms:
            raise RuntimeError(
                f"camera timestamp skew {camera_skew_ms:.3f} ms exceeds "
                f"{self.max_camera_skew_ms:.3f} ms"
            )

        transform = self.tf.get_tf_from_base_link(RIGHT_EE_FRAME)
        tf_timestamp_ns = int(self.tf.get_latest_timestamp(RIGHT_EE_FRAME))
        quaternion = _normalize_quaternion_xyzw(
            [
                float(transform.rotation.x),
                float(transform.rotation.y),
                float(transform.rotation.z),
                float(transform.rotation.w),
            ]
        )
        pose = [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
            *quaternion,
        ]
        gripper = self._right_gripper()
        joint_health = self._right_joint_health()
        body_health = self._right_body_health()
        camera_to_tf_skew_ms = abs(tf_timestamp_ns - head_record["timestamp_ns"]) / 1e6
        camera_to_joint_skew_ms = (
            abs(int(joint_health["timestamp"]) - head_record["timestamp_ns"]) / 1e6
        )
        maximum_state_camera_skew_ms = max(
            camera_to_tf_skew_ms, camera_to_joint_skew_ms
        )
        if enforce_sync and maximum_state_camera_skew_ms > self.max_state_camera_skew_ms:
            raise RuntimeError(
                f"state-camera timestamp skew {maximum_state_camera_skew_ms:.3f} ms "
                f"exceeds {self.max_state_camera_skew_ms:.3f} ms"
            )
        # Validate camera/state synchronization before CPU-side resizing so
        # preprocessing time cannot make an otherwise synchronized sample look
        # stale.  Original timestamps remain attached to the derived images.
        head_record, head_payload = _model_input_image_record(
            head_record, head_payload, self.model_input_jpeg_quality
        )
        right_record, right_payload = _model_input_image_record(
            right_record, right_payload, self.model_input_jpeg_quality
        )
        self.snapshot_count += 1
        finished_monotonic_ns = time.monotonic_ns()

        header = self.info()
        header.update(
            {
                "snapshot_index": self.snapshot_count,
                "capture_started_wall_ns": started_wall_ns,
                "capture_started_monotonic_ns": started_monotonic_ns,
                "capture_finished_monotonic_ns": finished_monotonic_ns,
                "capture_duration_ms": (finished_monotonic_ns - started_monotonic_ns) / 1e6,
                "camera_skew_ms": camera_skew_ms,
                "camera_to_tf_skew_ms": camera_to_tf_skew_ms,
                "camera_to_joint_skew_ms": camera_to_joint_skew_ms,
                "maximum_state_camera_skew_ms": maximum_state_camera_skew_ms,
                "images": [head_record, right_record],
                "right_eef_xyz_quaternion_xyzw": pose,
                "right_eef_tf_timestamp_ns": tf_timestamp_ns,
                "right_gripper": gripper,
                "right_joint_health": joint_health,
                "right_body_health": body_health,
            }
        )
        return header, [head_payload, right_payload]


def _read_request(connection: socket.socket) -> dict[str, Any]:
    buffer = bytearray()
    while b"\n" not in buffer:
        chunk = connection.recv(1024)
        if not chunk:
            raise ConnectionError("client closed before sending a request")
        buffer.extend(chunk)
        if len(buffer) > MAX_REQUEST_BYTES:
            raise ValueError("request is too large")
    line = bytes(buffer).split(b"\n", 1)[0]
    request = json.loads(line.decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    return request


def _send_packet(connection: socket.socket, header: dict[str, Any], blobs: list[bytes]) -> None:
    payload = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # One write avoids delayed-ACK/Nagle stalls between the tiny length/header
    # records and the two already-compressed JPEG payloads.  The image bytes are
    # forwarded unchanged.
    connection.sendall(b"".join((struct.pack("!I", len(payload)), payload, *blobs)))


def _serve(source: G2RightObservationSource, host: str, port: int) -> None:
    stop = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(4)
        listener.settimeout(0.5)
        print(json.dumps({"event": "ready", **source.info(), "bind": f"{host}:{port}"}), flush=True)
        while not stop:
            try:
                connection, peer = listener.accept()
            except socket.timeout:
                continue
            with connection:
                # Keep a tunnel channel open for repeated snapshots.  Opening a
                # fresh SSH channel for every JPEG pair adds hundreds of
                # milliseconds on this robot network.  One-shot clients remain
                # compatible: closing the socket simply exits this loop.
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                connection.settimeout(10.0)
                while not stop:
                    try:
                        request = _read_request(connection)
                    except (ConnectionError, socket.timeout):
                        break
                    try:
                        operation = request.get("op")
                        if operation == "info":
                            _send_packet(connection, source.info(), [])
                        elif operation == "snapshot":
                            header, blobs = source.snapshot()
                            _send_packet(connection, header, blobs)
                        else:
                            raise ValueError("unsupported operation; allowed: info, snapshot")
                    except Exception as error:
                        _send_packet(
                            connection,
                            {
                                "schema": SCHEMA,
                                "status": "ERROR",
                                "error": f"{type(error).__name__}: {error}",
                                "control_api_exposed": False,
                                "live_execution_enabled": False,
                                "motor_commands_sent": 0,
                                "images": [],
                                "peer": str(peer[0]),
                            },
                            [],
                        )


def main() -> None:
    args = parse_args()
    _require_loopback(args.bind_host)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in 1..65535")
    if (
        args.camera_timeout_ms <= 0
        or args.max_camera_skew_ms <= 0
        or args.max_state_camera_skew_ms <= 0
    ):
        raise ValueError("camera timeouts and skew limit must be positive")
    if args.model_input_jpeg_quality != 0 and not (
        80 <= args.model_input_jpeg_quality <= 100
    ):
        raise ValueError("--model-input-jpeg-quality must be 0 or in [80, 100]")

    import agibot_gdk  # Imported only on the G2 runtime after env.sh is sourced.

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("agibot_gdk.gdk_init failed")
    try:
        source = G2RightObservationSource(
            agibot_gdk,
            timeout_ms=args.camera_timeout_ms,
            max_camera_skew_ms=args.max_camera_skew_ms,
            max_state_camera_skew_ms=args.max_state_camera_skew_ms,
            model_input_jpeg_quality=args.model_input_jpeg_quality,
            gripper_joint_name=args.gripper_joint_name,
            gripper_feedback_encoding=args.gripper_feedback_encoding,
        )
        time.sleep(args.gdk_warmup_s)
        source.snapshot(enforce_sync=False)
        source.snapshot_count = 0
        _serve(source, args.bind_host, args.port)
    finally:
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    main()
