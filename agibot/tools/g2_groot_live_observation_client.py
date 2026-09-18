#!/usr/bin/env python3
"""Client and image preprocessing for the read-only G2 GR00T bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import socket
import struct
from typing import Any

import cv2
import numpy as np


SCHEMA = "g2_groot_right_observation_v1"
MAX_HEADER_BYTES = 1_000_000
MAX_IMAGE_BYTES = 20_000_000


@dataclass(frozen=True)
class G2LiveSnapshot:
    metadata: dict[str, Any]
    head_color_rgb: np.ndarray
    hand_right_rgb: np.ndarray
    source_payload_sha256: dict[str, str]
    preprocessing: dict[str, dict[str, Any]]


def _recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise ConnectionError(f"connection closed after {len(chunks)}/{count} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def _request_on_connection(
    connection: socket.socket, operation: str
) -> tuple[dict[str, Any], list[bytes]]:
    request = json.dumps({"op": operation}, separators=(",", ":")).encode("utf-8") + b"\n"
    connection.sendall(request)
    header_size = struct.unpack("!I", _recv_exact(connection, 4))[0]
    if not 1 <= header_size <= MAX_HEADER_BYTES:
        raise ValueError(f"invalid bridge header size {header_size}")
    metadata = json.loads(_recv_exact(connection, header_size).decode("utf-8"))
    if metadata.get("status") == "ERROR":
        raise RuntimeError(str(metadata.get("error", "unknown bridge error")))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unexpected bridge schema {metadata.get('schema')!r}")
    if metadata.get("control_api_exposed") is not False:
        raise ValueError("bridge unexpectedly exposes a control API")
    if metadata.get("live_execution_enabled") is not False:
        raise ValueError("bridge unexpectedly enables live execution")
    if int(metadata.get("motor_commands_sent", -1)) != 0:
        raise ValueError("bridge reports a nonzero motor command count")

    blobs = []
    for record in metadata.get("images", []):
        size = int(record.get("payload_bytes", -1))
        if not 1 <= size <= MAX_IMAGE_BYTES:
            raise ValueError(f"invalid image payload size {size}")
        blobs.append(_recv_exact(connection, size))
    return metadata, blobs


def _request(
    host: str, port: int, operation: str, timeout_s: float
) -> tuple[dict[str, Any], list[bytes]]:
    with socket.create_connection((host, port), timeout=timeout_s) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(timeout_s)
        return _request_on_connection(connection, operation)


def get_bridge_info(host: str, port: int, timeout_s: float = 5.0) -> dict[str, Any]:
    metadata, blobs = _request(host, port, "info", timeout_s)
    if blobs:
        raise ValueError("info response unexpectedly contains image data")
    return metadata


def letterbox_rgb(
    rgb: np.ndarray, target_width: int = 640, target_height: int = 480
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match the G2 collection pipeline's 640x480 letterbox implementation."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected uint8 RGB image, got {rgb.dtype} {rgb.shape}")
    source_height, source_width = rgb.shape[:2]
    metadata: dict[str, Any] = {
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "resize_mode": "letterbox",
        "sharpen_strength": 0.0,
    }
    if (source_height, source_width) == (target_height, target_width):
        metadata["resized"] = False
        return np.ascontiguousarray(rgb), metadata

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale <= 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    x0 = (target_width - resized_width) // 2
    y0 = (target_height - resized_height) // 2
    canvas[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    metadata.update(
        {
            "resized": True,
            "letterbox_scale": scale,
            "letterbox_width": resized_width,
            "letterbox_height": resized_height,
            "letterbox_x0": x0,
            "letterbox_y0": y0,
        }
    )
    return np.ascontiguousarray(canvas), metadata


def _decode_image(record: dict[str, Any], payload: bytes) -> np.ndarray:
    encoding = str(record.get("encoding", ""))
    if encoding in {"JPEG", "PNG"}:
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode {record.get('name')} {encoding}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if encoding == "UNCOMPRESSED":
        height = int(record["height"])
        width = int(record["width"])
        array = np.frombuffer(payload, dtype=np.uint8)
        if array.size != height * width * 3:
            raise ValueError("unsupported uncompressed image shape")
        image = array.reshape(height, width, 3).copy()
        color_format = str(record.get("color_format", ""))
        if color_format == "RGB":
            return image
        if color_format == "BGR":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    raise ValueError(f"unsupported image encoding/color: {encoding}/{record.get('color_format')}")


def _snapshot_from_packet(
    metadata: dict[str, Any], blobs: list[bytes], *, compute_payload_hashes: bool = True
) -> G2LiveSnapshot:
    records = metadata.get("images", [])
    if [record.get("name") for record in records] != ["head_color", "hand_right"]:
        raise ValueError("bridge did not return head_color and hand_right in canonical order")
    if len(blobs) != 2:
        raise ValueError(f"expected two image payloads, got {len(blobs)}")

    decoded = {}
    preprocessing = {}
    hashes = {}
    for record, blob in zip(records, blobs, strict=True):
        name = str(record["name"])
        if compute_payload_hashes:
            hashes[name] = hashlib.sha256(blob).hexdigest()
        rgb = _decode_image(record, blob)
        decoded[name], preprocessing[name] = letterbox_rgb(rgb)
    return G2LiveSnapshot(
        metadata=metadata,
        head_color_rgb=decoded["head_color"],
        hand_right_rgb=decoded["hand_right"],
        source_payload_sha256=hashes,
        preprocessing=preprocessing,
    )


def get_live_snapshot(host: str, port: int, timeout_s: float = 10.0) -> G2LiveSnapshot:
    metadata, blobs = _request(host, port, "snapshot", timeout_s)
    return _snapshot_from_packet(metadata, blobs)


class G2LiveObservationClient:
    """Persistent read-only client, primarily to avoid SSH channel setup jitter."""

    def __init__(
        self, host: str, port: int, timeout_s: float = 10.0, *, compute_payload_hashes: bool = True
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.compute_payload_hashes = compute_payload_hashes
        self._connection: socket.socket | None = None

    def __enter__(self) -> "G2LiveObservationClient":
        self._connect()
        return self

    def _connect(self) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._connection.settimeout(self.timeout_s)

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _request(self, operation: str) -> tuple[dict[str, Any], list[bytes]]:
        if self._connection is None:
            raise RuntimeError("client must be used as a context manager")
        try:
            return _request_on_connection(self._connection, operation)
        except (BrokenPipeError, ConnectionError, OSError):
            # The read-only robot bridge deliberately closes a persistent
            # channel after 10 seconds with no request.  A slowed physical
            # waypoint can exceed that idle interval.  Reconnect once and
            # request a new observation; never cache or replay sensor data.
            self._connect()
            assert self._connection is not None
            return _request_on_connection(self._connection, operation)

    def get_info(self) -> dict[str, Any]:
        metadata, blobs = self._request("info")
        if blobs:
            raise ValueError("info response unexpectedly contains image data")
        return metadata

    def get_snapshot(self) -> G2LiveSnapshot:
        return _snapshot_from_packet(
            *self._request("snapshot"), compute_payload_hashes=self.compute_payload_hashes
        )
