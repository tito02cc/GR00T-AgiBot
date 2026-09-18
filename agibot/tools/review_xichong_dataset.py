#!/usr/bin/env python3
"""Serve a local two-camera human review UI for the formal xichong dataset.

The tool is intentionally dependency-free beyond Python's standard library. It
never modifies the dataset: review decisions are atomically stored in a separate
JSON file and can be exported as CSV from the browser.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import re
import tempfile
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


AGIBOT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = AGIBOT_ROOT / "gr00t_data/xichong_right_single_grasp_300"
DEFAULT_REVIEW_FILE = AGIBOT_ROOT / "reviews/xichong_right_single_grasp_300_reviews.json"
UI_FILE = Path(__file__).with_name("review_xichong_dataset.html")
CAMERAS = {"head_color", "hand_right"}
STATUSES = {"pass", "fail", "uncertain"}
ISSUES = {
    "empty_grasp",
    "unstable_grasp",
    "insufficient_lift",
    "object_dropped",
    "occlusion_or_image_quality",
    "timing_anomaly",
    "trajectory_anomaly",
    "other",
}
VIDEO_PATTERN = re.compile(r"episode_(\d{6})\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


class ReviewStore:
    def __init__(self, path: Path, dataset_name: str, total_episodes: int):
        self.path = path
        self.dataset_name = dataset_name
        self.total_episodes = total_episodes
        self.lock = threading.Lock()
        if path.exists():
            self.data = read_json(path)
            if int(self.data.get("total_episodes", -1)) != total_episodes:
                raise ValueError(
                    f"review file expects {self.data.get('total_episodes')} episodes, "
                    f"dataset has {total_episodes}"
                )
        else:
            created = now_iso()
            self.data = {
                "schema_version": 1,
                "dataset_name": dataset_name,
                "total_episodes": total_episodes,
                "created_at": created,
                "updated_at": created,
                "reviews": {},
            }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.data))

    def set_review(self, episode_index: int, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status", ""))
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        issues = payload.get("issues", [])
        if not isinstance(issues, list) or any(issue not in ISSUES for issue in issues):
            raise ValueError("unknown issue code")
        note = str(payload.get("note", "")).strip()
        reviewer = str(payload.get("reviewer", "")).strip()
        if len(note) > 2000 or len(reviewer) > 100:
            raise ValueError("note or reviewer is too long")
        if status == "pass":
            issues = []
        review = {
            "episode_index": episode_index,
            "status": status,
            "issues": sorted(set(issues)),
            "note": note,
            "reviewer": reviewer,
            "reviewed_at": now_iso(),
        }
        with self.lock:
            self.data["reviews"][str(episode_index)] = review
            self.data["updated_at"] = now_iso()
            self._write_locked()
        return review

    def delete_review(self, episode_index: int) -> bool:
        with self.lock:
            removed = self.data["reviews"].pop(str(episode_index), None) is not None
            if removed:
                self.data["updated_at"] = now_iso()
                self._write_locked()
            return removed

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self.data, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise


class ReviewApplication:
    def __init__(self, dataset: Path, review_file: Path):
        self.dataset = dataset.resolve()
        required = [
            self.dataset / "meta/info.json",
            self.dataset / "meta/episodes.jsonl",
            self.dataset / "meta/source_episode_map.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing dataset metadata: {missing}")
        self.info = read_json(required[0])
        self.episode_meta = read_jsonl(required[1])
        self.source_map = read_json(required[2])["episodes"]
        if len(self.episode_meta) != len(self.source_map):
            raise ValueError("episodes.jsonl and source_episode_map.json disagree")
        self.total_episodes = len(self.episode_meta)
        self.store = ReviewStore(review_file.resolve(), self.dataset.name, self.total_episodes)

    def episodes_payload(self) -> dict[str, Any]:
        reviews = self.store.snapshot()["reviews"]
        episodes = []
        for index, (meta, mapping) in enumerate(zip(self.episode_meta, self.source_map)):
            episodes.append(
                {
                    "episode_index": index,
                    "source_episode": mapping["source_episode"],
                    "full_dataset_episode_index": mapping["full_dataset_episode_index"],
                    "length": int(meta["length"]),
                    "duration_s": round(int(meta["length"]) / float(self.info["fps"]), 3),
                    "tasks": meta["tasks"],
                    "review": reviews.get(str(index)),
                }
            )
        counts = {"pending": self.total_episodes, "pass": 0, "fail": 0, "uncertain": 0}
        for review in reviews.values():
            status = review.get("status")
            if status in STATUSES:
                counts[status] += 1
                counts["pending"] -= 1
        return {
            "dataset_name": self.dataset.name,
            "fps": self.info["fps"],
            "total_episodes": self.total_episodes,
            "counts": counts,
            "episodes": episodes,
        }

    def video_path(self, camera: str, filename: str) -> Path:
        if camera not in CAMERAS:
            raise ValueError("unknown camera")
        match = VIDEO_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError("invalid video filename")
        episode_index = int(match.group(1))
        if not 0 <= episode_index < self.total_episodes:
            raise ValueError("episode index out of range")
        chunk = episode_index // int(self.info["chunks_size"])
        path = (
            self.dataset
            / f"videos/chunk-{chunk:03d}"
            / f"observation.images.{camera}"
            / filename
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def export_csv(self) -> bytes:
        payload = self.episodes_payload()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "episode_index",
                "source_episode",
                "full_dataset_episode_index",
                "length",
                "duration_s",
                "status",
                "issues",
                "note",
                "reviewer",
                "reviewed_at",
            ]
        )
        for episode in payload["episodes"]:
            review = episode["review"] or {}
            writer.writerow(
                [
                    episode["episode_index"],
                    episode["source_episode"],
                    episode["full_dataset_episode_index"],
                    episode["length"],
                    episode["duration_s"],
                    review.get("status", "pending"),
                    "|".join(review.get("issues", [])),
                    review.get("note", ""),
                    review.get("reviewer", ""),
                    review.get("reviewed_at", ""),
                ]
            )
        return output.getvalue().encode("utf-8-sig")


def make_handler(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "AgibotReview/1.0"

        def send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: int = 200,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: Any, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def send_error_json(self, status: int, message: str) -> None:
            self.send_json({"error": message}, status)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                if path == "/":
                    self.send_bytes(UI_FILE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/episodes":
                    self.send_json(application.episodes_payload())
                elif path == "/api/health":
                    self.send_json({"status": "ok", "episodes": application.total_episodes})
                elif path == "/api/export.csv":
                    self.send_bytes(
                        application.export_csv(),
                        "text/csv; charset=utf-8",
                        extra_headers={
                            "Content-Disposition": (
                                'attachment; filename="xichong_right_single_grasp_300_reviews.csv"'
                            )
                        },
                    )
                elif path.startswith("/media/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 3:
                        raise ValueError("invalid media URL")
                    self.send_video(application.video_path(parts[1], parts[2]))
                else:
                    self.send_error_json(404, "not found")
            except (ValueError, FileNotFoundError) as exc:
                self.send_error_json(404, str(exc))
            except Exception as exc:  # noqa: BLE001 - keep the review server responsive
                self.send_error_json(500, f"{type(exc).__name__}: {exc}")

        def do_HEAD(self) -> None:  # noqa: N802 - browsers may probe video metadata with HEAD
            path = unquote(urlparse(self.path).path)
            try:
                if path.startswith("/media/"):
                    parts = path.strip("/").split("/")
                    if len(parts) != 3:
                        raise ValueError("invalid media URL")
                    self.send_video(
                        application.video_path(parts[1], parts[2]),
                        head_only=True,
                    )
                else:
                    self.send_error_json(404, "not found")
            except (ValueError, FileNotFoundError) as exc:
                self.send_error_json(404, str(exc))

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            match = re.fullmatch(r"/api/review/(\d+)", parsed.path)
            if match is None:
                self.send_error_json(404, "not found")
                return
            try:
                episode_index = int(match.group(1))
                if not 0 <= episode_index < application.total_episodes:
                    raise ValueError("episode index out of range")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16_384:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("JSON body must be an object")
                review = application.store.set_review(episode_index, payload)
                self.send_json({"review": review})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error_json(400, str(exc))
            except Exception as exc:  # noqa: BLE001 - report storage failures to UI
                self.send_error_json(500, f"{type(exc).__name__}: {exc}")

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler API
            match = re.fullmatch(r"/api/review/(\d+)", urlparse(self.path).path)
            if match is None:
                self.send_error_json(404, "not found")
                return
            episode_index = int(match.group(1))
            if not 0 <= episode_index < application.total_episodes:
                self.send_error_json(400, "episode index out of range")
                return
            self.send_json({"removed": application.store.delete_review(episode_index)})

        def send_video(self, path: Path, head_only: bool = False) -> None:
            file_size = path.stat().st_size
            start = 0
            end = file_size - 1
            status = 200
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if match is None or (not match.group(1) and not match.group(2)):
                    self.send_error_json(416, "unsupported byte range")
                    return
                if match.group(1):
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else end
                else:
                    suffix_length = int(match.group(2))
                    start = max(0, file_size - suffix_length)
                if start >= file_size or start > end:
                    self.send_error_json(416, "byte range outside file")
                    return
                end = min(end, file_size - 1)
                status = 206
            content_length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "private, max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()
            if head_only:
                return
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = content_length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def log_message(self, fmt: str, *args: Any) -> None:
            if args and str(args[1]) >= "400":
                super().log_message(fmt, *args)

    return ReviewHandler


def main() -> int:
    args = parse_args()
    application = ReviewApplication(args.dataset, args.review_file)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(application))
    url = f"http://{args.host}:{server.server_port}/"
    print(f"Dataset: {application.dataset}")
    print(f"Review file: {application.store.path}")
    print(f"Review UI: {url}")
    print("Press Ctrl-C to stop. Review decisions are saved immediately.")
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
