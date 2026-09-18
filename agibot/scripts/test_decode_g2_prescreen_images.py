import json
from pathlib import Path

from agibot.scripts.decode_g2_prescreen_images import decode_episode
from PIL import Image


def prepare(root: Path) -> Path:
    ep = root / "episode_000123"
    (ep / "images").mkdir(parents=True)
    images = {}
    for camera in ("head_color", "hand_right"):
        relative = f"images/{camera}_000000.jpg"
        images[camera] = relative
        Image.new("RGB", (640, 480), "gray").save(ep / relative, format="JPEG")
    (ep / "frames.jsonl").write_text(json.dumps({"frame_index": 0, "images": images}) + "\n")
    return ep


def test_all_selected_images_decode_and_source_unchanged(tmp_path):
    ep = prepare(tmp_path)
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in ep.rglob("*") if p.is_file()}
    row = decode_episode(str(tmp_path), ep.name, ("head_color", "hand_right"))
    assert row["status"] == "pass"
    assert row["decoded_images"] == 2
    assert before == {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in before}


def test_bad_image_reported(tmp_path):
    ep = prepare(tmp_path)
    (ep / "images/hand_right_000000.jpg").write_bytes(b"not a JPEG")
    row = decode_episode(str(tmp_path), ep.name, ("head_color", "hand_right"))
    assert row["status"] == "fail"
    assert row["errors"]


def test_bad_shape_reported(tmp_path):
    ep = prepare(tmp_path)
    Image.new("RGB", (32, 32)).save(ep / "images/head_color_000000.jpg")
    row = decode_episode(str(tmp_path), ep.name, ("head_color", "hand_right"))
    assert row["status"] == "fail"


def test_left_image_not_required_for_right_policy(tmp_path):
    ep = prepare(tmp_path)
    row = decode_episode(str(tmp_path), ep.name, ("head_color", "hand_right"))
    assert row["status"] == "pass"
