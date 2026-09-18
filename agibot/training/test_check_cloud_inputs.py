import json
import struct

from agibot.training.check_cloud_inputs import check_inventory, check_model, inventory
import pytest


def test_inventory_rejects_missing_extra_and_changed(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    item = root / "sample"
    item.write_bytes(b"123")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(inventory(root)))
    assert check_inventory(root, manifest)["bytes"] == 3
    item.write_bytes(b"1")
    with pytest.raises(ValueError, match="size_mismatch"):
        check_inventory(root, manifest)
    item.unlink()
    with pytest.raises(ValueError, match="missing"):
        check_inventory(root, manifest)
    item.write_bytes(b"123")
    (root / "unexpected").write_bytes(b"4")
    with pytest.raises(ValueError, match="extra"):
        check_inventory(root, manifest)


@pytest.mark.parametrize("mutation", ["none", "truncated", "wrong_index"])
def test_model_header_and_index(tmp_path, mutation):
    (tmp_path / "config.json").write_text('{"model_type":"test"}')
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    payload = b"1234" if mutation != "truncated" else b"123"
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    key = "bad" if mutation == "wrong_index" else "weight"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors"}})
    )
    if mutation == "none":
        assert check_model(tmp_path)["header_and_size"] == "PASS"
    else:
        with pytest.raises(ValueError):
            check_model(tmp_path)
