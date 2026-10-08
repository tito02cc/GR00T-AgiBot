import json

from agibot.tools.configure_model_backbone import configure


def test_configure_bundle_changes_only_active_model_paths(tmp_path):
    model = tmp_path / "model"
    backbone = tmp_path / "backbone" / "Cosmos-Reason2-2B"
    model.mkdir()
    backbone.mkdir(parents=True)
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (backbone / name).write_bytes(b"present")
    weights = model / "model-00001-of-00003.safetensors"
    weights.write_bytes(b"unchanged")
    (model / "config.json").write_text(json.dumps({"model_name": "/old/path", "horizon": 16}))
    (model / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"model_name": "/old/path", "use_percentiles": False}})
    )

    configure(tmp_path, check_only=True)
    assert json.loads((model / "config.json").read_text())["model_name"] == "/old/path"

    configure(tmp_path)
    configure(tmp_path)
    config = json.loads((model / "config.json").read_text())
    processor = json.loads((model / "processor_config.json").read_text())
    assert config == {"model_name": str(backbone), "horizon": 16}
    assert processor == {"processor_kwargs": {"model_name": str(backbone), "use_percentiles": False}}
    assert weights.read_bytes() == b"unchanged"
