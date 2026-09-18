"""No GDK initialization, network access, model loading or physical control."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pinned_commands", HERE / "commands.py")
commands = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(commands)


@pytest.mark.parametrize("task", commands.TASKS)
def test_robot_import_closure_and_compilation(task):
    directory = HERE / task / "robot"
    sources = json.loads((HERE / task / "sources.json").read_text())["sources"]
    assert set(sources) == {p.stem for p in directory.glob("*.py")}
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            for name in modules:
                if name and name.startswith("g2_"):
                    assert (directory / f"{name}.py").is_file(), (path, name)


@pytest.mark.parametrize("task", commands.TASKS)
def test_launcher_and_profile(task, tmp_path):
    profile = json.loads((HERE / task / "profile.json").read_text())
    launcher = HERE / task / "robot_bridge.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text()
    for key in ("workspace_min", "workspace_max"):
        assert "--" + key.replace("_", "-") + " " + " ".join(map(str, profile[key])) in text
    assert '"${1:-standby}"' in text
    assert "set --\nset +u\nsource" in text
    generated = commands.build_commands(task, HERE.parents[2], tmp_path / "report.json", Path("/robot"))
    live = generated["live_after_onsite_confirmation"]
    assert str(HERE / "workstation_runtime/agibot/scripts/run_g2_groot_full_place_inference.py") in live
    assert live[live.index("--prompt") + 1] == profile["prompt"]
    assert live[live.index("--max-cycles") + 1] == "0"
    for key in ("optimize_transport", "native_chunk_submission", "require_native_collision_latch",
                "freeze_compensation_after_calibration"):
        assert ("--" + key.replace("_", "-") in live) == profile[key]
    assert "rtc" not in " ".join(live).lower()
    assert ("--freeze-compensation-after-calibration" in text) == profile["freeze_compensation_after_calibration"]


def test_command_printer_never_executes(tmp_path):
    result = subprocess.run([sys.executable, str(HERE / "commands.py"), commands.TASKS[0],
                             "--report", str(tmp_path / "report.json")],
                            capture_output=True, text=True, check=True)
    assert "PRINT ONLY" in result.stdout
    assert not (tmp_path / "report.json").exists()
    existing = tmp_path / "existing.json"
    existing.touch()
    result = subprocess.run([sys.executable, str(HERE / "commands.py"), commands.TASKS[1],
                             "--report", str(existing)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stderr


def test_workstation_agibot_import_closure():
    runtime = HERE / "workstation_runtime"
    for path in runtime.rglob("*.py"):
        tree = ast.parse(path.read_text())
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("agibot."):
                assert (runtime / (node.module.replace(".", "/") + ".py")).is_file()
