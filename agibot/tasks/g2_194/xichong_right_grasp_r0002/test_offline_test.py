import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('grasp_offline', Path(__file__).with_name('offline_test.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_grasp_not_place_event():
    assert module.first_held([-.785, -.65, -.55, -.01], -.55) == 2
    assert module.first_held([-.785, -.65], -.55) is None


def test_reopening_has_hysteresis():
    assert module.reopen_indices([-.785, -.5, -.6, -.56, -.7, -.5, -.78], -.55, -.7) == [4, 6]


def test_lift_is_relative_to_closure_not_absolute_height():
    pose = np.zeros((4, 9)); pose[:, 2] = [.8, .7, .9, .9]
    assert np.isclose(module.lift_from_close(pose, 1, 2), 200)
    assert module.lift_from_close(pose, None, 2) is None
