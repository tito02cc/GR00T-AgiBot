"""Task-local JPEG transport must preserve the policy's decoded pixels."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
TASK = Path(__file__).resolve().parent
PINNED = ROOT / 'agibot/deployments/g2_194/zhewan_right_place_r0003/robot/g2_groot_right_observation_bridge.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_huffman_optimization_preserves_all_decoded_pixels():
    original = load('pinned_g2_observation', PINNED)
    optimized = load('task_g2_observation', TASK / 'observation_bridge.py')
    yy, xx = np.mgrid[:480, :640]
    bgr = np.stack(((xx + yy) % 256, (2 * xx + yy) % 256,
                    (xx + 3 * yy) % 256), axis=2).astype(np.uint8)
    ok, source = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    record = {'name': 'head_color', 'width': 640, 'height': 480,
              'encoding': 'JPEG', 'color_format': 'BGR', 'bit_depth': 8,
              'payload_bytes': len(source)}
    base_meta, base_bytes = original._model_input_image_record(record, source.tobytes(), 92)
    task_meta, task_bytes = optimized._model_input_image_record(record, source.tobytes(), 92)
    assert task_meta['server_preprocessing'].pop('jpeg_huffman_optimized') is True
    assert base_meta['server_preprocessing'] == task_meta['server_preprocessing']
    assert len(task_bytes) < len(base_bytes)
    np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(base_bytes, np.uint8), 1),
                                  cv2.imdecode(np.frombuffer(task_bytes, np.uint8), 1))
    assert optimized._model_input_image_record(record, source.tobytes(), 0) == (record, source.tobytes())
