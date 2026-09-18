"""G2 right-arm placement template: H16, 10 Hz, native-radian gripper."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


RIGHT_PLACE_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["head_color", "hand_right"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["right_eef", "right_gripper"]),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["right_eef", "right_gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="right_eef",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
                state_key="right_gripper",
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(RIGHT_PLACE_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
