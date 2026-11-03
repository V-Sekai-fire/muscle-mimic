"""MS-Human-700 (MuJoCo Menagerie) as a MuscleMimic full-body environment.

The class name keeps the substring "MyoFullBody" because `musclemimic/core/utils/site_mapping.py` picks its mimic-site mapper from the
class name. The model comes from the MSH700_XML environment variable or the `spec=` argument; see `ms_human_700_root.py` for the
pelvis (3 slides + 3 hinges) to free-joint conversion that trajectories need.
"""

import os

from mujoco import MjSpec

from musclemimic.environments.humanoids import ms_human_700_root as root
from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody

# SOMA/Kimodo clip sites on MS-Human-700 bodies, the table of retarget_msh700.py (interactor-garment-proxy), same site names as MyoFullBody
BODY2SITES_FOR_MIMIC = {
    "pelvis": "pelvis_mimic",
    "lumbar1": "upper_body_mimic",
    "head_neck": "head_mimic",
    "humerus_l": "left_shoulder_mimic",
    "ulna_l": "left_elbow_mimic",
    "hand_l": "left_hand_mimic",
    "humerus_r": "right_shoulder_mimic",
    "ulna_r": "right_elbow_mimic",
    "hand_r": "right_hand_mimic",
    "femur_l": "left_hip_mimic",
    "tibia_l": "left_knee_mimic",
    "talus_l": "left_ankle_mimic",
    "toes_l": "left_toes_mimic",
    "femur_r": "right_hip_mimic",
    "tibia_r": "right_knee_mimic",
    "talus_r": "right_ankle_mimic",
    "toes_r": "right_toes_mimic",
}
HEALTHY_ROOT_HEIGHT = (0.5, 1.5)  # [m] pelvis height; the model stands with the pelvis near 0.95 m


class MsHuman700MyoFullBody(MyoFullBody):
    """MS-Human-700 with a free root and the MyoFullBody observation, action and mimic-site conventions."""

    def __init__(self, enable_touch_sensor_observations: bool = False, **kwargs) -> None:
        # the model has no touch sensors
        super().__init__(enable_touch_sensor_observations=enable_touch_sensor_observations, **kwargs)

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        self.pelvis_body_pos, self.pelvis_body_quat = root.prepare_spec(spec, self.body2sites_for_mimic, self.root_free_joint_xml_name)
        return spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        path = os.environ.get("MSH700_XML")
        if not path:
            raise FileNotFoundError("set MSH700_XML to MS-Human-700.xml (MuJoCo Menagerie ms_human_700) or pass spec=")
        return path

    @property
    def upper_body_xml_name(self) -> str:
        return "lumbar1"

    @property
    def root_height_healthy_range(self) -> tuple[float, float]:
        return HEALTHY_ROOT_HEIGHT

    @property
    def body2sites_for_mimic(self) -> dict[str, str]:
        return BODY2SITES_FOR_MIMIC


class MjxMsHuman700MyoFullBody(MjxMyoFullBody, MsHuman700MyoFullBody):
    """MJX variant."""
