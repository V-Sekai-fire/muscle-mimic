"""The 6-joint pelvis of MS-Human-700 and the converted free joint must put every body in the same place.

Needs the MS-Human-700 folder from MuJoCo Menagerie: set MSH700_XML=/path/to/ms_human_700/MS-Human-700.xml.
"""

import os

import mujoco
import numpy as np
import pytest

import importlib.util
import pathlib

# load by path: the musclemimic package __init__ pulls in JAX, and this module only needs numpy
_p = pathlib.Path(__file__).parents[1] / "musclemimic/environments/humanoids/ms_human_700_root.py"
_s = importlib.util.spec_from_file_location("ms_human_700_root", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
PELVIS_JOINTS, root_to_freejoint = _m.PELVIS_JOINTS, _m.root_to_freejoint

XML = os.environ.get("MSH700_XML")
if not XML:
    # A skipped gate is a false pass: refuse to collect instead of skipping.
    raise RuntimeError("MSH700_XML must point at MS-Human-700.xml; this test does not skip")


def _free_model():
    spec = mujoco.MjSpec.from_file(XML)
    pelvis = spec.body("pelvis")
    for n in PELVIS_JOINTS:
        spec.delete(spec.joint(n))
    pelvis.add_freejoint()
    return spec.compile(), np.array(pelvis.pos), np.array(pelvis.quat)


def test_free_joint_matches_hinges():
    m6 = mujoco.MjModel.from_xml_path(XML)
    mf, pos, quat = _free_model()
    d6, df = mujoco.MjData(m6), mujoco.MjData(mf)
    adr = [m6.joint(n).qposadr[0] for n in PELVIS_JOINTS]
    rng = np.random.default_rng(0)
    for _ in range(50):
        q6 = np.concatenate([rng.uniform(-1, 1, 3), rng.uniform(-0.8, 0.8, 3)])
        d6.qpos[:] = 0
        d6.qpos[adr] = q6
        df.qpos[:] = 0
        df.qpos[:7] = root_to_freejoint(q6, pos, quat)
        mujoco.mj_kinematics(m6, d6)
        mujoco.mj_kinematics(mf, df)
        for b in ("pelvis", "femur_r", "humerus_l", "head_neck"):
            if mujoco.mj_name2id(m6, mujoco.mjtObj.mjOBJ_BODY, b) < 0:
                continue
            np.testing.assert_allclose(d6.body(b).xpos, df.body(b).xpos, atol=1e-9, err_msg=b)
            np.testing.assert_allclose(d6.body(b).xmat, df.body(b).xmat, atol=1e-9, err_msg=b)


def test_prepare_spec_builds_free_root_and_sites():
    sites = {"pelvis": "pelvis_mimic", "head_neck": "head_mimic", "toes_l": "left_toes_mimic"}
    spec = mujoco.MjSpec.from_file(XML)
    pos, quat = _m.prepare_spec(spec, sites)
    m = spec.compile()
    assert m.joint("root").type == mujoco.mjtJoint.mjJNT_FREE
    assert m.nq == 86 and m.nv == 85  # 85 - 6 pelvis joints + 7 free qpos
    for s in sites.values():
        assert m.site(s).id >= 0
    assert np.allclose(pos, [0, 0, 0.95]) and np.allclose(quat, [0.707107, 0.707107, 0, 0], atol=1e-5)
    ctrl = m.actuator_ctrlrange[m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE]
    assert ctrl.size and np.all(ctrl == [-1.0, 1.0])
    mujoco.mj_step(m, mujoco.MjData(m))  # compiles and steps with the floor in place
