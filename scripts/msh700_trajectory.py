"""Turn a retargeted MS-Human-700 qpos clip (<clip>.qpos.npy, 6 pelvis slides/hinges) into a free-root trajectory at 100 Hz.

  uv run --with mujoco --with numpy python scripts/msh700_trajectory.py MS-Human-700.xml walk.qpos.npy out.npz [--fps=30]

Writes qpos (T, nq_free), qvel (T, nv_free), joint_names, jnt_type, frequency, split_points. Gate: at the clip's own frames the
free-joint model puts pelvis, femur_r, humerus_l and head where the 6-joint model did (max error < 1e-8 m). Quaternions are wxyz.
Resampling: linear for slides and hinges, normalised lerp (sign-aligned) for the root quaternion; qvel from mj_differentiatePos.
"""

import importlib.util
import pathlib
import sys

import mujoco
import numpy as np

_p = pathlib.Path(__file__).parents[1] / "musclemimic/environments/humanoids/ms_human_700_root.py"
_s = importlib.util.spec_from_file_location("ms_human_700_root", _p)
root = importlib.util.module_from_spec(_s)
_s.loader.exec_module(root)

CONTROL_HZ = 100.0
CHECK_BODIES = ("pelvis", "femur_r", "humerus_l", "head_neck")


def free_model(xml):
    spec = mujoco.MjSpec.from_file(xml)
    pelvis = spec.body("pelvis")
    pos, quat = np.array(pelvis.pos), np.array(pelvis.quat)
    for n in root.PELVIS_JOINTS:
        spec.delete(spec.joint(n))
    pelvis.add_freejoint().name = "root"
    return spec.compile(), pos, quat


def convert(m6, mf, pos, quat, q6):
    """q6 (F, nq6) -> free-model qpos (F, nqf), same frames."""
    qf = np.zeros((q6.shape[0], mf.nq))
    qf[:, :7] = root.root_to_freejoint(q6[:, [m6.joint(n).qposadr[0] for n in root.PELVIS_JOINTS]], pos, quat)
    for j in range(m6.njnt):
        name = m6.joint(j).name
        if name in root.PELVIS_JOINTS:
            continue
        qf[:, mf.joint(name).qposadr[0]] = q6[:, m6.jnt_qposadr[j]]  # all other joints are 1-dof slide/hinge
    return qf


def makima_tangent(p0, p1, p2, p3):
    """Port of Godot's Animation::makima_tangent (scene/resources/animation.cpp): slopes p0..p3 around one knot."""
    w1 = np.abs(p3 - p2) + np.abs(p3 + p2) * 0.5
    w2 = np.abs(p1 - p0) + np.abs(p1 + p0) * 0.5
    den = w1 + w2
    return np.where(den == 0, (p1 + p2) * 0.5, (w1 * p1 + w2 * p2) / np.where(den == 0, 1, den))


def makima(y, fps, t):
    """Modified Akima (Godot's makima_interpolate_in_time) of y (F, C) sampled at fps, evaluated at times t. Ends use akima_extrapolate (2*edge - next)."""
    F = y.shape[0]
    m = np.diff(y, axis=0) * fps  # slopes, (F-1, C)
    m = np.concatenate([[2 * m[0] - m[1]], m, [2 * m[-1] - m[-2]]])  # one pad each side
    m = np.concatenate([[2 * m[0] - m[1]], m, [2 * m[-1] - m[-2]]])  # two pads: m[k+2] is the slope of interval k
    tau = np.clip(t * fps, 0, F - 1 - 1e-12)
    i = tau.astype(int)
    w = (tau - i)[:, None]
    h = 1.0 / fps
    t_from = makima_tangent(m[i], m[i + 1], m[i + 2], m[i + 3])
    t_to = makima_tangent(m[i + 1], m[i + 2], m[i + 3], m[i + 4])
    s2, s3 = w * w, w * w * w
    return (2 * s3 - 3 * s2 + 1) * y[i] + (s3 - 2 * s2 + w) * h * t_from + (-2 * s3 + 3 * s2) * y[i + 1] + (s3 - s2) * h * t_to


def resample(qf, fps, hz=CONTROL_HZ):
    """Upsample to hz with makima (as Godot's animation player does); the root quaternion is sign-aligned, makima'd per component, renormalised."""
    F = qf.shape[0]
    t = np.arange(0, (F - 1) / fps + 1e-9, 1.0 / hz)
    q = qf.copy()
    for k in range(1, F):  # keep the quaternion on one hemisphere so components are continuous
        if (q[k, 3:7] * q[k - 1, 3:7]).sum() < 0:
            q[k, 3:7] = -q[k, 3:7]
    out = makima(q, fps, t)
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def velocities(mf, qpos, hz=CONTROL_HZ):
    qvel = np.zeros((qpos.shape[0], mf.nv))
    for i in range(qpos.shape[0] - 1):
        mujoco.mj_differentiatePos(mf, qvel[i], 1.0 / hz, qpos[i], qpos[i + 1])
    qvel[-1] = qvel[-2]
    return qvel


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fps = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--fps=")), 30.0)
    xml, src, dst = args
    m6 = mujoco.MjModel.from_xml_path(xml)
    mf, pos, quat = free_model(xml)
    q6 = np.load(src)
    assert q6.shape[1] == m6.nq, (q6.shape, m6.nq)
    qf = convert(m6, mf, pos, quat, q6)
    d6, df = mujoco.MjData(m6), mujoco.MjData(mf)
    err = 0.0
    for f in range(q6.shape[0]):
        d6.qpos[:], df.qpos[:] = q6[f], qf[f]
        mujoco.mj_kinematics(m6, d6)
        mujoco.mj_kinematics(mf, df)
        for b in CHECK_BODIES:
            err = max(err, np.abs(d6.body(b).xpos - df.body(b).xpos).max(), np.abs(d6.body(b).xmat - df.body(b).xmat).max())
    print(f"GATE body placement 6-joint vs free-joint: max error {err:.2e} -> {'PASS' if err < 1e-8 else 'FAIL'}")
    if err >= 1e-8:
        sys.exit(1)
    qpos = resample(qf, fps)
    qvel = velocities(mf, qpos)
    names = [mf.joint(j).name for j in range(mf.njnt)]
    np.savez(dst, qpos=qpos, qvel=qvel, joint_names=np.array(names), jnt_type=mf.jnt_type.copy(), frequency=CONTROL_HZ,
             split_points=np.array([0, qpos.shape[0]]))
    print(f"wrote {dst}: {qpos.shape[0]} frames at {CONTROL_HZ:.0f} Hz, nq {mf.nq}, nv {mf.nv}, root qvel0 {np.round(qvel[0, :6], 3)}")


if __name__ == "__main__":
    main()
