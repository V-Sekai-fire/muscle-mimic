"""MS-Human-700 pelvis root: 3 slides + 3 hinges  <->  one free joint (7 qpos, wxyz quaternion).

MS-Human-700 (MuJoCo Menagerie) puts six joints on the pelvis body, in this order:
  pelvis_tz (slide, local z), pelvis_ty (slide, local y), pelvis_tx (slide, local x),
  pelvis_tilt (hinge, local z), pelvis_list (hinge, local x), pelvis_rotation (hinge, local y),
and the body itself sits at `pos` with a fixed `quat` (90 deg about x). MuJoCo applies the joints of a body in
listed order, each axis expressed in the frame left by the previous joints, so the hinges are INTRINSIC
z-x-y: R = Rz(tilt) @ Rx(list) @ Ry(rotation), taken in the body's own (fixed-quat) frame.

MuscleMimic wants a free root joint. After `add_freejoint()` the body frame equals the pose in qpos, so
  world_pos  = body_pos + R0 @ (tx, ty, tz)     with R0 = rotation of the body's fixed quat
  world_rot  = R0 @ Rz(tilt) @ Rx(list) @ Ry(rotation)
Quaternions are MuJoCo order, scalar first (w, x, y, z).
"""

import numpy as np

# axes of the six pelvis joints in the body frame, as in MS-Human-700.xml
SLIDE_AXES = {"pelvis_tz": (0, 0, 1), "pelvis_ty": (0, 1, 0), "pelvis_tx": (1, 0, 0)}
HINGE_ORDER = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")  # about z, x, y
PELVIS_JOINTS = ("pelvis_tz", "pelvis_ty", "pelvis_tx") + HINGE_ORDER


def _rot(axis: str, a: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) about a coordinate axis, angle a (...,), active, column vectors."""
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    if axis == "x":
        m = [[o, z, z], [z, c, -s], [z, s, c]]
    elif axis == "y":
        m = [[c, z, s], [z, o, z], [-s, z, c]]
    else:
        m = [[c, -s, z], [s, c, z], [z, z, o]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion(s) (..., 4) -> rotation matrices (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> wxyz quaternions (..., 4), w >= 0 (branch-stable per frame)."""
    R = np.asarray(R)
    out = np.empty(R.shape[:-2] + (4,))
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    for idx in np.ndindex(*R.shape[:-2]):
        m = R[idx]
        if t[idx] > 0:
            s = 2 * np.sqrt(1 + t[idx])
            q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
            q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] >= m[2, 2]:
            s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
            q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
        else:
            s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
            q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
        q = np.array(q)
        out[idx] = q if q[0] >= 0 else -q
    return out


def hinges_to_rotation(tilt, list_, rotation, body_quat) -> np.ndarray:
    """World rotation of the pelvis body: R0 @ Rz(tilt) @ Rx(list) @ Ry(rotation) (intrinsic z-x-y after the fixed body quat)."""
    R0 = quat_to_mat(np.asarray(body_quat, float))
    return R0 @ _rot("z", np.asarray(tilt, float)) @ _rot("x", np.asarray(list_, float)) @ _rot("y", np.asarray(rotation, float))


def root_to_freejoint(q6: np.ndarray, body_pos, body_quat) -> np.ndarray:
    """q6 (..., 6) in PELVIS_JOINTS order (tz, ty, tx, tilt, list, rotation) -> free-joint qpos (..., 7) = xyz + wxyz."""
    q6 = np.asarray(q6, float)
    tz, ty, tx = q6[..., 0], q6[..., 1], q6[..., 2]
    R0 = quat_to_mat(np.asarray(body_quat, float))
    pos = np.asarray(body_pos, float) + np.einsum("ij,...j->...i", R0, np.stack([tx, ty, tz], axis=-1))
    quat = mat_to_quat(hinges_to_rotation(q6[..., 3], q6[..., 4], q6[..., 5], body_quat))
    return np.concatenate([pos, quat], axis=-1)


def prepare_spec(spec, body2sites: dict, root_name: str = "root"):
    """Edit an MjSpec of MS-Human-700 in place for MuscleMimic: free root, mimic sites, muscle ctrlrange [-1, 1], ground plane.

    The six pelvis joints become one free joint named `root_name`. Mimic sites follow the MyoFullBody convention (a small box
    at the body origin, group 4). Returns the pelvis body pos and quat as they were before the edit, which
    `root_to_freejoint` needs to convert a retargeted clip.
    """
    import mujoco  # local import: the conversion functions above stay numpy-only

    pelvis = spec.body("pelvis")
    pos, quat = np.array(pelvis.pos), np.array(pelvis.quat)
    for name in PELVIS_JOINTS:
        spec.delete(spec.joint(name))
    pelvis.add_freejoint().name = root_name
    for body_name, site_name in body2sites.items():
        spec.body(body_name).add_site(
            name=site_name,
            group=4,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.075, 0.05, 0.025],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=[0.0, 0.0, 0.0],
        )
    for act in spec.actuators:
        if act.dyntype == mujoco.mjtDyn.mjDYN_MUSCLE:
            act.ctrlrange = [-1.0, 1.0]
            act.ctrllimited = True
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.worldbody.geoms):
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1)
    return pos, quat
