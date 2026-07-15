# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mujoco
import numpy as np
import pytest

from dora_openarm_mujoco.main import (
    _find_scene_joint_addrs,
    _reset_scene_objects,
    SceneRandomizer,
)


# Freejoint object + slide fixture + an openarm_ body that must be excluded.
_TEST_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="orange_cube" pos="0 0 0">
      <freejoint name="orange_cube"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
    </body>
    <body name="drawer" pos="1 0 0">
      <joint name="drawer_slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
    <body name="openarm_left_link" pos="2 0 0">
      <joint name="openarm_left_joint1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.5 0.2 0.1 1 0 0 0 0.15 0.4"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture()
def scene():
    model = mujoco.MjModel.from_xml_string(_TEST_XML)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    addrs = _find_scene_joint_addrs(model)
    return model, data, key_id, addrs


def test_find_scene_joint_addrs_returns_joint_type(scene):
    _, _, _, addrs = scene
    by_name = {name: jnt_type for _, _, name, jnt_type in addrs}
    assert by_name == {
        "orange_cube": mujoco.mjtJoint.mjJNT_FREE,
        "drawer_slide": mujoco.mjtJoint.mjJNT_SLIDE,
    }


def test_find_scene_joint_addrs_slices(scene):
    _, _, _, addrs = scene
    slices = {name: (qpos_sl, qvel_sl) for qpos_sl, qvel_sl, name, _ in addrs}
    assert slices["orange_cube"] == (slice(0, 7), slice(0, 6))
    assert slices["drawer_slide"] == (slice(7, 8), slice(6, 7))


def test_reset_scene_objects_accepts_four_tuples(scene):
    model, data, key_id, addrs = scene
    data.qpos[0] = 2.0
    data.qvel[0] = 1.0
    data.qpos[7] = 0.25
    _reset_scene_objects(model, data, key_id, addrs)
    assert data.qpos[0] == 0.5
    assert data.qvel[0] == 0.0
    assert data.qpos[7] == 0.15
    # openarm hinge is not a scene joint; untouched by reset
    assert data.qpos[8] == 0.4


def test_disabled_randomizer_is_noop(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.0, 0.0)
    assert not r.enabled
    before = data.qpos.copy()
    r.apply(model, data, key_id, addrs)
    np.testing.assert_array_equal(data.qpos, before)


def test_missing_keyframe_is_noop(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.05, 180.0)
    before = data.qpos.copy()
    r.apply(model, data, -1, addrs)
    np.testing.assert_array_equal(data.qpos, before)


def test_pos_randomization_stays_in_bounds(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.05, 0.0)
    assert r.enabled
    for _ in range(50):
        r.apply(model, data, key_id, addrs)
        x, y, z = data.qpos[0:3]
        assert 0.45 <= x <= 0.55
        assert 0.15 <= y <= 0.25
        assert z == 0.1
        np.testing.assert_array_equal(data.qpos[3:7], [1.0, 0.0, 0.0, 0.0])


def test_yaw_randomization_rotates_about_z_only(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.0, 180.0)
    for _ in range(50):
        r.apply(model, data, key_id, addrs)
        w, qx, qy, qz = data.qpos[3:7]
        assert qx == pytest.approx(0.0, abs=1e-12)
        assert qy == pytest.approx(0.0, abs=1e-12)
        assert w**2 + qz**2 == pytest.approx(1.0)
        np.testing.assert_array_equal(data.qpos[0:3], [0.5, 0.2, 0.1])


def test_offsets_center_on_keyframe_not_current_pose(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.05, 0.0)
    data.qpos[0] = 5.0  # drift far away; apply must recenter on keyframe x=0.5
    r.apply(model, data, key_id, addrs)
    assert 0.45 <= data.qpos[0] <= 0.55


def test_non_free_joints_untouched(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.05, 180.0)
    r.apply(model, data, key_id, addrs)
    assert data.qpos[7] == 0.15  # drawer_slide keeps keyframe value
    assert data.qpos[8] == 0.4  # openarm hinge not in addrs


def test_consecutive_applies_differ(scene):
    model, data, key_id, addrs = scene
    r = SceneRandomizer(0.05, 180.0)
    r.apply(model, data, key_id, addrs)
    first = data.qpos[0:7].copy()
    r.apply(model, data, key_id, addrs)
    assert not np.array_equal(data.qpos[0:7], first)
