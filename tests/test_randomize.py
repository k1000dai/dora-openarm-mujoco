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
import pytest

from dora_openarm_mujoco.main import (
    _find_scene_joint_addrs,
    _reset_scene_objects,
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
