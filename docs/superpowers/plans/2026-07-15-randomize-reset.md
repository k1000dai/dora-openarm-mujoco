# Randomized Object Placement on Reset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--randomize-pos METERS` / `--randomize-yaw DEGREES` CLI flags that perturb freejoint scene objects (XY + yaw around their keyframe pose) at node startup and on every `button_x` reset.

**Architecture:** A new `SceneRandomizer` class in `src/dora_openarm_mujoco/main.py` applies uniform random offsets to freejoint qpos entries after the existing keyframe snap. `_find_scene_joint_addrs` gains a joint-type field so freejoints can be identified. Wiring: `_setup_model` creates and applies the randomizer once at startup; the `button_x` handler in `_run_dora` applies it after `_reset_scene_objects`.

**Tech Stack:** Python ≥3.10, MuJoCo ≥3.6, numpy, pytest (new dev dependency), uv, ruff (via pre-commit).

**Spec:** `docs/superpowers/specs/2026-07-15-randomize-reset-design.md`

## Global Constraints

- Both flags default to `0.0`; when both are zero, behavior must be byte-identical to current (keyframe snap only).
- Only `mjJNT_FREE` joints are perturbed; hinge/slide scene joints and all `openarm_*` bodies are untouched.
- New Python files start with the repo's 13-line Apache-2.0 header (copy from `src/dora_openarm_mujoco/main.py:1-13`).
- Run commands with the project venv: `source .venv/bin/activate` first, or prefix with `uv run`.
- Ruff (default settings, 88-col) must pass: `pre-commit run --all-files`.
- No new runtime dependencies; pytest goes in the `dev` dependency group only.
- `pre-commit run --all-files` may auto-fix files (ruff `--fix` + ruff-format). If it reports "files were modified", re-stage (`git add -u`) and re-run until green before committing.
- Log format on each application: `[randomize] orange_cube: dx=+0.032 dy=-0.018 dyaw=+87°`.

---

### Task 1: pytest setup + joint type in `_find_scene_joint_addrs`

**Files:**
- Modify: `pyproject.toml` (dev dependency group)
- Create: `tests/test_randomize.py`
- Modify: `src/dora_openarm_mujoco/main.py:205-240` (`_find_scene_joint_addrs`, `_reset_scene_objects`), `main.py:409` (name join in `_run_dora`), `main.py:504` (name join in `_setup_model`)

**Interfaces:**
- Consumes: existing `_find_scene_joint_addrs(model)`, `_reset_scene_objects(model, data, key_id, addrs)`.
- Produces: `_find_scene_joint_addrs(model) -> list[tuple[slice, slice, str, mujoco.mjtJoint]]` — the 4th element is the joint type. All later tasks unpack 4-tuples. `_reset_scene_objects` keeps its signature but accepts the 4-tuples.

- [ ] **Step 1: Add pytest as a dev dependency**

```bash
uv add --dev pytest
```

Expected: `pyproject.toml` gains a `[dependency-groups] dev = ["pytest>=..."]` section and `uv.lock` updates.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_randomize.py` (prepend the 13-line Apache header from `src/dora_openarm_mujoco/main.py:1-13`):

```python
import mujoco
import pytest

from dora_openarm_mujoco.main import (
    _find_scene_joint_addrs,
    _reset_scene_objects,
)
```

(`import numpy as np` is deliberately NOT added yet — nothing in this task uses
it and ruff would strip it as unused. Task 2 adds it.)

```python

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: FAIL — `test_find_scene_joint_addrs_returns_joint_type` and `test_find_scene_joint_addrs_slices` raise `ValueError: not enough values to unpack (expected 4, got 3)`. (`test_reset_scene_objects_accepts_four_tuples` passes today since 3-tuples unpack fine — it guards the switch to 4-tuples.)

- [ ] **Step 4: Add the joint type to `_find_scene_joint_addrs` and update all unpack sites**

In `src/dora_openarm_mujoco/main.py`, replace `_find_scene_joint_addrs` (lines 205–225):

```python
def _find_scene_joint_addrs(
    model: mujoco.MjModel,
) -> list[tuple[slice, slice, str, mujoco.mjtJoint]]:
    """Find joints belonging to non-arm bodies.

    Returns a list of ``(qpos_slice, qvel_slice, name, joint_type)`` covering
    freejoint objects and articulated fixtures (drawers, doors, ...).  Bodies
    whose name starts with ``openarm_`` are skipped so the arms are never
    teleported.
    """
    addrs: list[tuple[slice, slice, str, mujoco.mjtJoint]] = []
    for jnt_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[jnt_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("openarm_"):
            continue
        jnt_type = mujoco.mjtJoint(model.jnt_type[jnt_id])
        nq, nv = _JOINT_WIDTHS[jnt_type]
        qpos_adr = int(model.jnt_qposadr[jnt_id])
        qvel_adr = int(model.jnt_dofadr[jnt_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or body_name
        addrs.append(
            (
                slice(qpos_adr, qpos_adr + nq),
                slice(qvel_adr, qvel_adr + nv),
                name,
                jnt_type,
            )
        )
    return addrs
```

In `_reset_scene_objects` (line 237), change the loop header:

```python
    for qpos_sl, qvel_sl, _, _ in addrs:
```

In `_reset_scene_objects`'s parameter annotation (line 232) and `_run_dora`'s (line 360), change:

```python
    addrs: list[tuple[slice, slice, str]],
```
to
```python
    addrs: list[tuple[slice, slice, str, mujoco.mjtJoint]],
```
(in `_run_dora` the parameter is named `object_addrs`).

In `_run_dora` (line 409), change the name join:

```python
                    names = ", ".join(name for _, _, name, _ in object_addrs) or "(none)"
```

In `_setup_model` (line 504), change the name join:

```python
        names = ", ".join(name for _, _, name, _ in object_addrs)
```

Also update `_setup_model`'s return annotation (line 468):

```python
    mujoco.MjModel, mujoco.MjData, JointResolver, int, list[tuple[slice, slice, str, mujoco.mjtJoint]]
```
(ruff-format will wrap this; let it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: 3 passed.

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --all-files
git add pyproject.toml uv.lock tests/test_randomize.py src/dora_openarm_mujoco/main.py
git commit -m "refactor: include joint type in scene joint addresses"
```

---

### Task 2: `SceneRandomizer` class

**Files:**
- Modify: `src/dora_openarm_mujoco/main.py` (insert after `_reset_scene_objects`, around line 241)
- Test: `tests/test_randomize.py` (append)

**Interfaces:**
- Consumes: `_find_scene_joint_addrs` 4-tuples from Task 1.
- Produces: `SceneRandomizer(pos_range: float, yaw_range_deg: float)` with `.enabled -> bool` property and `.apply(model: mujoco.MjModel, data: mujoco.MjData, key_id: int, addrs) -> None`. Task 3 wires these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_randomize.py`. Also add `import numpy as np` to the top-level imports and `SceneRandomizer` to the existing `from dora_openarm_mujoco.main import (...)` block:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: FAIL with `ImportError: cannot import name 'SceneRandomizer'`.

- [ ] **Step 3: Implement `SceneRandomizer`**

In `src/dora_openarm_mujoco/main.py`, insert after `_reset_scene_objects` (after line 240), keeping the section-divider comment style:

```python
# ── scene-object randomization ─────────────────────────────────────────────────


class SceneRandomizer:
    """Uniform random XY/yaw perturbation for freejoint scene objects.

    Offsets are centered on the keyframe pose: XY position is shifted by
    uniform noise in ``±pos_range`` metres and the orientation is rotated
    about the world Z axis by uniform noise in ``±yaw_range_deg`` degrees.
    Non-free joints (drawers, doors) and arm bodies are left untouched.
    """

    def __init__(self, pos_range: float, yaw_range_deg: float):
        self.pos_range = pos_range
        self.yaw_range = math.radians(yaw_range_deg)
        self._rng = np.random.default_rng()

    @property
    def enabled(self) -> bool:
        return self.pos_range > 0.0 or self.yaw_range > 0.0

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        key_id: int,
        addrs: list[tuple[slice, slice, str, mujoco.mjtJoint]],
    ) -> None:
        """Perturb each freejoint in ``addrs`` around its keyframe pose."""
        if not self.enabled or key_id < 0:
            return
        for qpos_sl, _, name, jnt_type in addrs:
            if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
                continue
            adr = qpos_sl.start
            dx, dy = self._rng.uniform(-self.pos_range, self.pos_range, size=2)
            dyaw = self._rng.uniform(-self.yaw_range, self.yaw_range)
            data.qpos[adr] = model.key_qpos[key_id, adr] + dx
            data.qpos[adr + 1] = model.key_qpos[key_id, adr + 1] + dy
            dquat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dquat, np.array([0.0, 0.0, 1.0]), dyaw)
            quat = np.zeros(4)
            key_quat = model.key_qpos[key_id, adr + 3 : adr + 7]
            mujoco.mju_mulQuat(quat, dquat, key_quat)
            data.qpos[adr + 3 : adr + 7] = quat
            print(
                f"[randomize] {name}: dx={dx:+.3f} dy={dy:+.3f} "
                f"dyaw={math.degrees(dyaw):+.0f}°"
            )
        mujoco.mj_forward(model, data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --all-files
git add tests/test_randomize.py src/dora_openarm_mujoco/main.py
git commit -m "feat: add SceneRandomizer for freejoint XY/yaw perturbation"
```

---

### Task 3: CLI flags and wiring into startup + button_x reset

**Files:**
- Modify: `src/dora_openarm_mujoco/main.py` — `_parse_args` (~line 525), new `_non_negative_float` next to `_positive_float` (~line 515), `_setup_model` (~line 465), `_run_dora` (~line 351), `main` (~line 595, both thread-args tuples)
- Test: `tests/test_randomize.py` (append)

**Interfaces:**
- Consumes: `SceneRandomizer` from Task 2 (constructor, `.enabled`, `.apply`).
- Produces: `args.randomize_pos` / `args.randomize_yaw` (floats, default 0.0); `_setup_model(args)` now returns a 6-tuple ending in the `SceneRandomizer`; `_run_dora` takes `randomizer` as the parameter after `object_addrs`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_randomize.py`, adding `import argparse` and `import sys` to the imports and `_parse_args`, `_setup_model` to the `from dora_openarm_mujoco.main import (...)` block:

```python
def test_parse_args_defaults_to_disabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dora-openarm-mujoco"])
    args = _parse_args()
    assert args.randomize_pos == 0.0
    assert args.randomize_yaw == 0.0


def test_parse_args_rejects_negative_pos(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["dora-openarm-mujoco", "--randomize-pos", "-0.1"]
    )
    with pytest.raises(SystemExit):
        _parse_args()


def test_setup_model_applies_startup_randomization():
    args = argparse.Namespace(
        xml=None,
        scene="demo",
        keyframe="home",
        enable_collision=False,
        ctrl=False,
        randomize_pos=0.05,
        randomize_yaw=180.0,
    )
    model, data, _, key_id, addrs, randomizer = _setup_model(args)
    assert randomizer.enabled
    free = [a for a in addrs if a[3] == mujoco.mjtJoint.mjJNT_FREE]
    assert free
    adr = free[0][0].start
    key_xy = model.key_qpos[key_id, adr : adr + 2]
    xy = data.qpos[adr : adr + 2]
    assert np.all(np.abs(xy - key_xy) <= 0.05)


def test_setup_model_disabled_randomizer_keeps_keyframe():
    args = argparse.Namespace(
        xml=None,
        scene="demo",
        keyframe="home",
        enable_collision=False,
        ctrl=False,
        randomize_pos=0.0,
        randomize_yaw=0.0,
    )
    model, data, _, key_id, addrs, randomizer = _setup_model(args)
    assert not randomizer.enabled
    np.testing.assert_array_equal(data.qpos, model.key_qpos[key_id])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: the 4 new tests FAIL — `test_parse_args_defaults_to_disabled` with `AttributeError: 'Namespace' object has no attribute 'randomize_pos'`, `test_parse_args_rejects_negative_pos` with `DID NOT RAISE`, the two `_setup_model` tests with `ValueError: not enough values to unpack (expected 6, got 5)`.

- [ ] **Step 3: Implement the CLI flags and wiring**

3a. Add `_non_negative_float` directly below `_positive_float` (~line 523):

```python
def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed
```

3b. In `_parse_args`, add after the `--keyframe` argument:

```python
    p.add_argument(
        "--randomize-pos",
        type=_non_negative_float,
        default=0.0,
        metavar="METERS",
        help=(
            "On startup and button_x reset, offset each freejoint object's "
            "keyframe XY position by uniform noise in ±METERS (default: 0, off)"
        ),
    )
    p.add_argument(
        "--randomize-yaw",
        type=_non_negative_float,
        default=0.0,
        metavar="DEGREES",
        help=(
            "On startup and button_x reset, rotate each freejoint object about "
            "the world Z axis by uniform noise in ±DEGREES (default: 0, off)"
        ),
    )
```

3c. In `_setup_model`: extend the return annotation with `SceneRandomizer`:

```python
def _setup_model(
    args,
) -> tuple[
    mujoco.MjModel,
    mujoco.MjData,
    JointResolver,
    int,
    list[tuple[slice, slice, str, mujoco.mjtJoint]],
    SceneRandomizer,
]:
```

and replace the final `return model, data, mapper, key_id, object_addrs` with:

```python
    randomizer = SceneRandomizer(args.randomize_pos, args.randomize_yaw)
    if randomizer.enabled:
        print(
            f"[randomize] Enabled: pos=±{args.randomize_pos:g}m "
            f"yaw=±{args.randomize_yaw:g}°"
        )
        randomizer.apply(model, data, key_id, object_addrs)

    return model, data, mapper, key_id, object_addrs, randomizer
```

3d. In `_run_dora`, add a parameter after `object_addrs`:

```python
    object_addrs: list[tuple[slice, slice, str, mujoco.mjtJoint]],
    randomizer: SceneRandomizer,
```

and in the `button_x` handler, apply after the reset (inside the lock):

```python
                if pressed and not button_x_prev:
                    with _lock(viewer, data_lock):
                        _reset_scene_objects(model, data, reset_key_id, object_addrs)
                        randomizer.apply(model, data, reset_key_id, object_addrs)
```

3e. In `main()`, update the unpack:

```python
    model, data, mapper, reset_key_id, object_addrs, randomizer = _setup_model(args)
```

and in BOTH `threading.Thread(target=_run_dora, args=(...))` tuples (viewer and headless branches), insert `randomizer` right after `object_addrs`:

```python
                    reset_key_id,
                    object_addrs,
                    randomizer,
                    args.ctrl,
                    args.debug_frames,
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest tests/test_randomize.py -v`
Expected: 14 passed.

- [ ] **Step 5: Smoke-test startup randomization end-to-end**

```bash
uv run python -c "
from types import SimpleNamespace
from dora_openarm_mujoco.main import _setup_model
args = SimpleNamespace(xml=None, scene='demo', keyframe='home',
                       enable_collision=False, ctrl=False,
                       randomize_pos=0.05, randomize_yaw=180.0)
_setup_model(args)
"
```

Expected output includes `[randomize] Enabled: pos=±0.05m yaw=±180°` and one `[randomize] orange_cube: dx=... dy=... dyaw=...°` line.

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --all-files
git add tests/test_randomize.py src/dora_openarm_mujoco/main.py
git commit -m "feat: add --randomize-pos/--randomize-yaw reset randomization"
```

---

### Task 4: Documentation

**Files:**
- Modify: `README.md` (Arguments table ~line 91, Inputs table `button_x` row ~line 71)
- Modify: `src/dora_openarm_mujoco/main.py` module docstring (CLI section ~line 71, `button_x` input description ~line 50)

**Interfaces:**
- Consumes: flag names/semantics from Task 3. No code changes.

- [ ] **Step 1: Update the README arguments table**

Add after the `--keyframe` row:

```markdown
| `--randomize-pos METERS` | `0` | On startup and `button_x` reset, offset each freejoint object's keyframe XY position by uniform noise in ±METERS. `0` disables. |
| `--randomize-yaw DEGREES` | `0` | On startup and `button_x` reset, rotate each freejoint object about the world Z axis by uniform noise in ±DEGREES. `0` disables. |
```

In the Inputs table, extend the `button_x` description's final sentence:

```markdown
| `button_x` | `bool[1]` | X button state. Edge-triggered: on press every scene joint on non-arm bodies (freejoint objects plus fixtures like drawers/doors) snaps back to the `--keyframe` pose, then freejoint objects are re-perturbed when `--randomize-pos`/`--randomize-yaw` are set; the button must be released to re-arm. |
```

- [ ] **Step 2: Update the module docstring in `main.py`**

In the `button_x` input description (after "next reset can fire."), add:

```
    When ``--randomize-pos``/``--randomize-yaw`` are set, freejoint objects
    are additionally perturbed around the keyframe pose after each reset.
```

In the CLI arguments section, add after the `--keyframe` entry:

```
--randomize-pos METERS  (default: 0, off)
    On startup and button_x reset, offset each freejoint object's keyframe
    XY position by uniform noise in ±METERS.

--randomize-yaw DEGREES  (default: 0, off)
    On startup and button_x reset, rotate each freejoint object about the
    world Z axis by uniform noise in ±DEGREES.
```

- [ ] **Step 3: Lint, full test run, and commit**

```bash
pre-commit run --all-files
uv run pytest tests/ -v
git add README.md src/dora_openarm_mujoco/main.py
git commit -m "docs: document --randomize-pos/--randomize-yaw flags"
```

Expected: pre-commit all green, 14 passed.
