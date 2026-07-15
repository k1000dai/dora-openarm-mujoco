# Randomized object placement on reset — design

Date: 2026-07-15
Status: approved

## Goal

Allow the MuJoCo sim node to randomize the placement of non-arm scene objects
(e.g. `orange_cube` in the demo scene) so that each episode starts with a
different object configuration. Randomization applies at node startup and on
every `button_x` reset.

## Requirements

- Randomize the XY position and yaw (rotation about world Z) of freejoint
  scene objects, centered on their `--keyframe` pose.
- Ranges are set with CLI flags, uniform across all objects:
  - `--randomize-pos METERS` — uniform offset in `[-METERS, +METERS]` applied
    independently to X and Y.
  - `--randomize-yaw DEGREES` — uniform rotation in `[-DEGREES, +DEGREES]`
    about the world Z axis, composed onto the keyframe orientation.
- Both default to `0.0`; when both are zero the behavior is exactly the
  current keyframe snap (no behavior change for existing dataflows).
- Applied at startup (after the initial keyframe reset) and on every
  `button_x` reset (after `_reset_scene_objects`).
- Log each application per object:
  `[randomize] orange_cube: dx=+0.032 dy=-0.018 dyaw=+87°`.

## Out of scope (YAGNI)

- Hinge/slide scene joints (drawers, doors) keep their keyframe values.
- Arm joints are never randomized.
- No `--seed` flag (add later if reproducibility is needed).
- No collision/overlap avoidance; ranges are the user's responsibility.
- No per-object configuration file.

## Design

All changes live in `src/dora_openarm_mujoco/main.py`.

1. `_find_scene_joint_addrs` additionally returns the joint type per entry so
   freejoints can be identified: `(qpos_slice, qvel_slice, name, jnt_type)`.
2. New `SceneRandomizer` class:
   - Holds `pos_range` (m), `yaw_range` (rad), and a `np.random.default_rng()`.
   - `enabled` property: true when either range is > 0.
   - `apply(model, data, key_id, addrs)`: for each freejoint entry, offset
     `qpos[adr:adr+2]` from the keyframe XY by uniform noise and compose a
     world-Z rotation onto the keyframe quaternion (`mju_axisAngle2Quat` +
     `mju_mulQuat`), then `mj_forward`. No-op when disabled or `key_id < 0`.
3. Call sites:
   - `_setup_model`: after the initial keyframe reset (single-threaded at
     this point, no lock needed).
   - `button_x` handler in `_run_dora`: immediately after
     `_reset_scene_objects`, inside the existing viewer/data lock.
4. CLI: two new `argparse` options with non-negative float validation.

## Error handling

- Non-finite or negative CLI values are rejected at argparse time.
- Scenes with no freejoint objects (e.g. `cell`): randomizer is a no-op.
- Missing keyframe (`key_id < 0`): randomizer is a no-op, matching the
  existing reset behavior.

## Testing / verification

The repo has no test infrastructure. Verification:

- Manual: run the node headless with `--randomize-pos`/`--randomize-yaw` set,
  confirm the cube qpos varies within range across resets via a short script.
- `pre-commit run --all-files` for lint/format.

## Documentation

- Add both flags to the README arguments table and the `main.py` docstring.
