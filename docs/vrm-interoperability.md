# ViiRM → Sight: one portable VRM contract

## Ownership

- **ViiRM authoring:** canonical procedural geometry, T-pose skeleton,
  VRM 1.0 humanoid mappings, expressions, textures, skinned garments and
  optional spring metadata. Its exported T-pose is intentional, independent
  of preview idle animation.
- **Sight consuming:** render either valid glTF `matrix` or TRS node encoding,
  apply head/body/arms/wrists, optional finger motion only where humanoid
  finger joints exist, and own lost tracking/easing. The producer must not
  fake extra TRS fields to compensate for a consumer bug.
- **Peepo and Otaku** exercise detailed finger rigs; **ViiRM** supplies a
  matrix-authored simplified humanoid. Different proportions and mesh
  coverage are intentional. Never turn one avatar's quirks into universal
  VRM rules.

## Repeatable cross-app verification

    SIGHT_BUILD_DIR=/tmp/sight-vrm-build tests/check_viirm_sight.sh       "$HOME/Downloads/Random_Mii_#610.vrm"

Any user-exported ViiRM VRM 1.0 can be supplied as the positional file.
The script builds configured targets, checks equivalent TRS/matrix
retargeting and runs headless native libobs **default, body, hand**
smokes against the real file. It does not modify the avatar, OBS, scene,
or installed plugin. Private VRMs never become repository fixtures.
Standard `ctest --test-dir /tmp/sight-vrm-build --output-on-failure -j4`
includes synthetic parity even without the private file.

ViiRM's independent producer check is `pnpm exec vitest run
src/vrmInterop.test.ts`, verifying actual exported matrix/TRS offsets,
left/right arm parentage and canonical horizontal T-pose.

## Actual evidence, 2026-11-04

`Random_Mii_#610.vrm` is glTF 2.0/VRM 1.0: 125 nodes, 23 humanoid bones,
37 meshes, 5 skins, 7 textures, 14 expression presets. 96 nodes have
`matrix` transforms and none separate TRS. No animation clip, no
humanoid finger bones. Sight's renderer handled matrices but arm
retargeting read absent TRS translations as zeros. Red/green tests
reproduced zero arm corrections then verified four relaxed rotations,
body signal and hand-driven IK after the shared `local_offset` fix.
Real libobs tested default face, body, relaxed arms, tracking-loss
hold/eased reacquisition and hand-driven arms; full 20/20 CTests passed.
The read-only rig coverage explicitly reports missing finger bones.

## Known gaps

- Rigid ViiRM hands cannot individually curl: extending ViiRM requires
  finger chains **and deformable geometry**, not metadata alone.
- Sight does not automatically use ViiRM browser lighting/postFX or
  reproduce `VRMC_springBone` simulations. Spring metadata in a file
  is not native Sight runtime physics.
- Real webcam framing, inference delivery rate, 3D coordinate calibration
  and subjective motion smoothness remain separate runtime validation.
- Normal desktop OBS must load the new module on its next restart;
  headless libobs proves the plugin code, not the current GUI process.
The native easing smoke verifies continued rendered motion after first
reacquisition; precise easing interpolation belongs to unit tests, not
a geometry- and scheduler-dependent pixel-ratio threshold.

For new compatibility bugs: reproduce in this contract first,
fix in the producer or consumer that owns the behavior, verify all
three real rig families and preserve prior rollback checkpoints.

## Installed local checkpoint

The actual user plugin at
`~/.config/obs-studio/plugins/live-latent-lab-sight/bin/64bit/live-latent-lab-sight.so`
is SHA256 `5f7d98bfdc736a03e0f402fc73778bf2b2732605ae3312b0c7469ae7dc5da906`,
with the preceding plugin and Peepo/Otaku JSON snapshots at
`~/.config/obs-studio/plugin-backups/sight-before-matrix-vrm-interop-20261104-215338/`.
The actual installed binary passed native libobs model-loading and face
checks for ViiRM, Peepo and Otaku; ViiRM real body test passed three
separate runs and hand-mode tests passed. The user's existing OBS
process was **left running on its previously mapped plugin**. Restart
OBS normally before expecting matrix retargeting in the desktop GUI.
No push or Vercel deployment was performed.
