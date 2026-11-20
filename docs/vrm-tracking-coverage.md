# Sight native VRM: tracking-to-rig coverage audit (2026-11-04)

Read-only source audit prompted by the user's repeated discovery of omitted
spine, head-yaw, and wrist behavior. Green CTests and pixel tests prove
specific behaviors, **not** full retargeting.

Evidence inspected: `src/vrm/vrm_tracking.cpp`, `src/native-vrm-source.cpp`,
`src/core/tasks_tracker.cpp`, `src/vrm/vrm_asset.hpp`, both private
PeepoSight.vrm and OtakuSight.vrm bone/expression maps, and the normal
OBS Peepo scene source settings.

## Post-audit progress — 2026-11-04

**Face dropout continuity (priority 1) is implemented and deployed** in
local source commit `5286ef3`. Head and expression tracking now hold
through missing signals, then optionally ease to rest and ease on
reacquisition; the new Head/Expression Ease slider uses existing
lost-tracking hold/return settings. 16/16 CTests, Peepo and Otaku native
pixel tests passed. Current OBS log confirms runtime marker
`FACE_CONTINUITY_ACTIVE_20261104_F284` and live face/body tracker output.
Actual webcam disappearance/reappearance has not been user-validated.

The table below preserves the **original audit snapshot** and describes
the missing face behavior *before* this correction. Other gaps remain
open; next is joint and signal coverage diagnostics.

### Metric world-landmark data path — 2026-11-04

Sight now **preserves** the optional 33-point hip-centered `Body`
world landmarks and up to two 21-point hand-centered `Hand` world
landmark arrays in typed tracking snapshots. Real MediaPipe native
inference confirmed these arrays using the installed runtime and task
models. This does **not** change the VRM tracking mapper, which still
uses normalized image points. Do not interpret data preservation as
calibrated 3D arm retargeting, timestamp synchronization, or a solved
body-to-hand coordinate transform. The real OBS scene runs the new module, and 478/33/21
face/body/hand landmarks are detected without changing the
avatar's animation settings. Backup of the previous installed binary
and both scene collections:
`~/.config/obs-studio/plugin-backups/sight-before-world-landmarks-20261104-192320/`.
The original coverage table below describes the pre-change data path.

### Joint coverage diagnostic — 2026-11-04

The native VRM source now exposes a **read-only Rig coverage** report
at the bottom of Properties. It uses the actual avatar bone map,
observation timestamps, source signal age and settings to distinguish
**not mapped, never driven, disabled, waiting, tracked input, held,
returning and near rest**. Source counts include optional 3D world
landmark presence. A relaxed default arm with no real camera input is
**waiting**, not falsely tracked. Snapshot refreshes at most once a
second; **reopen Properties to see a new snapshot**. No retarget
math or other signal ingestion path is changed.

Full 18-test suite and native Peepo/Otaku libobs renderer tests passed.
The words *tracked* and *near rest* describe accepted input and
estimated exponential settling, not proof of physically correct pose.
Runtime marker: RIG_COVERAGE_PROPERTIES_ACTIVE_20261104_D726.
The original audit table below records the historic prior state.

### Finger flexion direction correction — 2026-11-04

Sight now chooses local-X curl **sign** individually from each
finger's authored rest joint axes and the same hand's palm plane,
rather than using positive local-X on every finger. Peepo and Otaku
have different local-axis directions; Otaku's index MCP even
differs from its own index PIP. Red/green unit tests and actual private
VRM checks confirmed all 60 finger joints have a valid authored
inward-bend direction. The change preserves finger easing,
held tracking, wrist attitude and hand-driven arms; full rig-dependent
finger splay, richer thumb opposition and real-camera qualitative
validation are still outstanding.

The coverage table below was written *before* this direction fix.
It remains correct that flexion still rotates on one local bone axis
rather than implementing full three-dimensional finger articulation.

## Current coverage and known gaps

| Source | Presently animated | Available but not applied / limitations |
| --- | --- | --- |
| Face: 478 landmarks, blendshapes, transformation | Head pitch/yaw/roll; AA mouth from jawOpen; OU from pucker/funnel; blink; happy/smile when a matching VRM expression exists | Other facial classes including brows, squint, cheeks, frown, detailed visemes and gaze; separate eye bones are never driven; face pose and expressions **snap to neutral** on tracking loss (no smoothing/hold state). |
| Body: 33 landmarks | Upper/lower arm segment aim; shoulder/hip-relative spine lean; chest tilt; shoulder-only fallback sway when hips hidden | Shoulder, upperChest, neck, hips, legs/knees/ankles/feet/toes never written as output rotations. No root position/hip translation or world-space calibration. |
| Hands: up to 2 x 21 landmarks | Finger curl per joint, palm/wrist rotation, optional camera-plane two-bone arm IK, side-matching, hold/ease | No finger splay or independent spread, rig-specific bend axes, calibrated thumb opposition, or dependable absolute wrist depth. `gesture`, `gesture_score` and `pinch` are collected but not applied to **the VRM animator**. |
| Fusion | Independent face, body and hand channels; per-joint continuity for body and hands | No unified frame timestamps or consistent coordinate calibration across channels. |
| Pose representation | Joint quaternion rotations and expression morph weights | No animated bone/root translation. |

## Actual VRM differences

- **OtakuSight.vrm**: 52 humanoid bones including hips/spine/chest/
  upperChest/neck, shoulder/arm/hand, 30 finger bones, legs/feet/toes.
  **No leftEye/rightEye bones**. Standard expression presets: AA, EE, IH,
  OH, OU, Blink. 23 original custom shape keys; original recognized-gesture
  targets `fist`, `handopen`, `fingerpoint`, `victory`, `rocknroll`,
  `handgun` and `thumbsup` exist but are unused. Smile->happy cannot
  animate Otaku because it has no authored happy expression. EE/IH/OH
  visemes exist but aren't driven by the current face mapping.
- **PeepoSight.vrm**: 53 humanoid bones, including separate leftEye/
  rightEye (unused by Sight), shoulder, neck, both leg/foot chains and
  all finger bones. No upperChest. Has happy, angry, sad and surprised
  expression presets; only happy is driven among those.

## Verification boundaries

- 16/16 full CTests and separate synthetic Peepo/Otaku native pixel
  checks passed after wrist work. These establish specific changes to
  geometry, not that all joints are mapped, directions correct in the
  real webcam, or that high-frequency capture stays stable.
- Most recent saved Peepo Sight Preview scene points to **OtakuSight.vrm**
  and had `hands_drive_arms=false`: hand rotation and finger animation
  can operate without hand-position IK, but the gesture-hand wrist
  **position** will not drive the upper arm unless toggled on.
- Live hand detections in a later OBS log establish the task can
  return 21 landmarks, not that the mapped avatar is anatomically
  correct under real-world camera motion.

## Proposed next improvements, not implemented

1. Prevent head/expression snapping on face dropout and smooth recovery.
   Regression coverage: missing face, reacquisition, independent eyes.
2. Expose a **joint coverage/status diagnostic** (which bones are
   mapped, receiving valid observations, held, returning, or never
   driven), with per-channel timing and handedness. Avoid repeated
   feature-completeness claims based on a narrow passing smoke test.
3. Calibrate shoulder, neck and upperChest distribution safely without
   double-rotating the head or disrupting current hand/body controls.
4. Rig-aware finger bend axes, splay and thumb opposition (separate
   rig fixtures for Otaku and Peepo), with controls for shaky depth.
5. More expressions and eye gaze only where the avatar provides
   actual expression or eye targets.
6. Optional recognized gesture morphs for Otaku, with safeguards
   against duplicate finger deformation (bones + gesture morph).
7. Hip/leg/root motion only where the camera captures meaningful
   full-body landmarks. A desktop head-and-hands frame cannot
   recover unseen knees, ankles and depth accurately.

No code behavior, user assets, settings or OBS plugin was changed by
this audit. Audit findings should not be presented as implemented work.
