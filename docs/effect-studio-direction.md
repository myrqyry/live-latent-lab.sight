# Live Latent Lab — interactive effect authoring direction

2026-11-18. User confirmed the goal is to bring the useful ideas from
Effect House into the existing ecosystem, specifically our AI/ML OBS
plugins Live Latent Lab: Sight and Sound and the established HyperShaderFX
visual effect tools. This is an independent implementation inspired by
publicly described concepts; no ByteDance runtime, proprietary package
format, assets or copied implementation are required.

## Existing owner boundaries and evidence

- Sight (Loam checkout, base 9736652): owns visual inference, typed
  immutable snapshots, landmark/blendshape/head/gesture values, matte,
  depth, motion, source-local alignment, and native VRM/Live2D rendering.
  Value Router already writes compatible OBS numeric properties.
- HyperShaderFX (Loam checkout, inspected 63e8d7f): owns materials,
  shader rendering, native feedback buffers, reusable shader graph
  builder, real libobs compilation, and .hsfx asset packaging. Its
  existing Sight bridge consumes tracking/matte/depth/motion without
  rerunning inference. Work-in-progress builder C++ diffs are untouched.
- Sound (MQR checkout, inspected b82228d): owns audio ML/DSP, not the
  visual runtime. Its async native YAMNet candidate has real isolated OBS
  classification evidence; default Sound registrations are still
  quarantined. Do not promise live Sound signals until an opt-in verified
  producer/consumer boundary exists.
- ViiRM remains a producer of standard VRM/glTF assets, not a new effects
  runtime. Rose Cam can eventually consume portable authoring ideas but
  is not an initial owner here.

## First verified implementation slice: threshold gates

Extend Sight Value Router rather than duplicate inference or introduce
another plugin. A scalar signal becomes a stateful 0/1 gate with separate
activation/release thresholds. It drives a selected numeric property
using existing rate limiting and target validation. Acquired gates
release on missing tracking. Old continuous response is unchanged by
default. A real libobs target confirmed endpoint-only updates, and real
face inference followed by removal of the producer confirmed release.

## Tracker quality gate (2026-11-18)

User explicitly requested modern, optimized trackers as the foundation for
interactive effects. Preserve local inference and established output
contracts. Prefer version-qualified runtimes, consolidated tracking when
its outputs suffice, and one producer per necessary modality/framing.
Before changing models, delegates or capture scheduling, compare sustained
moving-video landmark quality, update age/p95 latency, CPU/GPU utilization,
render stalls and tracking-loss recovery on the actual target hardware.
Do not trade away gestures/head transforms by silently replacing dedicated
trackers with Holistic. Exact-frame dedup and modern default discovery
are implemented and verified; see
[tracking evidence](verification/2026-11-18-tracking-modernization.md).

## Next useful vertical slice (proposed, not yet implemented)

Use one actual visible HyperShaderFX effect as a target, e.g. a
face-anchored sparkle/reveal whose intensity is controlled by a Sight
expression or pinch gate. Verify pixels for inactive, active, mid-band,
release and tracking loss in isolated OBS with a real model or qualified
fixture. Check source coordinate framing and existing masks. Do not
claim arbitrary scene toggles or a complete AR engine.

Then extend only where creative requirements justify it:

1. Typed interaction graph: signal binding, threshold/hysteresis, edge
   detection, boolean composition, timers/cooldown, smoothing/envelopes,
   explicit missing-data behavior, then bounded numeric actions. Keep
   this distinct from per-pixel shader graph evaluation. Test deterministic
   state traces, reject cycles except explicitly stateful delayed edges,
   rate limit writes and validate destinations.
2. Visual authoring: extend the existing HyperShaderFX builder rather
   than fork a UI. Preview with fixture signals or opt-in live snapshots.
   Clearly distinguish simulated preview from native OBS output. Expose a
   scene/anchor hierarchy and inspector only where a concrete effect
   requires them.
3. Face materials and deformation: use existing face topology and
   barycentric/surface inputs. Make reusable masks, UV-local decals,
   deformations and tracked 2D/3D attachments independently verified;
   don't substitute a bounding box for a face mesh.
4. Sound inputs: expose qualified, timestamped classification events
   from Sound through an explicit read-only bridge. Preserve quarantine,
   avoid graphics/audio-thread blocking, expire stale events. Use
   confidence, class and debouncing rather than treating inferred
   laughter as a guaranteed fact.
5. Portable authored effect bundles: extend .hsfx only after agreeing
   on versioned, bounded interaction/anchor metadata. Reuse deterministic
   staging and hash verification; no automatic import/install, arbitrary
   scripts, hidden network calls, or embedded private model files.

## Acceptance boundary

Goal is useful interactive, reusable AR effects native to our OBS
ecosystem, not a standalone clone of Effect House. A first full
authoring loop should open an existing builder, select an existing
qualified Sight signal, bind it to an effect, preview, validate and
apply through the normal native effect path. It should serialize and
reload without losing identity, and degrade safely on missing inputs.
This is not delivered in the gate slice. All runtime tests remain
isolated; the desktop OBS instance, current scenes and installed plugins
are not touched.
