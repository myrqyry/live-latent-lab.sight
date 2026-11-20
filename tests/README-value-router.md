# Sight Value Router — waveforms and numeric effect modulation

## What actually works

A separate **Sight: Value Router (Signals + Waves)** OBS video filter does
not modify picture pixels; it reads/generates one scalar, remaps it, and writes
a selected **numeric property of another OBS filter**, via the public
`obs_source_get_settings`, `obs_data_set_double/int`, and
`obs_source_update` APIs.

The router can be attached to the same OBS source as its target filter
(default), or can select a named other source. Its Source and Effect controls
enumerate OBS sources and filters; the Numeric Setting control shows only
integer/float properties from the selected effect. Existing manually saved names remain visible if unavailable. Invalid/non-numeric properties are ignored.

A router rejects writes to itself or any other Value Router, to avoid
router cycles. It only updates a target if the requested numeric output
changes, and caps update frequency (default 20/s). Source lifetime is
handled with OBS-acquired references; no target properties are modified
at plugin startup or when the source or filter is unselected.

### Discovered input/output choices

Discovery is shared with the native receiver, Touch Field, VRM and Live2D
property controls. Other OBS plugins can use the receiver's optional
`sight_tracking_list_v1` procedure (caller-owned `obs_data_t *inventory`,
`version=1`, `channels` array containing `channel`, `label`, `kind`,
`enabled`, `ambiguous`, `gestures`). The kind numbers are the
snapshot ABI constants. This is a control-thread metadata query, not
a per-frame snapshot or model start. Reopen consumer properties to refresh.
HyperShaderFX uses the inventory for its supported face/body/hand inputs;
WholeBody is not a compatible shader input yet.

Open the router properties or use **Refresh inputs and outputs** to scan
the currently configured Sight tracking producers and OBS destinations.
Channels show source/filter names, topology and observed status (live,
no subject, idle, disabled, or duplicate). This includes configured
producers that have not yet inferred a frame. Selecting a channel
narrows the tracking value picker to its topology; Holistic hand
channels offer pinch/coordinates but not gesture classification.
Point/Expression is an ID-backed list with topology labels, real
observed expression category names when available, and numeric fallback
labels for unmapped points. No subject does not erase the configuration.

Destination sources (including scenes), filters and available numeric
properties are discovered on open, dependent selections and explicit
refresh. A saved unavailable channel, target or index is retained as
a disabled choice instead of silently selecting a replacement.
Saved settings use the existing property keys and values. Lists are
non-editable so OBS saves their actual IDs rather than the descriptive
display text. Refresh changes only picker metadata, not saved routing
settings or inference state. There is no background widget rebuild or
live polling while the dialog is open. Reopen or refresh after external
source/filter changes.

The input inventory currently covers Sight's configured face, body,
hand, Holistic and RTMW tracking filters visible to OBS plus the
router's attached parent. It is not a catalog of all Sound signals,
arbitrary object detector outputs, or every shader input/output.
Names are not stable expression identity across different model
topologies: existing expression index bindings remain unchanged.

### Inputs

**Wave oscillator**: Sine, Cosine, Square, Pulse, Saw Up/Down, Triangle,
Absolute Sine, Steps, Random Hold, Smooth Random. User settings include
frequency (0.01–10 Hz), amplitude, phase offset, pulse width, step count,
invert. Pulses are unipolar (rest at midpoint of output range). Each router
has its own deterministic local RNG; no audio or ML inference required.

**Tracking signal**: Read a named Sight channel (default `face`) and
select normalized face/body landmark X/Y, face expression score,
hand 0/1 fingertip point X/Y, normalized thumb-index pinch strength,
or confidence of a selected hand gesture class.
Data is the latest publisher's already-smoothed tracking result; no new
MediaPipe instance is started. Missing/expired/no-face/wrong-type continuous data releases the current
setting lease and restores its original value if still owned. Signal channel uses existing bus expiry
(100–5000 ms).

### Output

Input 0..1 maps linearly into the configured output min/max. The result
is clamped again to the target OBS numeric property limits; integer settings
are rounded. Other settings on the destination filter are retained.
Choose **Response → Threshold gate** for a discrete interaction instead
of the default continuous mapping. Activate at 0.65 and release at 0.45
(by default); the band between them retains state, so noisy landmarks
do not chatter. Output is either the configured minimum or maximum.
The controls require 0 ≤ release < activate ≤ 1. Invalid thresholds
produce no writes; missing input before the first valid observation also
produces no writes. Once acquired, missing/expired tracking releases the
gate to the minimum. Reconfiguration clears internal gate history;
disabling or removing the router restores the acquired baseline. Callers
should distinguish the gate output minimum on tracking loss from disabling
the router itself.
The router still only writes numeric properties, not boolean properties,
filter enable/disable, scene transforms, or external plugin signal APIs.

### Validation

- `sight_value_modulation_unit` checks waveform phase progression,
  multiple distinct waveforms, stepped output, typed signal values,
  index rejection and remapping without an OBS runtime.
- `sight_value_router_obs_wave` verifies that a **real OBS numeric filter
  setting** changes repeatedly over time, with more than five writes
  spanning at least 20 output units, from a waveform.
- `sight_value_router_obs_face` verifies that a *separate* real MediaPipe
  Face Motion filter publishes a signal read by the router and
  updates the target filter numeric setting. The script requires actual
  face inference and rejects rendering errors.
- The threshold gate tests exercise hysteresis, invalid thresholds,
  acquisition, release on missing samples, and mid-band recovery.
  `sight_value_router_obs_gate` uses a real libobs target and checks only
  the two endpoints are written repeatedly. `sight_value_router_obs_face_gate`
  uses the real Face Motion model, verifies activation and removes the
  producer to verify release. Both require the unique gate runtime marker.
- All original Sight filters have preserved tests.

The user's `obs-move-transition` fork informed the design. Its Wave Move
filter implements 11 waveforms and direct target settings or transforms.
Sight uses independently written modulation/routing code and does not
copy GPL code from Move.

### Reversible parameter modulation (2026-11-18)

Numeric writes are coalesced onto the OBS UI task path. Each router
acquires a weak-reference lease on one exact target filter and property;
a competing router cannot write the same property until the owner
releases it. The baseline preserves whether a user override existed.
Manual changes observed between writes become the baseline. Restoration
does not overwrite a differing current value. Disable, detach, retarget,
invalid continuous input and invalid destinations release ownership.
Tracking gates retain their established minimum-on-loss behavior.

The UI-thread save callback releases the lease and invalidates pending
old-revision samples before synchronous source JSON serialization.
The next eligible tick resumes modulation. The real libobs smoke tests
serialize the actual parent source with both filter orders, then verify
the baseline in parsed JSON and resumed modulation. This is not proof
of arbitrary worker-thread saves, saving only an external target
without the router, crash recovery, or asynchronous third-party
serialization. No promise of cross-plugin ownership against unrelated
modulators; indistinguishable same-value edits cannot be detected.
OBS frontend UI dispatch is required.

Evidence: `sight_value_router_obs_restore` RED before restoration,
GREEN lifecycle and contention, then RED save before the callback and
GREEN serialized JSON in both orders. Existing gate/face/hand/wave
tests pass. Full configured suite 87/87 (48.32 s).
Logs /tmp/sight-router-restore-red.log, /tmp/sight-router-save-red.log,
 /tmp/sight-router-save-green.log, /tmp/sight-router-final-ctest.log.
No installed binary or scene edits; desktop OBS running.
