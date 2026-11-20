# Hand Ripples moved to HyperShaderFX

Use LLL.sight Tracking in Hands & gestures mode to publish a hand channel
without an overlay. On the same source, add HyperShaderFX and choose
Sight: Hand Ripples; enable shared Sight snapshots and choose the hand
channel. The original fingertip refraction and eight decaying wavefronts
are now maintained on the consuming shader side. No second inference model
or standalone Sight Touch Field filter is necessary.

See the HyperShaderFX repository's data/shaders/SHADERS.md and
scripts/verify-sight-interop.sh. Its tests now own wave emission, geometry
checks, GPU decay and real two-plugin output. Sight retains a native-wide
hand snapshot test for the producer geometry contract.
