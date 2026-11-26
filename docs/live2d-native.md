# Optional native Live2D development build

Sight retains its native Live2D adapter in source control. The default
validated build does not compile or register it. The adapter uses Cubism
Core through a separately acquired SDK; the proprietary headers, library
and sample avatars must remain outside the repository and release archives.

## Local development

1. Read the current terms and obtain **Cubism SDK for Native 5 R5** directly
   from [Live2D](https://www.live2d.com/en/sdk/download/native/). The qualified
   adapter checks for Core version 6.0.1. Do not substitute the Web SDK.
2. Extract the SDK to a stable directory outside the Sight checkout.
3. Configure an explicit local build; retain any other local effect flags
   required by your installation:

```sh
cmake -S . -B build-local-live2d -DCMAKE_BUILD_TYPE=Release \
  -DSIGHT_CUBISM_SDK_ROOT=/path/to/CubismSdkForNative-5-r.5
cmake --build build-local-live2d --parallel 2
ctest --test-dir build-local-live2d -R 'sight_.*live2d.*' --output-on-failure
```

The optional adapter compiles against the SDK header and dynamically loads
Core from the SDK path supplied at build time. Keep that directory available;
moving it requires reconfiguring/rebuilding. The SDK is not an installation
payload. The local SDK's Haru sample is used by the optional render regression,
never copied to Sight. See the [native SDK documentation](https://docs.live2d.com/en/cubism-sdk-manual/cubism-sdk-for-native/).

Use only your own or appropriately licensed model packages. In OBS the
optional source is **LLL.sight Live2D Avatar (Native)**. Select a local
model3.json and the desired face-tracking channel. Not every rig or advanced
Cubism compositing mode is supported. Missing/incompatible packages are
rejected rather than simulated.

## Publication boundary

- Sight's own integration source may be retained separately from the SDK,
  subject to the pending source-origin and project-license review.
- Standard preview builds leave SIGHT_CUBISM_SDK_ROOT empty. No official
  SDK file, model or Cubism-enabled binary is approved for public release.
- Live2D's [SDK publication terms](https://www.live2d.com/en/sdk/license/)
  distinguish development from release. Expandable avatar applications
  require separate review and agreement, even when offered free of charge.
  Confirm the applicable requirements with Live2D before distributing an
  enabled build. User-supplied SDKs or open-source integration code do not
  automatically establish an exemption.
- Sight's default non-Cubism release preparation can proceed independently;
  the separate [release checklist](release-readiness.md) still applies.

No automatic SDK acquisition, consent on the user's behalf, or claim of
publication approval is made.
