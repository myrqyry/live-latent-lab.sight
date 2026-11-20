#!/usr/bin/env python3
"""Release-safety regression tests: no assumptions about user's OBS install."""
import pathlib
import re
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
CMAKE = ROOT / "cmake" / "validated_effects.cmake"
PRIVACY = ROOT / "tests" / "verify-privacy-v2.sh"


class ReleaseSafetyTests(unittest.TestCase):
    def test_privacy_runner_uses_supplied_image_module_and_build_runtime(self):
        source = PRIVACY.read_text()
        self.assertNotIn('image_plugin=/usr/local/lib/obs-plugins/image-source.so', source)
        self.assertIn('SIGHT_SMOKE_IMAGE_PLUGIN', source)
        self.assertNotIn('$HOME/.config/obs-studio/plugins/live-latent-lab-sight', source)
        self.assertIn('$(dirname "$module")', source)

    def test_privacy_ctest_passes_discovered_module(self):
        source = CMAKE.read_text()
        privacy = source.index("add_test(NAME sight_privacy_v2_obs_integration")
        self.assertLess(source.index("find_file(SIGHT_IMAGE_SOURCE_MODULE"), privacy)
        match = re.search(
            r"set_tests_properties\(sight_privacy_v2_obs_integration\s+"
            r"PROPERTIES(?:(?!\)\s*\n).)*"
            r"SIGHT_SMOKE_IMAGE_PLUGIN=\$\{SIGHT_IMAGE_SOURCE_MODULE\}",
            source[privacy:],
            re.DOTALL,
        )
        self.assertIsNotNone(match)

    def test_explicit_system_subdirectory_is_also_blocked(self):
        # A public user might explicitly choose /usr/local/obs; it must not
        # bypass the safety guard simply because it differs from /usr/local.
        with tempfile.TemporaryDirectory(prefix="sight-system-prefix-") as tmp:
            build = pathlib.Path(tmp) / "build"
            configured = subprocess.run(
                ["cmake", "-S", str(ROOT), "-B", str(build),
                 "-DCMAKE_INSTALL_PREFIX=/usr/local/sight"],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(configured.returncode, 0, configured.stderr[-2000:])
            attempted = subprocess.run(
                ["cmake", "--install", str(build)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(attempted.returncode, 0)
            self.assertIn(
                "Sight refuses system-wide installation",
                attempted.stdout + attempted.stderr,
            )

    def test_default_prefix_is_guarded_at_install_time(self):
        # Build not required: refusal must happen before any install writes.
        with tempfile.TemporaryDirectory(prefix="sight-install-guard-") as tmp:
            build = pathlib.Path(tmp) / "build"
            configured = subprocess.run(
                ["cmake", "-S", str(ROOT), "-B", str(build)],
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(configured.returncode, 0, configured.stderr[-2000:])
            attempted = subprocess.run(
                ["cmake", "--install", str(build)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(attempted.returncode, 0)
            self.assertIn(
                "Sight refuses system-wide installation",
                attempted.stdout + attempted.stderr,
            )


if __name__ == "__main__":
    unittest.main()
