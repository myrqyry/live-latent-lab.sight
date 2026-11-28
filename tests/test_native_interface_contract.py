"""Regression tests for the native Sight preprocessor ABI contract.

These tests deliberately mutate source in memory: they must fail closed when
a Rust FFI type, C header type, OBS procedure declaration or callback drifts.
No native build, OBS session, network, or model files are needed.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools/check_native_interface_contract.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("sight_contract_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot():
    return {
        "manifest": json.loads((ROOT / "docs/native-preprocessor-abi.json").read_text()),
        "c_header": (ROOT / "src/core/rust_frame_core.h").read_text(),
        "rust": (ROOT / "rust/sight-frame-core/src/lib.rs").read_text(),
        "obs": (ROOT / "src/preprocessors-filter.cpp").read_text(),
        "docs": (ROOT / "docs/native-preprocessor-interface.md").read_text(),
    }


class NativeContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_checker()
        self.content = snapshot()

    def audit(self):
        self.module.audit(**self.content)

    def expect_failure(self, needle):
        with self.assertRaises(self.module.ContractError) as caught:
            self.audit()
        self.assertIn(needle, str(caught.exception))

    def test_current_native_contract(self):
        self.audit()

    def test_detects_c_parameter_drift(self):
        self.content["c_header"] = self.content["c_header"].replace(
            "size_t rgb_len", "uint32_t rgb_len", 1
        )
        self.expect_failure("sight_normalize_rgb_to_chw_f32_v1")

    def test_detects_rust_parameter_drift(self):
        source = self.content["rust"]
        begin = source.index('pub unsafe extern "C" fn sight_rgb8_to_chw_f32_v2')
        source = source[:begin] + source[begin:].replace("scale: f32", "scale: u32", 1)
        self.content["rust"] = source
        self.expect_failure("sight_rgb8_to_chw_f32_v2")

    def test_detects_unlisted_rust_export(self):
        self.content["rust"] += '''
#[no_mangle]
pub unsafe extern "C" fn sight_surprise_v1(value: i32) -> i32 {
    value
}
'''
        self.expect_failure("sight_surprise_v1")

    def test_detects_obsolete_obs_procedure_signature(self):
        self.content["obs"] = self.content["obs"].replace(
            "void sight_preprocessor_pidinet_stats_v1(out int pidinet_completed)",
            "void sight_preprocessor_pidinet_stats_v1(out int changed)",
        )
        self.expect_failure("sight_preprocessor_pidinet_stats_v1")

    def test_detects_callback_output_drift(self):
        self.content["obs"] = self.content["obs"].replace(
            'calldata_set_int(args, "pidinet_completed"',
            'calldata_set_int(args, "wrong_count"',
        )
        self.expect_failure("pidinet_completed")

    def test_detects_hed_callback_field_drift(self):
        self.content["obs"] = self.content["obs"].replace(
            'calldata_set_int(args, "hed_completed"',
            'calldata_set_int(args, "some_other_counter"',
        )
        self.expect_failure("hed_completed")

    def test_detects_missing_public_documentation(self):
        self.content["docs"] = self.content["docs"].replace(
            "sight_teed_edges_to_gray_v1", "unlisted_abi_function"
        )
        self.expect_failure("sight_teed_edges_to_gray_v1")


if __name__ == "__main__":
    unittest.main()
