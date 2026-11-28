#!/usr/bin/env python3
"""Fail-closed stdlib check of Sight's native preprocessor ABI declarations."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path


class ContractError(ValueError):
    """Native interface drift or unsupported contract syntax."""


def normalize_c_type(c_type: str) -> str:
    c_type = re.sub(r"\s+", " ", c_type.strip())
    pointers = re.fullmatch(r"(const )?(uint8_t|uint32_t|int32_t|float)\s*\*", c_type)
    if pointers:
        primitive = {"uint8_t":"u8","uint32_t":"u32","int32_t":"i32","float":"f32"}[pointers.group(2)]
        return ("*const " if pointers.group(1) else "*mut ") + primitive
    return {"int32_t":"i32","uint32_t":"u32","size_t":"usize","float":"f32"}.get(c_type, "?"+c_type)


def c_exports(source: str) -> dict:
    pattern = re.compile(r"\b(int32_t)\s+(sight_\w+)\s*\(([^;]*?)\)\s*;", re.S)
    parsed = {}
    for result, name, args in pattern.findall(source):
        if name in parsed:
            raise ContractError("duplicate C ABI declaration: "+name)
        parameters = []
        for parameter in args.split(","):
            match = re.fullmatch(r"\s*(.+?\s*\*?\s*)([A-Za-z_]\w*)\s*", parameter)
            if not match:
                raise ContractError("unsupported C parameter: "+name+" "+parameter)
            parameters.append((match.group(2),normalize_c_type(match.group(1))))
        parsed[name] = {"return":normalize_c_type(result),"parameters":parameters}
    return parsed


def rust_exports(source: str) -> dict:
    pattern = re.compile(
        r'#\[no_mangle\]\s*pub\s+unsafe\s+extern\s+"C"\s+fn\s+'
        r'(sight_\w+)\s*\(([^)]*?)\)\s*->\s*(\w+)', re.S)
    parsed = {}
    for name, args, result in pattern.findall(source):
        if name in parsed:
            raise ContractError("duplicate Rust export: "+name)
        parameters = []
        for parameter in (item for item in args.split(",") if item.strip()):
            match = re.fullmatch(
                r"\s*([a-zA-Z_]\w*)\s*:\s*(\*const\s+\w+|\*mut\s+\w+|\w+)\s*",
                parameter)
            if not match:
                raise ContractError("unsupported Rust parameter: "+name+" "+parameter)
            parameters.append((match.group(1),re.sub(r"\s+"," ",match.group(2))))
        parsed[name] = {"return":result,"parameters":parameters}
    return parsed


def obs_procedures(source: str) -> dict:
    pattern = re.compile(
        r'proc_handler_add\s*\(\s*obs_source_get_proc_handler\s*\(\s*source\s*\)\s*,'
        r'\s*"([^"]+)"\s*,\s*([A-Za-z_]\w*)\s*,\s*f\.get\(\)\s*\)', re.S)
    parsed = {}
    for declaration, callback in pattern.findall(source):
        match = re.fullmatch(r"\s*void\s+(sight_preprocessor_\w+)\s*\(([^)]*)\)\s*",declaration)
        if not match:
            raise ContractError("invalid native OBS procedure: "+declaration)
        name, params = match.groups()
        if name in parsed:
            raise ContractError("duplicate native OBS procedure: "+name)
        declared = []
        for item in [p.strip() for p in params.split(",")]:
            out = re.fullmatch(r"out int ([A-Za-z_]\w*)",item)
            if not out:
                raise ContractError("unsupported procedure output: "+name+" "+item)
            declared.append(out.group(1))
        parsed[name] = {"callback":callback,"outputs":declared}
    return parsed


def audit(manifest: dict, c_header: str, rust: str, obs: str, docs: str) -> None:
    if manifest.get("schema_version") != 1:
        raise ContractError("unsupported native contract version")
    expected_ffi = manifest["rust_ffi"]
    expected_obs = manifest["obs_procedures"]
    c_actual = c_exports(c_header)
    rust_actual = rust_exports(rust)
    obs_actual = obs_procedures(obs)
    errors: list[str] = []
    for label, observed, expected in (
        ("C header",c_actual,expected_ffi),
        ("Rust library",rust_actual,expected_ffi),
        ("OBS procedures",obs_actual,expected_obs),
    ):
        for name in sorted(set(expected)-set(observed)):
            errors.append(f"{label} missing: {name}")
        for name in sorted(set(observed)-set(expected)):
            errors.append(f"{label} unreviewed: {name}")
    for name, entry in expected_ffi.items():
        expected_params = [(p[0],p[1]) for p in entry["parameters"]]
        for label, actual in (("C header",c_actual),("Rust library",rust_actual)):
            if name not in actual:
                continue
            current = actual[name]
            if current["return"] != entry["return"]:
                errors.append(f"{label} return mismatch: {name}")
            expected_types = [p[1] for p in expected_params]
            current_types = [p[1] for p in current["parameters"]]
            if expected_types != current_types:
                errors.append(f"{label} parameter mismatch: {name}; "
                              f"expected {expected_types}, got {current_types}")
            names = [p[0] for p in expected_params]
            if label == "Rust library":
                overrides = entry.get("rust_name_overrides",{})
                names = [overrides.get(p,p) for p in names]
            current_names = [p[0] for p in current["parameters"]]
            if names != current_names:
                errors.append(f"{label} parameter name drift: {name}; "
                              f"expected {names}, got {current_names}")
    for name, entry in expected_obs.items():
        if name not in obs_actual:
            continue
        actual = obs_actual[name]
        if actual != entry:
            errors.append(f"OBS procedure mismatch: {name}; "
                          f"expected {entry}, got {actual}")
        callback = entry["callback"]
        match = re.search(
            r"\bvoid\s+"+re.escape(callback)+
            r"\s*\(\s*void\s*\*\s*data\s*,\s*calldata_t\s*\*\s*args\s*\)"
            r"\s*\{(.*?)\n\}",obs,re.S)
        if not match:
            errors.append(f"missing or nonconforming OBS callback: {callback}")
            continue
        writes = re.findall(r'calldata_set_int\s*\(\s*args\s*,\s*"(\w+)"',match.group(1))
        if writes != entry["outputs"]:
            errors.append(f"callback output drift: {name}; "
                          f"expected {entry['outputs']}, got {writes}")
    for name in list(expected_ffi)+list(expected_obs):
        if not re.search(r"\b"+re.escape(name)+r"\b",docs):
            errors.append(f"native interface documentation missing: {name}")
    if errors:
        raise ContractError("\n".join(errors))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    content = {
        "manifest":json.loads((root/"docs/native-preprocessor-abi.json").read_text()),
        "c_header":(root/"src/core/rust_frame_core.h").read_text(),
        "rust":(root/"rust/sight-frame-core/src/lib.rs").read_text(),
        "obs":(root/"src/preprocessors-filter.cpp").read_text(),
        "docs":(root/"docs/native-preprocessor-interface.md").read_text(),
    }
    try:
        audit(**content)
    except ContractError as err:
        print("FAIL: Sight native interface contract drift\n"+str(err),file=sys.stderr)
        return 1
    print(f"PASS: Sight native ABI contract ({len(content['manifest']['rust_ffi'])} Rust exports, {len(content['manifest']['obs_procedures'])} OBS procedures)")
    return 0


if __name__=="__main__":
    sys.exit(main())
