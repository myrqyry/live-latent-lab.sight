"""Native Sight caller for the pinned shared OBS SDK/packaging engine."""
import plistlib
import shutil
import sys
from pathlib import Path
import native_caller as caller
import native_sdk as sdk

MODULE = 'live-latent-lab-sight'
JSON_VERSION = '3.11.3'
JSON_FILES = {'single_include/nlohmann/json.hpp': '9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6',
              'LICENSE.MIT': '86b998c792894ccb911a1cb7994f7a9652894e7a094c0b5e45be2f553f45cf14'}

def configure(root, work, target, version):
    ort = work / 'onnxruntime'
    sdk.run(sys.executable, root / 'tools/acquire-native-ort.py', target, ort)
    json_root = work / 'json'
    for name, digest in JSON_FILES.items():
        path = json_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sdk.download(f'https://raw.githubusercontent.com/nlohmann/json/v{JSON_VERSION}/{name}', path, digest)
    return ['-DSIGHT_NATIVE_BASELINE=ON', f'-DSIGHT_ORT_ROOT={ort.as_posix()}',
            f'-DSIGHT_JSON_INCLUDE={(json_root / "single_include").as_posix()}',
            f'-DSIGHT_JSON_NOTICE={(json_root / "LICENSE.MIT").as_posix()}',
            f'-DSIGHT_NATIVE_OBS_VERSION={version}']


def require_native_report(path):
    import xml.etree.ElementTree as E
    cases = E.parse(path).findall('.//testcase')
    expected = {'sight_native_ort_utf8', 'sight_tasks_platform_policy', 'sight_native_vrm_asset', 'sight_native_install_guard'}
    if len(cases) != len(expected) or {c.get('name') for c in cases} != expected or any(
        c.get('status') not in (None, 'run') or any(c.find(k) is not None for k in ('failure', 'error', 'skipped')) for c in cases):
        raise ValueError('Native smoke report incomplete or unsuccessful')
    print('Verified native tests: ONNX inference, MediaPipe host boundary, VRM asset, installation guard')


def stage(root, installed, stage_root, target, version):
    work = installed.parent
    report = work / 'native.xml'
    sdk.run('ctest', '--test-dir', work / 'build', '-C', 'RelWithDebInfo', '--output-on-failure', '--no-tests=error', '--output-junit', report)
    require_native_report(report)
    binary, data, notice = caller.layout(stage_root, MODULE, target, version)
    ort = work / 'onnxruntime'
    runtime = {'onnxruntime.dll': 'onnxruntime.dll', 'onnxruntime_providers_shared.dll': 'onnxruntime_providers_shared.dll'} if target == 'windows-x64' else {'libonnxruntime.1.17.1.dylib': 'libonnxruntime.dylib'}
    module = installed / 'bin' / (MODULE + ('.dll' if target == 'windows-x64' else '.so'))
    if target.startswith('macos-') and not module.is_file():
        module = installed / 'bin' / (MODULE + '.dylib')
    inventory = {'licenses/sight/LICENSE': root / 'LICENSE', 'licenses/sight/LICENSING.md': root / 'LICENSING.md',
                 'licenses/onnxruntime/LICENSE': ort / 'LICENSE', 'licenses/onnxruntime/ThirdPartyNotices.txt': ort / 'ThirdPartyNotices.txt',
                 'licenses/libfacedetection/LICENSE': root / 'src/3rdparty/libfacedetection/LICENSE',
                 'licenses/tinygltf/LICENSE': root / 'src/vrm/vendor/tinygltf-LICENSE', 'licenses/stb/LICENSE': root / 'src/vrm/vendor/stb-LICENSE',
                 'licenses/nlohmann-json/LICENSE': work / 'json/LICENSE.MIT'}
    for name in ('locale/en-US.ini','presets/anime.json','sight-depth-camera.effect','sight-privacy-conceal.effect','sight-spatial-motion.effect','sight-subject-matte.effect','sight-text-obfuscation.effect','sight-tracking-overlay.effect'):
        inventory[name] = root / 'data' / name
    paths = {module} | {installed / 'bin' / n for n in runtime} | {installed / 'data' / n for n in inventory}
    entries = list(installed.rglob('*'))
    if any(p.is_symlink() for p in entries) or {p for p in entries if p.is_file()} != paths or not module.stat().st_size:
        raise ValueError('Unexpected native install inventory')
    for name, source in inventory.items():
        if sdk.digest(installed / 'data' / name) != sdk.digest(source):
            raise ValueError(f'Native notice or asset changed: {name}')
    for name, source in runtime.items():
        if sdk.digest(installed / 'bin' / name) != sdk.digest(ort / 'lib' / source):
            raise ValueError(f'Native runtime changed: {name}')
    binary.parent.mkdir(parents=True)
    shutil.copyfile(module, binary)
    for name in runtime:
        shutil.copyfile(installed / 'bin' / name, binary.parent / name)
    shutil.copytree(installed / 'data', data)
    if target.startswith('macos-'):
        dylib = binary.parent / 'libonnxruntime.1.17.1.dylib'
        sdk.run(*sdk.mac_arch_command(dylib, target))
        sdk.run('codesign', '--force', '--sign', '-', dylib)
        sdk.run('codesign', '--verify', '--strict', dylib)
        (binary.parent.parent / 'Info.plist').write_bytes(plistlib.dumps({'CFBundleExecutable': MODULE, 'CFBundleIdentifier': 'io.github.myrqyry.live-latent-lab-sight', 'CFBundleName': MODULE, 'CFBundlePackageType': 'BNDL', 'CFBundleVersion': '0.3.0'}))
    return binary
