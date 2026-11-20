#include "core/holistic_runtime_discovery.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>

int main() {
    namespace fs=std::filesystem;
    const auto directory=fs::temp_directory_path() /
        ("sight-holistic-default-"+std::to_string(getpid()));
    fs::create_directories(directory);
    struct Cleanup {
        fs::path p;
        ~Cleanup(){std::error_code ec;fs::remove_all(p,ec);}
    } cleanup{directory};
    const auto module=directory/"live-latent-lab-sight.so";
    const auto modern=directory/"libmediapipe-1.2.3.so";
    const auto legacy=directory/"libmediapipe.so";

    // Fresh install without bundled runtime: never select the legacy ABI.
    assert(sight::holistic_runtime_default(module).empty());
    std::ofstream(module).close(); // create module-like file for tests
    std::ofstream(legacy).close();
    fs::resize_file(legacy,33886744); // sparse; no large data written
    assert(sight::holistic_runtime_default(module).empty());

    // An unverified or incomplete modern library must not be auto-selected.
    std::ofstream(modern).close();
    fs::resize_file(modern,1024);
    assert(sight::holistic_runtime_default(module).empty());
    fs::resize_file(modern,33886744);
    assert(sight::holistic_runtime_default(module).empty());

    assert(sight::tracking_runtime_default(module)==legacy.string());
    // Only the official 1.2.3 ABI-sized sibling is selected.
    fs::resize_file(modern,127908512); // sparse; do not actually run it
    assert(sight::holistic_runtime_default(module)==modern.string());
    assert(sight::tracking_runtime_default(module)==modern.string());

    // Do not follow symlinked libraries outside the plugin directory.
    fs::remove(modern);
    fs::create_symlink(legacy,modern);
    assert(sight::holistic_runtime_default(module).empty());

    // Absent/invalid module path must never turn into a system-wide guess.
    assert(sight::tracking_runtime_default(module)==legacy.string());
    assert(sight::tracking_runtime_default({}).empty());
    assert(sight::holistic_runtime_default({}).empty());
    assert(sight::holistic_runtime_default(directory).empty());

    std::cout << "HOLISTIC_RUNTIME_DISCOVERY_PASS=YES\n";
}
