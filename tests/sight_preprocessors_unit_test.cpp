#include "preprocessors_processor.hpp"
#include <algorithm>
// Keep assertions enabled even in CMake Release/NDEBUG test builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace preprocessors;

int main()
{
    const auto &modes = catalog();
    std::set<std::string> ids;
    size_t native = 0, model = 0;
    for (const auto &mode : modes) {
        assert(ids.insert(mode.id).second && "preprocessor IDs must be unique");
        assert(mode.label && *mode.label && mode.category && *mode.category);
        if (mode.backend == Backend::Native) ++native;
        else ++model;
    }
    assert(native >= 10 && model >= 20);
    assert(find_mode("standard_lineart") && is_native("standard_lineart"));
    assert(find_mode("anime_lineart") && !is_native("anime_lineart"));
    assert(find_mode("depth_anything_v2"));
    assert(find_mode("dwpose"));

    constexpr int w = 128, h = 96;
    std::vector<uint8_t> input(w * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto *p = &input[(y * w + x) * 4];
            const bool block = x >= 32 && x < 92 && y >= 20 && y < 72;
            p[0] = block ? 245 : 20;
            p[1] = block ? 30 : 190;
            p[2] = block ? 55 : 60;
            p[3] = (x % 3) ? 255 : 128;
        }
    Settings s;
    auto has_change = [&](std::string_view mode) {
        std::vector<uint8_t> output;
        const bool ok = process_rgba(input.data(), w, h, w * 4, mode, s, output);
        if (!ok || output.size() != input.size()) return false;
        for (size_t i = 3; i < input.size(); i += 4)
            if (output[i] != input[i]) return false; // preserve alpha
        return output != input;
    };
    for (const auto &mode : {"canny", "binary", "standard_lineart",
                             "pyracanny", "scribble_canny", "scribble_xdog",
                             "luminance", "intensity", "color_palette",
                             "tile_shuffle"}) {
        assert(has_change(mode) && "native mode must change actual pixels");
    }

    // Lineart must not collapse to a flat map on colorful high-contrast
    // boundaries (a plain "output != input" test misses this).
    std::vector<uint8_t> ink;
    assert(process_rgba(input.data(), w, h, w * 4, "standard_lineart", s, ink));
    uint8_t darkest = 255, brightest = 0;
    for (size_t i = 0; i < ink.size(); i += 4) {
        darkest = std::min(darkest, ink[i]);
        brightest = std::max(brightest, ink[i]);
    }
    assert(brightest > darkest + 24 && "lineart should retain visible structure");

    // Two different recolor choices must not be cosmetic labels only.
    std::vector<uint8_t> luma, intensity;
    assert(process_rgba(input.data(), w, h, w * 4, "luminance", s, luma));
    assert(process_rgba(input.data(), w, h, w * 4, "intensity", s, intensity));
    assert(luma != intensity);
    assert(luma[0] == luma[1] && luma[1] == luma[2]);

    // A missing neural model is never simulated with an unrelated edge map.
    const std::vector<uint8_t> sentinel{5, 4, 3};
    auto output = sentinel;
    assert(!process_rgba(input.data(), w, h, w * 4, "anime_lineart", s, output));
    assert(output == sentinel);
    assert(!process_rgba(input.data(), 0, h, w * 4, "canny", s, output));
    assert(output == sentinel);
    assert(!process_rgba(input.data(), w, h, w * 4 - 1, "canny", s, output));
    assert(output == sentinel);
    assert(!process_rgba(input.data(), w, h, w * 4, "not-real", s, output));
    assert(output == sentinel);

    std::cout << "PASS: " << native << " native modes and " << model
              << " explicit model-required modes in unified catalog\n";
}
