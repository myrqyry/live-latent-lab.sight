#include "core/litert_keypoint_estimator.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

int main(int argc, char **argv)
{
    std::vector<lll::LiteRtKeypoint> keypoints;
    assert(!lll::LiteRtKeypointEstimator::decode_simcc(
        {}, {}, 17, 384, 512, 0.0f, 0.0f, 192.0f, 256.0f, 192, 256, keypoints));

    std::vector<float> x(2 * 4, -10.0f);
    std::vector<float> y(2 * 4, -10.0f);
    x[1] = 10.0f;
    y[2] = 10.0f;
    x[4 + 3] = 10.0f;
    y[4 + 0] = 10.0f;
    assert(lll::LiteRtKeypointEstimator::decode_simcc(
        x, y, 2, 4, 4, 10.0f, 20.0f, 100.0f, 200.0f, 100, 200, keypoints));
    assert(keypoints.size() == 2);
    assert(std::fabs(keypoints[0].x - 10.5f) < 0.001f);
    assert(std::fabs(keypoints[0].y - 21.0f) < 0.001f);
    assert(keypoints[0].confidence > 0.99f);
    assert(std::fabs(keypoints[1].x - 11.5f) < 0.001f);
    assert(std::fabs(keypoints[1].y - 20.0f) < 0.001f);

    if (argc > 1) {
        lll::LiteRtKeypointModel model = lll::LiteRtKeypointModel::Pose17;
        int width = 192;
        int height = 256;
        size_t expected_keypoints = 17;
        if (argc > 2 && std::strcmp(argv[2], "face") == 0) {
            model = lll::LiteRtKeypointModel::Face98;
            width = height = 256;
            expected_keypoints = 98;
        } else if (argc > 2 && std::strcmp(argv[2], "hand") == 0) {
            model = lll::LiteRtKeypointModel::Hand21;
            width = height = 256;
            expected_keypoints = 21;
        } else if (argc > 2 && std::strcmp(argv[2], "wholebody") == 0) {
            model = lll::LiteRtKeypointModel::WholeBody133;
            expected_keypoints = 133;
        }

        assert(lll::LiteRtRuntime::instance().initialize());
        lll::LiteRtKeypointEstimator estimator;
        assert(estimator.load(argv[1], model));
        std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3, 128);
        assert(estimator.estimate(rgb.data(), width, height, keypoints, lll::LiteRtDevice::Gpu));
        assert(keypoints.size() == expected_keypoints);
    }
    return 0;
}
