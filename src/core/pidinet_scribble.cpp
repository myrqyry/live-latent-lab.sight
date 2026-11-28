#include "pidinet_scribble.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>

namespace lll {
bool pidinet_scribble_edges(std::vector<uint8_t> &gray,
                           uint32_t w, uint32_t h)
{
    if (w < 8 || h < 8 || w > 1024 || h > 1024 ||
        gray.size() != size_t(w) * h)
        return false;
    // Port of the original ControlNet Aux PiDiNet scribble branch:
    // nms(x,127,sigma=3), GaussianBlur sigma=3, threshold >4.
    cv::Mat source(int(h), int(w), CV_8UC1, gray.data());
    cv::Mat float_edges, blurred;
    source.convertTo(float_edges, CV_32FC1);
    cv::GaussianBlur(float_edges, blurred, cv::Size(0, 0), 3.0);
    cv::Mat maxima = cv::Mat::zeros(blurred.size(), CV_32FC1);
    const std::array<std::array<uint8_t, 9>, 4> kernel_values = {{
        {{0, 0, 0, 1, 1, 1, 0, 0, 0}},
        {{0, 1, 0, 0, 1, 0, 0, 1, 0}},
        {{1, 0, 0, 0, 1, 0, 0, 0, 1}},
        {{0, 0, 1, 0, 1, 0, 1, 0, 0}},
    }};
    for (const auto &values : kernel_values) {
        cv::Mat kernel(3, 3, CV_8UC1,
                       const_cast<uint8_t *>(values.data()));
        cv::Mat dilated, keep;
        cv::dilate(blurred, dilated, kernel);
        cv::compare(dilated, blurred, keep, cv::CMP_EQ);
        blurred.copyTo(maxima, keep);
    }
    cv::Mat maxima_float, maxima_binary, softened, final;
    cv::threshold(maxima, maxima_float, 127.0, 255.0, cv::THRESH_BINARY);
    maxima_float.convertTo(maxima_binary, CV_8U);
    cv::GaussianBlur(maxima_binary, softened, cv::Size(0, 0), 3.0);
    cv::threshold(softened, final, 4.0, 255.0, cv::THRESH_BINARY);
    if (!final.isContinuous() || final.total() != gray.size())
        return false;
    std::copy(final.datastart, final.datastart + gray.size(), gray.begin());
    return true;
}
} // namespace lll
