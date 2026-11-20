#pragma once

#include <vector>
#include <cstdint>

struct FastFace {
    float score;
    int x;
    int y;
    int w;
    int h;
    // Landmarks (5 points * 2 coords)
    int landmarks[10];
};

class __attribute__((visibility("default"))) FastFaceDetector {
public:
    FastFaceDetector();
    ~FastFaceDetector();

    /**
     * @brief Detect faces in an image.
     * 
     * @param rgb_data Pointer to BGR image data (despite legacy parameter name).
     * @param width Image width.
     * @param height Image height.
     * @param stride Image stride (bytes per row).
     * @param inference_ok Optional output: true when native inference returned
     *                     a valid result, even if zero faces were detected.
     * @return std::vector<FastFace> List of detected faces.
     */
    std::vector<FastFace> detect(const uint8_t* rgb_data, int width, int height, int stride,
                                 bool* inference_ok = nullptr);

private:
    // Buffer for detection results, required by libfacedetection
    // Must be 0x20000 bytes (128KB)
    std::vector<unsigned char> result_buffer;
};
