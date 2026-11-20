#include "fast-face-detector.hpp"
#include "3rdparty/libfacedetection/facedetectcnn.h"
#include "common/plugin-support.h"
#include <algorithm>
#include <mutex>

// Standard size required by libfacedetection
static const int RESULT_BUFFER_SIZE = 0x20000;

FastFaceDetector::FastFaceDetector() {
    blog(LOG_INFO, "[FastFaceDetector] Initializing detector instance");
    result_buffer.resize(RESULT_BUFFER_SIZE);
}

FastFaceDetector::~FastFaceDetector() = default;

std::vector<FastFace> FastFaceDetector::detect(const uint8_t* rgb_data, int width, int height, int stride,
                                                bool* inference_ok) {
    if(inference_ok)*inference_ok=false;
    std::vector<FastFace> faces;

    if (result_buffer.empty() || !rgb_data || width <= 0 || height <= 0 || stride < width * 3) {
        return faces;
    }

    // The API expects non-const pointer but doesn't modify input image
    unsigned char* input_image = const_cast<unsigned char*>(rgb_data);

    // The bundled CNN consumes packed BGR bytes with stride >= width*3.
    // The CNN's model initialization/state is process-wide; separate Sight
    // filter workers must serialize inference through this native library.
    static std::mutex cnn_mutex;
    std::lock_guard<std::mutex> lock(cnn_mutex);
    int* pResults = facedetect_cnn(result_buffer.data(), input_image, width, height, stride);

    if (pResults) {
        // libfacedetection copies 16 *short* values per detection at byte 4.
        // The older wrapper incorrectly cast those packed records to
        // FaceRect (float+int), producing garbage rectangles.
        const int count=*pResults;
        if(count<0||count>1024)return {};
        if(inference_ok)*inference_ok=true;
        for(int i=0;i<count;++i) {
            const short *p=reinterpret_cast<const short*>(
                result_buffer.data()+4)+16*i;
            if(p[0]<70||p[3]<=0||p[4]<=0||
               p[1]<0||p[2]<0||p[1]+p[3]>width||p[2]+p[4]>height)
                continue;
            FastFace face{};
            face.score=p[0]/100.f;
            face.x=p[1];face.y=p[2];face.w=p[3];face.h=p[4];
            for(int k=0;k<10;++k)face.landmarks[k]=p[5+k];
            faces.push_back(face);
        }
    }

    return faces;
}
