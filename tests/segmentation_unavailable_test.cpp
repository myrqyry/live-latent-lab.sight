#include "core/segmentation_processor.hpp"
#include <array>
#include <iostream>

int main() {
    lll::SegmentationProcessor processor;
    std::array<uint8_t,12> input{};
    std::array<uint8_t,4> output{17,31,53,79};
    const auto original=output;
    const bool unavailable=!processor.is_ready() &&
        !processor.load("unused.onnx",lll::SegModelType::U2Net,true) &&
        !processor.is_ready() &&
        !processor.process(input.data(),2,2,output.data()) &&
        output==original;
    std::cout<<"no_backend_fails_closed="<<unavailable<<"\n";
    return unavailable?0:1;
}
