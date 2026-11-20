#include "core/litert_runtime.hpp"

#include <cassert>
#include <filesystem>

int main()
{
    auto &runtime = lll::LiteRtRuntime::instance();
    runtime.shutdown();

    std::vector<float> output;
    assert(!runtime.run("missing-model.tflite", lll::LiteRtDevice::Gpu, {},
                        output));
    assert(runtime.last_error().find("not initialized") != std::string::npos);

    assert(runtime.initialize());
    assert(runtime.ready());
    assert(!runtime.run("missing-model.tflite", lll::LiteRtDevice::Gpu, {},
                        output));
    assert(runtime.last_error().find("does not exist") != std::string::npos);

    runtime.shutdown();
    assert(!runtime.ready());
    return 0;
}
