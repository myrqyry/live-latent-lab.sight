#include <litert/cc/litert_compiled_model.h>

int main()
{
    const auto accelerator = kLiteRtHwAcceleratorGpu;
    return accelerator == kLiteRtHwAcceleratorGpu ? 0 : 1;
}
