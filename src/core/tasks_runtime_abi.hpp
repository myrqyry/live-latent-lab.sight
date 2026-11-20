#pragma once
#include <cstdint>

namespace sight {
// Only these two exact Linux x86_64 C ABI layouts have been tested. Never
// infer ABI compatibility from matching symbol names or a semver string.
enum class TasksRuntimeAbi { Unknown, Legacy, V123 };

inline constexpr TasksRuntimeAbi tasks_runtime_abi(std::uintmax_t bytes)
{
    switch(bytes) {
        case 33886744: return TasksRuntimeAbi::Legacy;
        case 127908512: return TasksRuntimeAbi::V123;
        default: return TasksRuntimeAbi::Unknown;
    }
}
} // namespace sight
