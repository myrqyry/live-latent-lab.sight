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
// No verified Windows/macOS/ARM MediaPipe ABI exists yet. File length alone
// must not select a Linux ABI on another host.
#if defined(__linux__) && (defined(__x86_64__) || defined(__amd64__))
inline constexpr bool tasks_runtime_host_supported = true;
#else
inline constexpr bool tasks_runtime_host_supported = false;
#endif
inline constexpr TasksRuntimeAbi tasks_runtime_for_host(
    std::uintmax_t bytes, bool supported = tasks_runtime_host_supported)
{
    return supported ? tasks_runtime_abi(bytes) : TasksRuntimeAbi::Unknown;
}
} // namespace sight
