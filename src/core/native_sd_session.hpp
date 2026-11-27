#pragma once
#include <mutex>
namespace sight::sd {
// Serialize multi-GB native diffusion contexts across input sources and filters.
inline std::mutex session_slot;
}
