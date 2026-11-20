#include "tracking_signal_bus.hpp"
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace sight {
namespace {
using Clock = std::chrono::steady_clock;
struct Bus {
    std::mutex mutex;
    std::unordered_map<std::string, TrackingSignal> channels;
    std::atomic<uint64_t> next_publisher{1};
    uint64_t next_sequence = 1;
};
Bus& bus() {
    // Keep this per-process bus alive until the OS tears down the process.
    // OBS/Qt can begin static destruction while the MediaPipe worker is still
    // publishing (notably if X11 disconnects during shutdown). A normal
    // function-local static Bus would destruct its unordered_map first and
    // cause a real use-after-free. Tests reproduce this under AddressSanitizer.
    static Bus* const shared = new Bus;
    return *shared;
}
constexpr size_t MaxChannels = 64;
constexpr size_t MaxChannelLength = 96;
}

uint64_t TrackingSignalBus::new_publisher_id()
{
    return bus().next_publisher.fetch_add(1);
}

bool TrackingSignalBus::publish(const std::string& channel, uint64_t publisher,
                                TrackingKind kind, const TrackingData& data,
                                uint32_t width, uint32_t height)
{
    if (channel.empty() || channel.size() > MaxChannelLength || publisher == 0 ||
        width == 0 || height == 0 ||
        data.landmarks.size() > 500 || data.blendshapes.size() > 64)
        return false;
    Bus& state = bus();
    const auto now = Clock::now();
    std::lock_guard lock(state.mutex);
    auto it = state.channels.find(channel);
    if (it == state.channels.end() && state.channels.size() >= MaxChannels)
        return false;
    // A channel has exactly one owner; two producer filters cannot silently
    // overwrite each other's results. Reassign the duplicate filter's channel.
    if (it != state.channels.end() && it->second.publisher != publisher)
        return false;
    TrackingSignal snapshot;
    snapshot.kind = kind;
    snapshot.data = data;
    snapshot.source_width = width;
    snapshot.source_height = height;
    snapshot.publisher = publisher;
    snapshot.sequence = state.next_sequence++;
    snapshot.updated = now;
    state.channels.insert_or_assign(channel, std::move(snapshot));
    return true;
}

bool TrackingSignalBus::publish_group(
    const std::array<std::string,3>& channels,
    uint64_t publisher, const std::array<TrackingData,3>& data,
    uint32_t width, uint32_t height)
{
    if (publisher == 0 || width == 0 || height == 0 ||
        channels[0] == channels[1] || channels[0] == channels[2] ||
        channels[1] == channels[2])
        return false;
    for(size_t i=0;i<3;++i) {
        if (channels[i].empty() || channels[i].size() > MaxChannelLength ||
            data[i].landmarks.size() > 500 || data[i].blendshapes.size() > 64)
            return false;
    }
    Bus& state=bus();
    std::lock_guard lock(state.mutex);
    size_t new_channels=0;
    for(const auto& channel:channels) {
        const auto it=state.channels.find(channel);
        if(it==state.channels.end()) ++new_channels;
        else if(it->second.publisher!=publisher) return false;
    }
    if(state.channels.size()+new_channels>MaxChannels)
        return false;
    const auto now=Clock::now();
    constexpr std::array<TrackingKind,3> kinds{
        TrackingKind::Face, TrackingKind::Body, TrackingKind::Hand
    };
    for(size_t i=0;i<3;++i) {
        TrackingSignal snapshot;
        snapshot.kind=kinds[i];
        snapshot.data=data[i];
        snapshot.source_width=width;
        snapshot.source_height=height;
        snapshot.publisher=publisher;
        snapshot.sequence=state.next_sequence++;
        snapshot.updated=now;
        state.channels.insert_or_assign(channels[i],std::move(snapshot));
    }
    return true;
}

bool TrackingSignalBus::read(const std::string& channel,
                             std::chrono::milliseconds max_age, TrackingSignal& out)
{
    if (channel.empty() || max_age.count() <= 0)
        return false;
    Bus& state = bus();
    std::lock_guard lock(state.mutex);
    auto it = state.channels.find(channel);
    if (it == state.channels.end() || Clock::now() - it->second.updated > max_age)
        return false;
    out = it->second;
    return true;
}

void TrackingSignalBus::withdraw(const std::string& channel, uint64_t publisher)
{
    Bus& state = bus();
    std::lock_guard lock(state.mutex);
    auto it = state.channels.find(channel);
    if (it != state.channels.end() && it->second.publisher == publisher)
        state.channels.erase(it);
}

void TrackingSignalBus::clear_for_tests()
{
    Bus& state = bus();
    std::lock_guard lock(state.mutex);
    state.channels.clear();
    state.next_sequence = 1;
}
}
