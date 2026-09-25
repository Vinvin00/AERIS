#pragma once
#include <cmath>
#include <cstdint>

namespace aeris {
struct FrameStamp {
    std::uint64_t frame_index;
    double timestamp_s;
    [[nodiscard]] bool valid() const noexcept {
        return std::isfinite(timestamp_s) && timestamp_s >= 0.0;
    }
};
}  // namespace aeris
