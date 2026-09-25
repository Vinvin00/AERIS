#include "aeris/frame_stamp.hpp"
#include <limits>

int main() {
    if (!aeris::FrameStamp{0, 0.0}.valid()) return 1;
    if (aeris::FrameStamp{1, -1.0}.valid()) return 2;
    if (aeris::FrameStamp{2, std::numeric_limits<double>::quiet_NaN()}.valid()) return 3;
    if (aeris::FrameStamp{3, std::numeric_limits<double>::infinity()}.valid()) return 4;
    return 0;
}
