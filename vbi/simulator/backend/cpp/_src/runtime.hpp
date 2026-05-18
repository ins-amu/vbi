// VBI C++ backend runtime — shared helpers.
// Layout convention: arrays are [sv/cvar/param * n_nodes + node]  (row-major, C order).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vbi::runtime {

// ---------------------------------------------------------------------------
// RingBuffer — delayed coupling history
// Layout: buf[slot * (n_cvar * n_nodes) + cv * n_nodes + node]
// ---------------------------------------------------------------------------

class RingBuffer {
public:
    RingBuffer(std::size_t horizon, std::size_t n_cvar, std::size_t n_nodes)
        : horizon_(std::max<std::size_t>(horizon, 1)),
          n_cvar_(n_cvar),
          n_nodes_(n_nodes),
          frame_(n_cvar * n_nodes),
          next_(0),
          filled_(0),
          data_(horizon_ * n_cvar * n_nodes, 0.0) {}

    // Fill all slots with a constant cvar state (n_cvar * n_nodes).
    void initialize(const double* cvar_state) {
        for (std::size_t s = 0; s < horizon_; ++s)
            std::copy(cvar_state, cvar_state + frame_,
                      data_.data() + s * frame_);
        filled_ = horizon_;
    }

    // Write one frame of cvar state.
    void write(const double* cvar_state) {
        std::copy(cvar_state, cvar_state + frame_,
                  data_.data() + next_ * frame_);
        next_ = (next_ + 1) % horizon_;
        filled_ = std::min(filled_ + 1, horizon_);
    }

    // Read a single cvar value from `delay_steps` steps ago.
    // delay=0 → most recently written frame.
    double read(std::size_t cv, std::size_t node, std::size_t delay_steps) const {
        // latest written slot is (next_ + horizon_ - 1) % horizon_
        const std::size_t latest = (next_ + horizon_ - 1) % horizon_;
        const std::size_t slot   = (latest + horizon_ - delay_steps) % horizon_;
        return data_[slot * frame_ + cv * n_nodes_ + node];
    }

    std::size_t horizon() const { return horizon_; }

private:
    std::size_t horizon_, n_cvar_, n_nodes_, frame_, next_, filled_;
    std::vector<double> data_;
};

} // namespace vbi::runtime
