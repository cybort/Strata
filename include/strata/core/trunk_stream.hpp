// include/strata/core/trunk_stream.hpp - stream dense blk.* layers so VRAM is a dial, not a floor.
//
// Ported policy from kimi-k3-in-c's k3_trunk: pin a PREFIX of layers, cycle the rest through a small ring.
// Async prefetch runs only when the ring has >= 2 slots (one slot + async overwrites the live layer).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

class WeightTable;

struct TrunkTensor {
    std::string name;
    uint64_t src_off = 0;     ///< in dense.bin
    uint64_t src_bytes = 0;
    uint64_t slot_off = 0;    ///< offset within the layer slot (engine form)
    uint64_t dst_bytes = 0;   ///< engine-form size
    int kind = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0;
    int act_kind = 0;
    int64_t ne0 = 0, ne1 = 0;
    bool raw16 = false;
};

struct TrunkLayer {
    uint64_t nbytes = 0;   ///< sum of dst_bytes, aligned
    std::vector<TrunkTensor> tensors;
};

class TrunkStreamer {
public:
    TrunkStreamer() = default;
    ~TrunkStreamer();

    TrunkStreamer(const TrunkStreamer&) = delete;
    TrunkStreamer& operator=(const TrunkStreamer&) = delete;

    /// Parse pack index for blk.<L>.* rows (file dense.bin), size pin+ring from budget_bytes.
    /// When host_only is true, skips cudaMalloc (unit tests).
    bool open(const std::string& pack_dir, uint64_t budget_bytes, std::string& err, bool host_only = false);

    /// Names that must be skipped by WeightTable::load when streaming (all streamed blk tensors).
    void streamed_names(std::vector<std::string>& out) const;

    int n_layers() const { return n_layers_; }
    int npin() const { return npin_; }
    int nslot() const { return nslot_; }
    uint64_t slot_bytes() const { return slot_bytes_; }
    bool enabled() const { return n_layers_ > 0 && (npin_ + nslot_) > 0; }

    /// Make layer L resident on device (or host when host_only) and rebind WeightTable pointers.
    bool bind_layer(int L, WeightTable& wt, std::string& err);

    /// Async read of layer L into a free ring slot when nslot >= 2; no-op otherwise.
    void prefetch_layer(int L);

    void report() const;

    /// Pure budget helper for tests: given per-layer sizes and budget, compute npin and nslot (1 or 2).
    static void plan_budget(const std::vector<uint64_t>& layer_bytes, uint64_t budget_bytes, int& npin, int& nslot,
                            uint64_t& slot_bytes);

private:
    void io_loop();
    bool read_layer_into(int L, uint8_t* host_dst, std::string& err);
    bool ensure_resident(int L, std::string& err);
    uint8_t* host_slot_ptr(int slot_index);
    void* device_slot_ptr(int slot_index);

    std::string pack_dir_;
    std::FILE* dense_ = nullptr;
    bool host_only_ = false;

    int n_layers_ = 0;
    std::vector<TrunkLayer> layers_;

    int npin_ = 0;
    int nslot_ = 0;
    uint64_t slot_bytes_ = 0;
    uint64_t align_ = 256;

    std::vector<uint8_t*> pin_host_;
    std::vector<void*> pin_dev_;
    uint8_t* ring_host_ = nullptr;
    void* ring_dev_ = nullptr;

    std::vector<int> layer_of_slot_;   ///< ring: which layer occupies each ring slot
    std::vector<int> slot_of_layer_;   ///< -1 absent; 0..npin-1 pin; npin.. pin+ring

    int ring_next_ = 0;

    std::thread io_thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool io_enabled_ = false;
    int req_layer_ = -1;
    int req_slot_ = -1;
    bool req_busy_ = false;
    bool req_done_ = false;
    int req_result_ = 0;
    std::string io_err_;

    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
    uint64_t bytes_read_ = 0;
    double load_seconds_ = 0.0;
};

/// Session hook: when set, session_token binds/prefetches before each layer.
void trunk_bind_hook_set(TrunkStreamer* streamer, WeightTable* table);
void trunk_bind_hook_clear();
bool trunk_bind_before_layer(int L, int n_layers, std::string& err);

}  // namespace strata::core
