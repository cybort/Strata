// src/core/trunk_stream.cpp - see trunk_stream.hpp
#include "strata/core/trunk_stream.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {
namespace {

TrunkStreamer* g_hook_streamer = nullptr;
WeightTable* g_hook_table = nullptr;

constexpr uint64_t kAlign = 256;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool parse_blk_layer(const char* name, int& layer) {
    // blk.<N>.
    if (std::strncmp(name, "blk.", 4) != 0) return false;
    char* end = nullptr;
    const long v = std::strtol(name + 4, &end, 10);
    if (end == name + 4 || *end != '.') return false;
    if (v < 0 || v > 100000) return false;
    layer = (int) v;
    return true;
}

bool read_at(std::FILE* f, uint64_t off, void* dst, size_t n, std::string& err, const char* what) {
#if defined(_WIN32)
    if (_fseeki64(f, (long long) off, SEEK_SET) != 0) {
        err = std::string(what) + ": seek failed";
        return false;
    }
#else
    if (fseeko(f, (off_t) off, SEEK_SET) != 0) {
        err = std::string(what) + ": seek failed";
        return false;
    }
#endif
    if (std::fread(dst, 1, n, f) != n) {
        err = std::string(what) + ": short read";
        return false;
    }
    return true;
}

enum class Conv : int { Copy = 0, Bf16High = 1, ToF16 = 2, WidenF16 = 3 };

bool convert_plane(Conv c, const uint8_t* src, uint64_t src_bytes, uint8_t* dst, uint64_t& dst_bytes, std::string& err) {
    switch (c) {
        case Conv::Copy:
            std::memcpy(dst, src, (size_t) src_bytes);
            dst_bytes = src_bytes;
            return true;
        case Conv::Bf16High: {
            if (src_bytes % 4) {
                err = "bf16 plane: src not multiple of 4";
                return false;
            }
            const uint64_t elems = src_bytes / 4;
            for (uint64_t i = 0; i < elems; ++i) {
                uint32_t v;
                std::memcpy(&v, src + i * 4, 4);
                const uint16_t h = (uint16_t) (v >> 16);
                std::memcpy(dst + i * 2, &h, 2);
            }
            dst_bytes = elems * 2;
            return true;
        }
        case Conv::ToF16: {
            if (src_bytes % 4) {
                err = "f16 plane: src not multiple of 4";
                return false;
            }
            const uint64_t elems = src_bytes / 4;
            for (uint64_t i = 0; i < elems; ++i) {
                float v;
                std::memcpy(&v, src + i * 4, 4);
                const uint16_t h = strata::kernels::f16_from_f32(v);
                std::memcpy(dst + i * 2, &h, 2);
            }
            dst_bytes = elems * 2;
            return true;
        }
        case Conv::WidenF16: {
            if (src_bytes % 2) {
                err = "widen f16: odd src";
                return false;
            }
            const uint64_t elems = src_bytes / 2;
            float* o = (float*) dst;
            for (uint64_t i = 0; i < elems; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                o[i] = strata::kernels::f32_from_f16(h);
            }
            dst_bytes = elems * 4;
            return true;
        }
    }
    err = "unknown conv";
    return false;
}

bool materialize_tensor(std::FILE* dense, const TrunkTensor& t, uint8_t* dst, std::string& err) {
    std::vector<uint8_t> src((size_t) t.src_bytes);
    if (!read_at(dense, t.src_off, src.data(), (size_t) t.src_bytes, err, t.name.c_str())) return false;

    if (t.codes_bytes != 0) {
        uint64_t at = 0;
        // codes
        std::memcpy(dst + at, src.data(), (size_t) t.codes_bytes);
        at += t.codes_bytes;
        if (t.scales_bytes) {
            const uint64_t src_scale = t.scales_fp16 ? t.scales_bytes / 2 : t.scales_bytes;
            uint64_t wrote = 0;
            if (!convert_plane(t.scales_fp16 ? Conv::WidenF16 : Conv::Copy, src.data() + t.codes_bytes, src_scale,
                               dst + at, wrote, err))
                return false;
            if (wrote != t.scales_bytes) {
                err = t.name + ": scale plane size mismatch";
                return false;
            }
            at += wrote;
        }
        if (t.offset_bytes) {
            const uint64_t after = t.codes_bytes + (t.scales_fp16 ? t.scales_bytes / 2 : t.scales_bytes);
            std::memcpy(dst + at, src.data() + after, (size_t) t.offset_bytes);
            at += t.offset_bytes;
        }
        if (at != t.dst_bytes) {
            err = t.name + ": materialized size mismatch";
            return false;
        }
        return true;
    }

    Conv c = Conv::Copy;
    if (!t.raw16) {
        if (t.kind == (int) WeightKind::Bf16InF32) c = Conv::Bf16High;
        else if (t.kind == (int) WeightKind::F16InF32) c = Conv::ToF16;
    }
    uint64_t wrote = 0;
    if (!convert_plane(c, src.data(), t.src_bytes, dst, wrote, err)) return false;
    if (wrote != t.dst_bytes) {
        err = t.name + ": dst size mismatch after convert";
        return false;
    }
    return true;
}

}  // namespace

void trunk_bind_hook_set(TrunkStreamer* streamer, WeightTable* table) {
    g_hook_streamer = streamer;
    g_hook_table = table;
}

void trunk_bind_hook_clear() {
    g_hook_streamer = nullptr;
    g_hook_table = nullptr;
}

bool trunk_bind_before_layer(int L, int n_layers, std::string& err) {
    if (!g_hook_streamer || !g_hook_table) return true;
    if (!g_hook_streamer->enabled()) return true;
    if (!g_hook_streamer->bind_layer(L, *g_hook_table, err)) return false;
    if (L + 1 < n_layers) g_hook_streamer->prefetch_layer(L + 1);
    return true;
}

void TrunkStreamer::plan_budget(const std::vector<uint64_t>& layer_bytes, uint64_t budget_bytes, int& npin,
                                int& nslot, uint64_t& slot_bytes) {
    const int n = (int) layer_bytes.size();
    npin = 0;
    nslot = 2;
    slot_bytes = 0;
    if (n == 0 || budget_bytes == 0) {
        nslot = 0;
        return;
    }
    for (int pass = 0; pass < 4; ++pass) {
        uint64_t big = 0;
        for (int i = npin; i < n; ++i) big = std::max(big, layer_bytes[(size_t) i]);
        if (big == 0 && n > 0) big = layer_bytes[(size_t) (n - 1)];
        uint64_t rs = (big + kAlign - 1) / kAlign * kAlign;
        int ring = 2;
        while (ring > 1 && (uint64_t) ring * rs > budget_bytes) --ring;
        uint64_t spent = (uint64_t) ring * rs;
        int pin = 0;
        while (pin < n) {
            const uint64_t need = (layer_bytes[(size_t) pin] + kAlign - 1) / kAlign * kAlign;
            if (spent + need > budget_bytes) break;
            spent += need;
            ++pin;
        }
        if (rs == slot_bytes && pin == npin) {
            nslot = ring;
            break;
        }
        slot_bytes = rs;
        npin = pin;
        nslot = ring;
    }
    if (npin >= n) {
        nslot = 0;  // everything pinned
        slot_bytes = 0;
    }
}

TrunkStreamer::~TrunkStreamer() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();
    if (dense_) {
        std::fclose(dense_);
        dense_ = nullptr;
    }
    for (uint8_t* p : pin_host_) std::free(p);
    pin_host_.clear();
    if (ring_host_) {
        std::free(ring_host_);
        ring_host_ = nullptr;
    }
    if (!host_only_) {
        for (void* p : pin_dev_)
            if (p) cudaFree(p);
        pin_dev_.clear();
        if (ring_dev_) {
            cudaFree(ring_dev_);
            ring_dev_ = nullptr;
        }
    }
}

bool TrunkStreamer::open(const std::string& pack_dir, uint64_t budget_bytes, std::string& err, bool host_only) {
    pack_dir_ = pack_dir;
    host_only_ = host_only;
    const std::string path = pack_dir + "/index.txt";
    std::FILE* idx = std::fopen(path.c_str(), "rb");
    if (!idx) {
        err = "trunk: cannot open " + path;
        return false;
    }

    int max_layer = -1;
    struct Row {
        TrunkTensor t;
        int layer = 0;
    };
    std::vector<Row> rows;
    char line[1024];
    while (std::fgets(line, sizeof line, idx)) {
        if (line[0] == '#') continue;
        char name[256] = {0};
        int file = 0, kind = 0;
        TrunkTensor t;
        uint64_t dst_off = 0;
        const int n = std::sscanf(
            line,
            "%255s %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNd64 " %" SCNd64
            " %d %d %d %d %d %" SCNu64 " %" SCNu64 " %" SCNu64 " %d %d",
            name, &file, &kind, &t.src_off, &t.src_bytes, &dst_off, &t.dst_bytes, &t.ne0, &t.ne1, &t.code_bits,
            &t.code_bias, &t.group_elems, &t.codebook, &t.has_offset, &t.codes_bytes, &t.scales_bytes,
            &t.offset_bytes, &t.scales_fp16, &t.act_kind);
        (void) dst_off;
        if (n != 19) continue;
        if (file != 0) continue;  // dense.bin only
        int layer = -1;
        if (!parse_blk_layer(name, layer)) continue;
        if (kind == 4 || kind == 5) {
            t.raw16 = true;
            kind = kind == 4 ? (int) WeightKind::Bf16InF32 : (int) WeightKind::F16InF32;
        }
        t.name = name;
        t.kind = kind;
        t.slot_off = 0;
        Row r;
        r.t = t;
        r.layer = layer;
        rows.push_back(r);
        max_layer = std::max(max_layer, layer);
    }
    std::fclose(idx);

    if (max_layer < 0 || rows.empty()) {
        err = "trunk: no blk.* tensors in index.txt";
        return false;
    }
    n_layers_ = max_layer + 1;
    layers_.assign((size_t) n_layers_, TrunkLayer{});
    for (auto& r : rows) {
        auto& L = layers_[(size_t) r.layer];
        r.t.slot_off = L.nbytes;
        L.nbytes += r.t.dst_bytes;
        L.tensors.push_back(std::move(r.t));
    }
    for (auto& L : layers_) L.nbytes = (L.nbytes + kAlign - 1) / kAlign * kAlign;

    std::vector<uint64_t> sizes((size_t) n_layers_);
    for (int i = 0; i < n_layers_; ++i) sizes[(size_t) i] = layers_[(size_t) i].nbytes;
    plan_budget(sizes, budget_bytes, npin_, nslot_, slot_bytes_);

    const std::string dense_path = pack_dir + "/dense.bin";
    dense_ = std::fopen(dense_path.c_str(), "rb");
    if (!dense_) {
        err = "trunk: cannot open " + dense_path;
        return false;
    }

    pin_host_.assign((size_t) std::max(npin_, 0), nullptr);
    pin_dev_.assign((size_t) std::max(npin_, 0), nullptr);
    for (int i = 0; i < npin_; ++i) {
        const uint64_t need = layers_[(size_t) i].nbytes;
        pin_host_[(size_t) i] = (uint8_t*) std::malloc((size_t) need);
        if (!pin_host_[(size_t) i]) {
            err = "trunk: host pin alloc failed";
            return false;
        }
        if (!host_only_) {
            if (cudaMalloc(&pin_dev_[(size_t) i], (size_t) need) != cudaSuccess) {
                err = "trunk: device pin alloc failed";
                return false;
            }
        }
        std::string e2;
        if (!read_layer_into(i, pin_host_[(size_t) i], e2)) {
            err = e2;
            return false;
        }
        if (!host_only_) {
            if (cudaMemcpy(pin_dev_[(size_t) i], pin_host_[(size_t) i], (size_t) need, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
                err = "trunk: pin H2D failed";
                return false;
            }
        }
    }

    if (nslot_ > 0) {
        ring_host_ = (uint8_t*) std::malloc((size_t) nslot_ * (size_t) slot_bytes_);
        if (!ring_host_) {
            err = "trunk: ring host alloc failed";
            return false;
        }
        if (!host_only_) {
            if (cudaMalloc(&ring_dev_, (size_t) nslot_ * (size_t) slot_bytes_) != cudaSuccess) {
                err = "trunk: ring device alloc failed";
                return false;
            }
        }
    }

    slot_of_layer_.assign((size_t) n_layers_, -1);
    for (int i = 0; i < npin_; ++i) slot_of_layer_[(size_t) i] = i;
    layer_of_slot_.assign((size_t) std::max(nslot_, 0), -1);

    // Async reader only with >= 2 ring slots (kimi correctness rule).
    io_enabled_ = (nslot_ >= 2);
    if (io_enabled_) {
        stop_ = false;
        io_thread_ = std::thread([this] { io_loop(); });
    }

    std::fprintf(stderr, "trunk stream: %d layers, %d pinned, ring %d x %.2f MiB%s\n", n_layers_, npin_, nslot_,
                 (double) slot_bytes_ / (1024.0 * 1024.0), io_enabled_ ? ", async prefetch on" : "");
    return true;
}

void TrunkStreamer::streamed_names(std::vector<std::string>& out) const {
    out.clear();
    for (const auto& L : layers_)
        for (const auto& t : L.tensors) out.push_back(t.name);
}

uint8_t* TrunkStreamer::host_slot_ptr(int slot_index) {
    if (slot_index < npin_) return pin_host_[(size_t) slot_index];
    const int r = slot_index - npin_;
    return ring_host_ + (size_t) r * (size_t) slot_bytes_;
}

void* TrunkStreamer::device_slot_ptr(int slot_index) {
    if (host_only_) return host_slot_ptr(slot_index);
    if (slot_index < npin_) return pin_dev_[(size_t) slot_index];
    const int r = slot_index - npin_;
    return (uint8_t*) ring_dev_ + (size_t) r * (size_t) slot_bytes_;
}

bool TrunkStreamer::read_layer_into(int L, uint8_t* host_dst, std::string& err) {
    if (L < 0 || L >= n_layers_) {
        err = "trunk: bad layer";
        return false;
    }
    const double t0 = now_s();
    const TrunkLayer& lay = layers_[(size_t) L];
    for (const auto& t : lay.tensors) {
        if (!materialize_tensor(dense_, t, host_dst + t.slot_off, err)) return false;
    }
    load_seconds_ += now_s() - t0;
    bytes_read_ += lay.nbytes;
    return true;
}

void TrunkStreamer::io_loop() {
    while (true) {
        int L = -1, slot = -1;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return stop_ || req_busy_; });
            if (stop_) break;
            L = req_layer_;
            slot = req_slot_;
        }
        std::string err;
        uint8_t* dst = host_slot_ptr(slot);
        const bool ok = read_layer_into(L, dst, err);
        if (ok && !host_only_) {
            void* d = device_slot_ptr(slot);
            if (cudaMemcpy(d, dst, (size_t) layers_[(size_t) L].nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "trunk: async H2D failed";
            }
        }
        {
            std::lock_guard<std::mutex> g(mu_);
            req_busy_ = false;
            req_done_ = true;
            req_result_ = err.empty() ? 0 : -1;
            io_err_ = err;
            if (req_result_ == 0) {
                // Clear old occupant mapping
                for (int i = 0; i < n_layers_; ++i)
                    if (slot_of_layer_[(size_t) i] == slot) slot_of_layer_[(size_t) i] = -1;
                layer_of_slot_[(size_t) (slot - npin_)] = L;
                slot_of_layer_[(size_t) L] = slot;
                ++misses_;
            }
        }
        cv_.notify_all();
    }
}

bool TrunkStreamer::ensure_resident(int L, std::string& err) {
    if (L < 0 || L >= n_layers_) {
        err = "trunk: bad layer";
        return false;
    }
    if (slot_of_layer_[(size_t) L] >= 0) {
        ++hits_;
        return true;
    }

    // Wait if async is loading this layer
    if (io_enabled_) {
        std::unique_lock<std::mutex> lock(mu_);
        if ((req_busy_ || req_done_) && req_layer_ == L) {
            cv_.wait(lock, [this] { return req_done_ || stop_; });
            if (req_result_ != 0) {
                err = io_err_.empty() ? "trunk: async read failed" : io_err_;
                req_done_ = false;
                return false;
            }
            req_done_ = false;
            ++hits_;  // treated as hit after wait
            return slot_of_layer_[(size_t) L] >= 0;
        }
    }

    if (nslot_ <= 0) {
        err = "trunk: layer not pinned and no ring";
        return false;
    }

    // Sync path into next ring slot
    const int r = ring_next_ % nslot_;
    ring_next_ = (ring_next_ + 1) % nslot_;
    const int slot = npin_ + r;
    // Evict previous
    const int old = layer_of_slot_[(size_t) r];
    if (old >= 0) slot_of_layer_[(size_t) old] = -1;

    uint8_t* host = host_slot_ptr(slot);
    if (!read_layer_into(L, host, err)) return false;
    if (!host_only_) {
        if (cudaMemcpy(device_slot_ptr(slot), host, (size_t) layers_[(size_t) L].nbytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            err = "trunk: H2D failed";
            return false;
        }
    }
    layer_of_slot_[(size_t) r] = L;
    slot_of_layer_[(size_t) L] = slot;
    ++misses_;
    return true;
}

bool TrunkStreamer::bind_layer(int L, WeightTable& wt, std::string& err) {
    if (!ensure_resident(L, err)) return false;
    const int slot = slot_of_layer_[(size_t) L];
    uint8_t* base = (uint8_t*) device_slot_ptr(slot);
    const TrunkLayer& lay = layers_[(size_t) L];
    for (const auto& t : lay.tensors) {
        WeightRef* wr = wt.find_mut(t.name);
        if (!wr) {
            // Metadata may be missing if load skipped before open filled table — install a row.
            WeightRef neu;
            neu.bytes = t.dst_bytes;
            neu.ne0 = t.ne0;
            neu.ne1 = t.ne1;
            neu.elements = t.ne0 * (t.ne1 > 0 ? t.ne1 : 1);
            neu.kind = (WeightKind) t.kind;
            neu.code_bits = t.code_bits;
            neu.code_bias = t.code_bias;
            neu.group_elems = t.group_elems;
            neu.codebook_iq4nl = t.codebook != 0;
            neu.has_offset = t.has_offset != 0;
            neu.codes_bytes = t.codes_bytes;
            neu.scales_bytes = t.scales_bytes;
            neu.offset_bytes = t.offset_bytes;
            neu.src_off = t.src_off;
            neu.src_bytes = t.src_bytes;
            neu.file_id = 0;
            neu.scales_fp16 = t.scales_fp16 != 0;
            neu.act_kind = t.act_kind;
            neu.resident = true;
            neu.data = base + t.slot_off;
            wt.install(t.name, neu);
            continue;
        }
        wr->data = base + t.slot_off;
        wr->resident = true;
    }
    return true;
}

void TrunkStreamer::prefetch_layer(int L) {
    if (!io_enabled_ || L < 0 || L >= n_layers_) return;
    if (slot_of_layer_[(size_t) L] >= 0) return;
    if (L < npin_) return;

    std::lock_guard<std::mutex> g(mu_);
    if (req_busy_) return;
    const int r = ring_next_ % nslot_;
    // Do not claim the slot currently bound for compute: use the other ring slot.
    // Prefer a free slot; if both occupied, take ring_next_ (same as kimi when budget allows 2).
    int use = r;
    if (nslot_ >= 2) {
        const int other = (r + 1) % nslot_;
        // Prefer slot not holding the previous layer if possible
        use = other;
        ring_next_ = (use + 1) % nslot_;
    }
    const int slot = npin_ + use;
    req_layer_ = L;
    req_slot_ = slot;
    req_busy_ = true;
    req_done_ = false;
    cv_.notify_one();
}

void TrunkStreamer::report() const {
    std::printf("TrunkStreamer: hits=%llu misses=%llu read=%.2f GB in %.2f s\n", (unsigned long long) hits_,
                (unsigned long long) misses_, (double) bytes_read_ / 1e9, load_seconds_);
}

}  // namespace strata::core
