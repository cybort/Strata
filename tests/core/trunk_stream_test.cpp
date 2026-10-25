// tests/core/trunk_stream_test.cpp - budget math and host-only open/bind against a tiny synthetic pack.
#include "strata/core/trunk_stream.hpp"
#include "strata/core/weights.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_fail = 0;
static void check(bool ok, const char* msg) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        ++g_fail;
    }
}

static void test_plan_budget() {
    // Three layers: 100, 50, 50 (already "aligned" units). Budget fits pin 0 + 2 ring of 50.
    std::vector<uint64_t> layers = {100, 50, 50};
    int npin = 0, nslot = 0;
    uint64_t slot = 0;
    // Align helper uses 256; pass multiples of 256
    layers = {256, 256, 256};
    strata::core::TrunkStreamer::plan_budget(layers, 256 * 3, npin, nslot, slot);
    check(npin + nslot >= 1, "budget plans at least one residency path");
    check(nslot == 0 || nslot == 1 || nslot == 2, "ring is 0..2");

    // Tiny budget: only ring
    strata::core::TrunkStreamer::plan_budget(layers, 256 * 2, npin, nslot, slot);
    check(nslot >= 1, "small budget still has a ring or pins");
    check(nslot <= 2, "ring never exceeds 2");

    // Huge budget: all pinned, no ring
    strata::core::TrunkStreamer::plan_budget(layers, 256ull * 100, npin, nslot, slot);
    check(npin == 3, "large budget pins all layers");
    check(nslot == 0, "all-pinned means no ring");
}

static bool write_tiny_pack(const fs::path& dir) {
    fs::create_directories(dir);
    // One layer, one verbatim F32 tensor: 8 floats = 32 bytes src/dst, kind F32=2
    // index: name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group codebook
    //        has_offset codes scales offset scales_fp16 act_kind
    std::ofstream idx(dir / "index.txt");
    idx << "# align 256 pool 256 tensors 1\n";
    idx << "blk.0.attn_q.weight 0 2 0 32 0 32 8 1 0 0 0 0 0 0 0 0 0 0\n";
    idx << "token_embd.weight 1 2 0 16 0 16 4 1 0 0 0 0 0 0 0 0 0 0\n";
    idx.close();

    std::vector<float> dense(8, 1.5f);
    std::ofstream d(dir / "dense.bin", std::ios::binary);
    d.write(reinterpret_cast<const char*>(dense.data()), 32);
    d.close();

    std::vector<float> emb(4, 2.0f);
    std::ofstream e(dir / "embd.bin", std::ios::binary);
    e.write(reinterpret_cast<const char*>(emb.data()), 16);
    e.close();
    return true;
}

static void test_host_open_bind() {
    const fs::path dir = fs::temp_directory_path() / "strata_trunk_stream_test";
    check(write_tiny_pack(dir), "write tiny pack");
    strata::core::TrunkStreamer ts;
    std::string err;
    // Budget large enough to pin the one layer
    check(ts.open(dir.string(), 64ull << 20, err, /*host_only=*/true), err.c_str());
    check(ts.n_layers() == 1, "one layer");
    check(ts.npin() == 1, "layer pinned");
    check(ts.nslot() == 0, "no ring when fully pinned");

    strata::core::WeightTable wt;
    check(ts.bind_layer(0, wt, err), err.c_str());
    const strata::core::WeightRef* w = wt.find("blk.0.attn_q.weight");
    check(w != nullptr && w->data != nullptr, "bound tensor has data");
    check(w->resident, "bound tensor resident");
}

int main() {
    test_plan_budget();
    test_host_open_bind();
    if (g_fail) {
        std::fprintf(stderr, "trunk_stream_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("trunk_stream_test: OK\n");
    return 0;
}
