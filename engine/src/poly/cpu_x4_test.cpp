// src/poly/cpu_x4_test.cpp - the four-row dot products cpu_experts() takes (cpu_dot_x4) give, bit for bit, the one-row
// products they stand in for: random weight rows with real scales against activation rows quantized by the layer's own
// quantizer, several row lengths.  Exit 0 = pass (a type this CPU has no four-row kernel for is skipped), 1 = a value
// differs or a four-row kernel has a type this test does not know how to fill.  The opt-in four-row kernels
// (STRATA_CPU_X4_NVFP4) are switched on here, and the AVX2 level's four-row kernels are also checked against the AVX2
// one-row kernels where this CPU has a higher level (a Zen 2's kernels on an AVX-512 machine).
#include "cpu_experts.hpp"

#include "strata/quant/act.hpp"
#include "strata/quant/dot.hpp"
#include "strata/quant/fp16.hpp"
#include "strata/quant/types.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace Q = strata::quant;

namespace {

// a finite half in +/- [2^-7, 2): a block scale
uint16_t scale_half(std::mt19937& rng) {
    return (uint16_t) (((rng() & 1u) << 15) | ((8u + rng() % 8u) << 10) | (rng() & 0x3ffu));
}

// Random bytes, then each block's scale fields made finite: where they sit in each type's block.
bool fill(Q::Type t, std::vector<uint8_t>& w, int blocks, std::mt19937& rng) {
    for (auto& b : w) b = (uint8_t) rng();
    const Q::TypeInfo* ti = Q::type_info((int) t);
    for (int i = 0; i < blocks; ++i) {
        uint8_t* b = w.data() + (size_t) i * ti->bytes;
        auto put = [&](size_t at) { const uint16_t h = scale_half(rng); std::memcpy(b + at, &h, 2); };
        switch (t) {
            case Q::Type::Q4_0: case Q::Type::Q8_0: case Q::Type::IQ4_XS: case Q::Type::IQ3_XXS: put(0); break;
            case Q::Type::Q4_K: case Q::Type::Q5_K: put(0); put(2); break;
            case Q::Type::Q6_K: put(208); break;
            case Q::Type::MXFP4: b[0] = (uint8_t) (118 + rng() % 18); break;
            case Q::Type::NVFP4:   // four UE4M3 scales of exponent 4 to 11: no NaN, no zero
                for (int j = 0; j < 4; ++j) b[j] = (uint8_t) (((4u + rng() % 8u) << 3) | (rng() & 7u));
                break;
            default: return false;
        }
    }
    return true;
}

using OneFn = void (*)(int n, float* s, const void* w, const void* a);
using FourFn = void (*)(int n, float out[4], const void* w, size_t row_bytes, const void* a);

// how many of the values four rows a call give differ from the one-row kernel's, over several row lengths; -1: fill()
// does not know the type's blocks
int compare(Q::Type t, OneFn one_fn, FourFn four_fn, int* values_out) {
    const Q::TypeInfo* ti = Q::type_info((int) t);
    const Q::ActKernel* act = Q::act_kernel(ti->act, Q::default_arith());
    std::mt19937 rng(0x5eedu + (uint32_t) t);
    std::normal_distribution<float> nd(0.f, 1.f);
    int differ = 0, values = 0;
    for (const int n : {256, 512, 1024, 1152, 2048, 2304, 4096}) {
        if (n % ti->blck != 0) continue;
        const int blocks = n / ti->blck;
        const size_t row = Q::row_bytes((int) t, n);
        std::vector<uint8_t> w(4 * row), a(Q::row_bytes((int) ti->act, n));
        std::vector<float> x((size_t) n);
        for (int trial = 0; trial < 64; ++trial) {
            if (!fill(t, w, 4 * blocks, rng)) return -1;
            for (auto& v : x) v = nd(rng) * (trial % 4 == 0 ? 8.f : 1.f);
            act->fn(x.data(), a.data(), n);
            float four[4], one[4];
            four_fn(n, four, w.data(), row, a.data());
            for (int r = 0; r < 4; ++r) one_fn(n, one + r, w.data() + (size_t) r * row, a.data());
            for (int r = 0; r < 4; ++r, ++values) differ += std::memcmp(four + r, one + r, sizeof(float)) != 0;
        }
    }
    *values_out = values;
    return differ;
}

}  // namespace

int main() {
    setenv("STRATA_CPU_X4_NVFP4", "1", 1);   // the opt-in four-row kernels, read once at the first lookup
    const Q::Type types[] = {Q::Type::Q4_0, Q::Type::Q8_0, Q::Type::MXFP4, Q::Type::Q4_K, Q::Type::Q5_K,
                             Q::Type::Q6_K, Q::Type::IQ4_XS, Q::Type::IQ3_XXS, Q::Type::IQ4_NL, Q::Type::Q3_K,
                             Q::Type::Q2_K, Q::Type::IQ2_S, Q::Type::IQ3_S, Q::Type::NVFP4};
    int failed = 0, checked = 0;
    for (const Q::Type t : types) {
        const strata::poly::CpuDotX4 k = strata::poly::cpu_dot_x4((int) t);
        const Q::TypeInfo* ti = Q::type_info((int) t);
        if (k.four == nullptr) {
            std::printf("cpu_x4_test: %-8s one row at a time\n", ti->name);
            continue;
        }
        int values = 0;
        const int differ = compare(t, k.one, k.four, &values);
        if (differ < 0) {
            std::printf("cpu_x4_test: %s has a four-row kernel; fill() does not know its blocks\n", ti->name);
            return 1;
        }
        ++checked;
        failed += differ != 0;
        std::printf("cpu_x4_test: %-8s four rows a call: %d of %d values differ from the one-row kernel\n", ti->name,
                    differ, values);
    }
    // the AVX2 level's own pairs (what an AVX2 CPU without AVX-512 takes), wherever this CPU runs AVX2
    for (const Q::Type t : types) {
        const Q::DotX4Kernel* four = Q::dot_x4_kernel(t, Q::Arith::Avx2, Q::Isa::Avx2);
        const Q::DotKernel* one = Q::dot_kernel(t, Q::Arith::Avx2, Q::Isa::Avx2);
        if (four == nullptr || one == nullptr || four->isa != one->isa) continue;
        const Q::TypeInfo* ti = Q::type_info((int) t);
        int values = 0;
        const int differ = compare(t, one->fn, four->fn, &values);
        if (differ < 0) {
            std::printf("cpu_x4_test: %s has a four-row kernel; fill() does not know its blocks\n", ti->name);
            return 1;
        }
        ++checked;
        failed += differ != 0;
        std::printf("cpu_x4_test: %-8s AVX2 level, %s vs %s: %d of %d values differ\n", ti->name, four->name, one->name,
                    differ, values);
    }
    std::printf("cpu_x4_test: %d checks of a four-row kernel, %s\n", checked, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
