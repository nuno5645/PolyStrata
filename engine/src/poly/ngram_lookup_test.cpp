// src/poly/ngram_lookup_test.cpp - strata::spec::NgramLookup (STRATA_LOOKUP_NINFER) against NInfer's lookup_draft()
// as its header has it (exact n-gram, the followed copy's lag first, then the newest occurrence), on random ledgers
// of few distinct tokens, as they grow a token at a time; and the shorter matches of min_n < n.  No card, no model.
#include "strata/spec/ngram_lookup.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// NInfer's runtime::lookup_draft() (src/runtime/contract/lookup_draft.h), with std::vector for its span
uint32_t ref_lookup(const std::vector<int32_t>& ledger, uint32_t ngram, uint32_t max_drafts, int32_t* out, size_t& lag) {
    if (ngram == 0 || max_drafts == 0 || out == nullptr) return 0;
    const size_t size = ledger.size();
    if (size <= ngram) return 0;
    const int32_t* const tail = ledger.data() + (size - ngram);
    const auto matches = [&](size_t start) {
        for (uint32_t j = 0; j < ngram; ++j)
            if (ledger[start + j] != tail[j]) return false;
        return true;
    };
    const auto propose = [&](size_t follow) {
        const size_t available = size - follow;
        const auto count = (uint32_t) (available < max_drafts ? available : max_drafts);
        for (uint32_t j = 0; j < count; ++j) out[j] = ledger[follow + j];
        lag = size - follow;
        return count;
    };
    if (lag > 0 && lag < size) {
        const size_t follow = size - lag;
        if (follow >= ngram && matches(follow - ngram)) return propose(follow);
    }
    for (size_t start = size - ngram; start-- > 0;)
        if (matches(start)) return propose(start + ngram);
    lag = 0;
    return 0;
}

int failures = 0;

void check(bool ok, const char* what, int trial, size_t at) {
    if (ok) return;
    if (++failures <= 10) std::fprintf(stderr, "FAIL %s: trial %d, ledger of %zu\n", what, trial, at);
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    long long proposals = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int alphabet = 2 + trial % 6, n = 1 + trial % 9, d = 1 + trial % 8;
        std::uniform_int_distribution<int> tok(0, alphabet - 1);
        std::vector<int32_t> ledger;
        strata::spec::NgramLookup mine;
        mine.n = mine.min_n = n;
        size_t ref_lag = 0;
        for (int step = 0; step < 300; ++step) {
            // a copy of an earlier stretch now and then, as an answer quoting its prompt
            if (step % 50 == 25 && ledger.size() > 40) {
                const size_t from = ledger.size() / 3;
                for (size_t i = 0; i < 20; ++i) ledger.push_back(ledger[from + i]);
            } else {
                ledger.push_back(tok(rng));
            }
            int32_t a[16] = {}, b[16] = {};
            const int got = mine.propose(ledger.data(), ledger.size(), d, a);
            const int want = (int) ref_lookup(ledger, (uint32_t) n, (uint32_t) d, b, ref_lag);
            check(got == want, "count", trial, ledger.size());
            check(mine.lag == ref_lag, "lag", trial, ledger.size());
            for (int j = 0; j < got && j < want; ++j) check(a[j] == b[j], "token", trial, ledger.size());
            check(got == 0 || mine.last_match == std::min<int>(n, (int) ledger.size() - 1), "match", trial,
                  ledger.size());
            proposals += got;
        }
    }
    // min_n < n: a match of min_n..n-1 tokens where no n-token one is; the longest wins, the newest of equals
    {
        strata::spec::NgramLookup l;
        l.n = 8;
        l.min_n = 3;
        const std::vector<int32_t> led = {1, 2, 3, 4, 50, 9, 2, 3, 4, 60, 7, 2, 3, 4};
        int32_t out[8] = {};
        const int k = l.propose(led.data(), led.size(), 4, out);
        check(k == 4 && out[0] == 60 && out[1] == 7 && l.last_match == 3, "newest of equal short matches", -1, led.size());
        const std::vector<int32_t> led2 = {5, 1, 2, 3, 4, 70, 2, 3, 4, 80, 5, 1, 2, 3, 4};
        l.reset();
        const int k2 = l.propose(led2.data(), led2.size(), 3, out);
        check(k2 == 3 && out[0] == 70 && l.last_match == 5, "longest short match", -1, led2.size());
        l.reset();
        l.min_n = 8;
        check(l.propose(led2.data(), led2.size(), 3, out) == 0, "exact n finds none", -1, led2.size());
    }
    if (failures) {
        std::fprintf(stderr, "ngram_lookup_test: %d failures\n", failures);
        return 1;
    }
    std::printf("ngram_lookup_test: ok (%lld tokens proposed, all as NInfer's lookup_draft)\n", proposals);
    return 0;
}
