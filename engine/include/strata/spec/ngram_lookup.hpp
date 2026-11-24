// include/strata/spec/ngram_lookup.hpp - NInfer's context lookup (--lookup-ngram N --draft-tokens D), for strata-poly's
// STRATA_LOOKUP_NINFER=1.
//
// The tail of the ledger (the prompt, every token shown, the last one the token about to go in) is looked for in the
// whole ledger, newest occurrence first, and the D tokens that followed it there are proposed.  Unlike SuffixDrafter,
// which keeps the 4 newest end positions of each trigram, every occurrence is a candidate, and the caller verifies
// all of them in one window whatever the match's length: a copy of a long prompt (an edit, the reasoning quoting the
// code) is found where the four newest trigrams lie elsewhere.
//
// As NInfer's runtime::lookup_draft(): `lag` remembers the copy the answer follows (the distance from the ledger's
// end to the copy's next source token); while that source still matches the last N tokens it is proposed again, so
// a text that occurs twice keeps its source instead of jumping to the newest copy at each place they differ.  A
// stale lag finds no match and falls back to the scan.
//
// min_n < n: a tail that matches nowhere for n tokens may match for fewer, down to min_n; the longest match wins,
// the newest of equals.  min_n == n is NInfer's exact policy.
//
// It only proposes: the caller's window verifies each token against the model's own pick, so the output is the
// same token for token whatever is proposed.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::spec {

struct NgramLookup {
    int n = 8;         ///< tokens of the tail matched (NInfer's --lookup-ngram)
    int min_n = 8;     ///< the shortest match taken when no n-token one is found
    size_t lag = 0;    ///< the followed copy's source, as its distance from the ledger's end; 0: none
    int last_match = 0;   ///< the match length behind the last proposal, 0 for none

    void reset() { lag = 0; last_match = 0; }

    /// Up to max_k tokens that followed the newest earlier occurrence of the ledger's tail, written to out; the count.
    int propose(const int32_t* ledger, size_t size, int max_k, int32_t* out) {
        last_match = 0;
        if (n <= 0 || max_k <= 0 || out == nullptr) return 0;
        const int lo = min_n < 1 ? 1 : (min_n > n ? n : min_n);
        if (size <= (size_t) lo) return 0;
        const int32_t* tail_end = ledger + size;   // one past the tail's last token
        // the match length (up to n) of the stretch that ends just before `follow` with the tail
        const auto match_at = [&](size_t follow) {
            int len = 0;
            while (len < n && (size_t) len < follow && ledger[follow - 1 - (size_t) len] == tail_end[-1 - len]) ++len;
            return len;
        };
        const auto take = [&](size_t follow, int len) {
            const size_t available = size - follow;
            const int count = available < (size_t) max_k ? (int) available : max_k;
            for (int j = 0; j < count; ++j) out[j] = ledger[follow + (size_t) j];
            lag = size - follow;
            last_match = len;
            return count;
        };
        const int full = size > (size_t) n ? n : (int) size - 1;   // the longest match the ledger allows
        if (full < lo) return 0;
        if (lag > 0 && lag < size) {   // the copy this answer followed, its source moved on with the ledger
            const size_t follow = size - lag;
            const int len = match_at(follow);
            if (len >= full) return take(follow, len);
        }
        // backwards, so that the newest occurrence wins: in repeating text the nearest copy most often goes on alike
        size_t best_follow = 0;
        int best = 0;
        const int32_t last = tail_end[-1];
        for (size_t follow = size - 1; follow > 0; --follow) {
            if (ledger[follow - 1] != last) continue;
            const int len = match_at(follow);
            if (len > best) {
                best = len;
                best_follow = follow;
                if (len >= full) break;
            }
        }
        if (best < lo) {
            lag = 0;
            return 0;
        }
        return take(best_follow, best);
    }
};

}  // namespace strata::spec
