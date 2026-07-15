# K4 Argmax-Segment Run: Excluded from RQ3

These historical configurations are retained for provenance only. Post-run source audit found that the loss prepended a designated target logit while also retaining the same target occurrence and all repeated occurrences of that GPT-2 token in the comparison bank.

The run therefore optimized an unmasked, token-occurrence-weighted ranking loss rather than the intended positive-masked per-position InfoNCE. Its recognition values do not test SEA and must not be included in active Chapter 6 comparisons.

See `REPRODUCIBILITY.md`, section "Excluded K4 argmax-segment run", for the exact objective and canonical result paths.
