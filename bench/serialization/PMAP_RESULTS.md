# Real-world demo: Distributed `pmap` roundtrip

A "normal" data-analytics pattern: a coordinator dispatches 200 chunks of work
to 4 workers via `pmap`; each worker generates a 50 000-row chunk, computes a
small per-chunk aggregate, and returns the rows back to the coordinator. The
master then has a `Vector` of 200 chunks to feed into the next stage.

This exercises the stdlib `Serialization` library through Distributed.jl's
wire protocol — which is precisely the path our optimizations target (unlike
the sysimage path, which goes through `src/staticdata.c`).

## Workload (`/tmp/serialization_bench/pmap_bench.jl`)

Per-chunk payload (50 000 rows × 6 columns):

| Column | Type | Why it stresses serialization |
|---|---|---|
| `x`   | `Vector{Float64}` | bits-array baseline |
| `z`   | `Vector{Int}` | bits-array baseline |
| `flg` | `Vector{Bool}` | bits-array baseline |
| `y`   | `Vector{Union{Float64,Missing}}` | **bits-union fast path (f)** |
| `cat` | `Vector{String}` | **`SHARED_REF_TAG` / `Vector{Any}` deserialize (g, h)** |
| `summary` | `NamedTuple` of 6 scalars | tuple unroll (tup_unroll) |

Plus a small per-chunk aggregate (`mean`, `std`, `count`, `Set` size) so
workers do real compute, not just shipping.

## Results

`time(pmap)` end-to-end, 4 workers, 200 chunks. Trial 1 includes JIT/cache
warmup; trials 2-3 are steady state.

| | trial 1 | trial 2 | trial 3 |
|---|---|---|---|
| **baseline** `df12394092` (run A) | 2.785 s | 2.344 s | 2.244 s |
| **baseline** `df12394092` (run B) | 2.645 s | 2.351 s | 2.197 s |
| **HEAD** `6d56f3f914` (run A)     | 2.048 s | 1.598 s | 1.700 s |
| **HEAD** `6d56f3f914` (run B)     | 2.094 s | 1.679 s | 1.691 s |

Steady-state (trials 2-3) medians:

| | seconds | ratio |
|---|---|---|
| baseline | 2.270 s | 1.00× |
| HEAD     | 1.685 s | **0.74× (≈26 % faster)** |

## Why this matters

`pmap` time is dominated by the result roundtrip: each worker serializes a
chunk and ships it to the master, which deserializes sequentially before
handing the value to the user. The wins concentrate exactly where the
optimizations land:

- The `Vector{Union{Float64,Missing}}` column ships the bits-union memory
  block as a single `unsafe_write` and reconstructs it with one
  `unsafe_read` (commit `f`), rather than per-element typed dispatch.
- The 50 000 short strings are deduplicated via the open-addressed
  `_sertag_lookup` (commit `flat_sertag`) and read back through the
  specialized `Vector{Any}` deserialize (commit `g`), which avoids the
  per-back-ref `IdDict` boxing.
- Cycle table accesses on both sides go through the typed `CycleTable`
  (commit `h`) instead of `IdDict`/`Dict{Int}`.

For a workload where the master is the bottleneck (many workers shipping
results to one coordinator), the ~26 % wall-clock improvement comes
essentially for free — no API change, no opt-in.

## Reproducing

```sh
# HEAD
./usr/bin/julia --startup-file=no /tmp/serialization_bench/pmap_bench.jl

# baseline — restore the single source file, run, then restore HEAD
git checkout df12394092 -- stdlib/Serialization/src/Serialization.jl
./usr/bin/julia --startup-file=no /tmp/serialization_bench/pmap_bench.jl
git checkout HEAD -- stdlib/Serialization/src/Serialization.jl
```

(`Serialization.jl` is pure Julia — no rebuild needed; Julia re-precompiles
the stdlib automatically when the source changes.)
