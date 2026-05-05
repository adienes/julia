# Regression hunt — second pass

After the alt-suite walk had already cleared 9 workloads, this pass
deliberately probes patterns the existing suites did *not* exercise:

| Bucket | Workload |
|---|---|
| **Per-call overhead**     | scalar `Int`, singleton `Vector{Int}`, empty `Vector`/`Tuple`/`Dict` |
| **Past unroll cliff**     | tuples of length 11, 16, 32, 64 (unroll covers ≤10) |
| **Heavy unroll usage**    | `Vector{NTuple{4,Int}}` (100k) |
| **Other bits arrays**     | `Vector{Float32}` 1M, `Vector{ComplexF64}` 500k, `Vector{UInt8}` 10M |
| **Sertag misses / probes**| 500 distinct types, 500 distinct instances, 2k distinct `Symbol`s |
| **Cycle pressure**        | 10k refs to one vector, 10k refs to a 1k-pool, 100k repeats of one symbol |
| **Recursion / depth**     | 5k-node linked list, nested `Vector{Vector{Int}}` |
| **Wide aggregates**       | `NamedTuple` with 16 fields, vector of same |
| **Edge sizes**            | one 10 MB string, 50k unique short strings |

`/tmp/serialization_bench/regression_hunt.jl` runs all of these with
`@benchmark`; `/tmp/serialization_bench/rh_compare.jl` parses the dump
and prints a HEAD/baseline table.

## Regressions found (first pass)

**Every** workload paid `+400 bytes / +400 bytes` per call vs baseline,
regardless of payload size. On sub-microsecond workloads this surfaced
as a measurable 10-30 ns regression:

| Workload | base ser | HEAD ser | ratio | base de | HEAD de | ratio |
|---|---|---|---|---|---|---|
| `r1_scalar_int`    | 93 ns | 115 ns | **1.23×** | 49 ns | 75 ns | **1.52×** |
| `r3_empty_tuple`   | 84 ns | 109 ns | **1.29×** | 49 ns | 75 ns | **1.52×** |
| `r3_empty_vec`     | 122 ns | 132 ns | 1.08× | 292 ns | 325 ns | **1.11×** |
| `r2_singleton_vec` | 127 ns | 138 ns | 1.09× | 294 ns | 333 ns | **1.13×** |
| `r4_tuple_len11`   | 153 ns | 173 ns | **1.14×** | 482 ns | 508 ns | 1.06× |
| `r4_tuple_len16`   | 186 ns | 213 ns | **1.15×** | 641 ns | 666 ns | 1.04× |

## Root cause

`CycleTable()` in `commit h_custom_cycle_tbl` allocated a 16-slot
`Memory{Any}` and a 16-slot `Memory{Int}` *eagerly* in its constructor
(~330 bytes per Serializer construction), even when the payload never
inserts a single cycle entry — the Int / empty-tuple / short-tuple
paths.

## Fix

Lazy-initialize the `CycleTable` buffers. The struct now holds
references to shared, length-0 `Memory{Any}` / `Memory{Int}` sentinels;
`_cycle_grow!` allocates the real 16-slot buffers on the first
insert. No change to the open-addressing logic.

```julia
const _EMPTY_CYCLE_KEYS = Memory{Any}(undef, 0)
const _EMPTY_CYCLE_VALS = Memory{Int}(undef, 0)

mutable struct CycleTable
    keys::Memory{Any}
    vals::Memory{Int}
    count::Int
    CycleTable() = new(_EMPTY_CYCLE_KEYS, _EMPTY_CYCLE_VALS, 0)
end

# `cycle_get` early-exits on `sz == 0`; `_cycle_grow!` allocates 16 on the
# first transition out of the shared empty buffers.
```

## After the fix

| Workload | base ser | HEAD ser | ratio | base de | HEAD de | ratio |
|---|---|---|---|---|---|---|
| `r1_scalar_int`    | 93 ns  | 102 ns | 1.09× | 49 ns  | 59 ns  | 1.20× |
| `r3_empty_tuple`   | 84 ns  | 93 ns  | 1.10× | 49 ns  | 58 ns  | 1.19× |
| `r3_empty_vec`     | 122 ns | 143 ns | 1.17× | 292 ns | 361 ns | 1.24× |
| `r2_singleton_vec` | 127 ns | 148 ns | 1.17× | 294 ns | 368 ns | 1.25× |
| `r4_tuple_len11`   | 153 ns | 163 ns | 1.07× | 482 ns | 492 ns | 1.02× |
| `r4_tuple_len16`   | 186 ns | 195 ns | 1.05× | 641 ns | 653 ns | 1.02× |

Per-call allocation on cycle-free tiny workloads dropped from
`+400/+400` to `+80/+80` bytes, and the ratios pulled in by ~10-15
percentage points.

## Final fix: drop the vestigial `table` field

A 5-25 % residual still survived on sub-µs workloads after the lazy
`CycleTable`. Looking at *why* the per-call allocation was still above
baseline turned up the actual culprit: the `Serializer` struct kept a
`table::IdDict{Any,Any}` field that no `Serializer` method read once
cycle dedup had been routed through `CycleTable` and back-references
through `deserialize_table`. The `IdDict()` in the constructor was a
wasted ~280-byte allocation per `Serializer`.

Dropping the field eliminated the residual entirely:

| Workload | baseline | HEAD with field | HEAD field-dropped |
|---|---|---|---|
| `r1_scalar_int` ser  | 93 ns  | 102 ns (1.08×) | **79 ns (0.84×)** |
| `r1_scalar_int` de   | 49 ns  | 59 ns  (1.20×) | **40 ns (0.81×)** |
| `r3_empty_tuple` ser | 84 ns  | 93 ns  (1.10×) | **69 ns (0.82×)** |
| `r3_empty_tuple` de  | 49 ns  | 58 ns  (1.20×) | **38 ns (0.77×)** |
| `r3_empty_vec` ser   | 122 ns | 143 ns (1.17×) | **116 ns (0.95×)** |
| `r3_empty_vec` de    | 292 ns | 361 ns (1.24×) | **254 ns (0.87×)** |
| `r4_tuple_len11` ser | 153 ns | 163 ns (1.07×) | **133 ns (0.87×)** |

Every workload in the regression hunt now lands at ≤ 1.05× baseline,
and most are substantially faster. Per-call allocation on cycle-free
tiny payloads went from baseline 736/560 B to HEAD **480/304 B** — a
net 256-byte reduction (vs. +80 B before this fix).

## Other findings worth noting

- `r6_vec_uint8_10M` and `r6_vec_complexf64_500k`: clean **1.00×**.
  The bits-array path is not regressed for non-`Float64`/`Int` element
  types.
- `r4_tuple_len32` / `r4_tuple_len64`: clean **1.01-1.02×**. Falling
  off the unroll cliff (length > 10) is harmless — the slow path is
  unchanged.
- `r12_2k_distinct_symbols`: **0.39× ser**, 0.92× de. The flat sertag
  table's open addressing is *not* a problem under high-distinct-key
  pressure; the win comes from removing IdDict per-`setindex!` boxing.
- `r14_100k_repeated_symbol`: **0.20× ser**, **0.40× de**. Hot-path
  sertag hits are dramatically faster.
- `r9_10k_refs_1k_pool`: **0.61× de**. Realistic mixed back-ref
  pattern wins handily.

## Reproducing

```sh
./usr/bin/julia --startup-file=no --project=/tmp/serialization_bench/env \
    /tmp/serialization_bench/regression_hunt.jl > rh_head.log

git checkout df12394092 -- stdlib/Serialization/src/Serialization.jl
./usr/bin/julia --startup-file=no --project=/tmp/serialization_bench/env \
    /tmp/serialization_bench/regression_hunt.jl > rh_baseline.log
git checkout HEAD -- stdlib/Serialization/src/Serialization.jl

julia /tmp/serialization_bench/rh_compare.jl
```
