# Serialization optimization benchmarks

`@btime` median across 3 runs from `bench/serialization/suite.jl`. Numbers
within ~1% across runs are treated as noise.

## Time

| Stage | V{Int} ser | V{Int} de | V{U{I,M}} ser | V{U{I,M}} de | Trees ser | Trees de | herk ser | herk de |
|---|---|---|---|---|---|---|---|---|
| baseline                          | 168 μs | 157 μs | 15.16 ms  | 12.03 ms  | 9.62 ms | 8.15 ms |  889 μs |  60.0 μs |
| (a) UnionAll cycle detect (#58007)| 161 μs | 150 μs | 15.18 ms  | 12.06 ms  | 9.68 ms | 8.27 ms |  913 μs |  55.6 μs |
| (f) bits-union array fast-path    | 161 μs | 152 μs | **184 μs**| **255 μs**| 9.61 ms | 9.20 ms |  912 μs |  53.8 μs |
| (g) Vector{Any} deserialize table | 167 μs | 153 μs |   186 μs  |   260 μs  | 9.70 ms | **6.75 ms** | 912 μs | 49.7 μs |
| (h) custom serialize cycle table  | 165 μs | 152 μs |   185 μs  |   258 μs  | **5.18 ms** | 6.65 ms |  931 μs |  48.7 μs |
| @generated unrolled tuple deser   | 167 μs | 154 μs |   188 μs  |   258 μs  | 5.18 ms | **4.91 ms** | 931 μs | 48.4 μs |
| geometric `deserialize_table` grow| 167 μs | 152 μs |   186 μs  |   258 μs  | 5.18 ms | **4.79 ms** | 929 μs | 48.0 μs |
| flat sertag lookup table          | 162 μs | 152 μs |   189 μs  |   261 μs  | **4.95 ms** | 4.97 ms | 928 μs | 48.0 μs |
| `serialize(::Union)` specialization| 161 μs | 151 μs |   183 μs  |   250 μs  | 4.44 ms | 4.60 ms | **43 μs** | **40 μs** |
| factor cycle-table accessors      | **156 μs** | **146 μs** | **181 μs** | **248 μs** | **4.43 ms** | **3.97 ms** | 43 μs | 39.7 μs |

## Allocations (serialize / deserialize)

| Stage | V{Int} | V{U{I,M}} | Trees | herk |
|---|---|---|---|---|
| baseline | 13 / 10           | 1,000,050 / 999,018 | 61,239 / 114,018 | 513 / 481 |
| (a)      | 13 / 10           | 1,000,050 / 999,018 | 61,239 / 114,018 | 504 / 472 |
| (f)      | 13 / 18           | 14 / 22             | 61,239 / 114,025 | 504 / 472 |
| (g)      | 14 / 20           | 15 / 24             | 61,240 / 83,690     | 505 / 473 |
| (h)      | 17 / 23           | 18 / 27             | **30,910** / 83,693 | 510 / 476 |
| tuple unroll | 17 / 23       | 18 / 27             | 30,910 / **48,493** | 510 / 476 |
| flat sertag | 17 / 23        | 18 / 27             | 30,910 / 48,493     | 510 / 476 |
| Union spec | 17 / 23         | 18 / 27             | 30,910 / 48,493     | **150** / 476 |
| inline cycle | 17 / 23       | 18 / 27             | 30,910 / 48,493     | 150 / 476 |

## Cumulative effect (baseline → latest)

| Workload | serialize | deserialize |
|---|---|---|
| Vector{Int} (1M)                | 168 μs → 156 μs (within noise) | 157 μs → 146 μs |
| Vector{Union{Int,Missing}} (1M) | **15.16 ms → 181 μs (~84×)** | **12.03 ms → 248 μs (~49×)** |
| 50 trees of depth 4             | **9.62 ms → 4.43 ms (~54%)**, **61k → 31k allocs** | **8.15 ms → 3.97 ms (~51%)**, **114k → 48k allocs** |
| herk inner Union                | **889 μs → 43 μs (~21×)**    | **60 μs → 40 μs (~33%)** |

## Notes

- **(a)** UnionAll cycle detect — output bytes ~5% smaller on signature types
  with shared UnionAll subterms (#58007). Time-wise within noise.
- **(f)** bits-union array fast-path — headline fix for #30148.
- **(g)** Vector{Any} deserialize back-ref table — replaces the IdDict.
- **(h)** Custom open-addressed cycle table for serialize side, growing 4×
  per resize.
- **`@generated` tuple unroll** — avoids the per-call `Vector{Any}` allocation
  in `ntupleany`.
- **geometric `deserialize_table` grow** — `settable!` extends to
  `max(slot+1, 2*length(tbl)+1)` so sequential slot writes only resize
  logarithmic times.
- **flat sertag lookup** — `Memory{Any}` keys + `Memory{Int32}` values,
  identity hash, ~3 ns/call (vs the original ~30 ns linear scan over `TAGS`).
- **`serialize(::Union)` specialization** — Union values had been going
  through `serialize_any` (sertag, typeof, ..., nfields/getfield loop). Inline
  two recursive `serialize` calls. The `@nospecialize` on the dispatch
  argument matters.
- **factor cycle-table accessors per-Serializer-type** — split the storage
  interface into single-statement `_cycle_lookup` (read) and
  `_cycle_register!` (write) helpers. `serialize_cycle` stays as one shared
  function while the per-type backing (IdDict for `AbstractSerializer`,
  `CycleTable` for `Serializer`) is reached through the helpers. Better
  inlining than the previous combined helper.

## Skipped recommendations

- **(b) sertag primitive bypass** — confirmed wash; the per-element dispatch
  for `Vector{Union{Int,Missing}}` picks the specific Int method, so `sertag`
  is never on that hot path.
- **(d) coalesced tag+byte writes** — savings on `IOBuffer` are at the noise
  floor. Would help `IOStream`/Sockets.
- **(e) internal IO buffering** — same reason as (d).
- **(i) eliminate `RefValue{Cint}` in `IdDict.setindex!`** — sidestepped by
  (h), which avoids the `IdDict` path for serialize entirely.
- `deserialize(s, ::Type{Union})` specialization tried, reverted: `Union{a, b}`
  at runtime is ~3.5× slower than the `jl_new_structv` path the generic
  `deserialize(s, t::DataType)` already uses.
- `handle_deserialize` elseif reorder tried, reverted: branch-prediction
  already handles the chain well; reorder was at the noise floor.
- `pending_refs` swap to `Memory{Int}` stack — `push!`/`pop!` on `Vector{Int}`
  microbenched at the same speed as a manual `Memory`-backed stack.
- `write(::IO, ::UInt16)` `Ref` alloc avoidance — explored stack-buffer +
  `unsafe_write` and split-byte writes; both end up slower than the existing
  `Ref(x)` path on `IOBuffer`.

## History

An earlier intermediate `(c) IdDict-based sertag` step landed first as a
standalone commit, then was fully replaced by the flat-table commit. The
post-rebase history goes straight from the linear scan to the flat table —
the IdDict version was a stepping stone with no value at HEAD.

A final cleanup commit dropped the `table::IdDict{Any,Any}` field from the
`Serializer` struct. After the cycle-table and `deserialize_table`
overrides, no `Serializer` method read the field — only the
`AbstractSerializer` fallback bodies did, and those fire for subtypes
(e.g. `ClusterSerializer`) which carry their own `table` field. The
constructor's `IdDict()` was a ~280-byte wasted allocation per
`Serializer`. Dropping the field eliminated the residual sub-µs
"regression" the regression-hunt suite still showed against baseline —
single-`Int` and short-tuple roundtrips are now 0.77-0.87× baseline
(see `REGRESSION_HUNT.md`).
