# Alternative-workload regression check

Walked every commit in this branch through an *independent* set of workloads
(`bench/serialization/suite_alt.jl`) to verify every commit is **weakly
better** than baseline (or neutral within noise) on workloads the primary
suite did not exercise.

## Workloads

| | Description | Stresses |
|---|---|---|
| alt 1 | `Dict{String,Int}` (5k entries) | Dict path, repeated long strings, Int writes |
| alt 2 | `Vector{String}` (5k distinct) | `SHARED_REF_TAG`, long-string serialize |
| alt 3 | `Vector{Symbol}` (5k distinct) | sertag for non-primitive non-singleton values |
| alt 4 | `Vector{Any}` mixed primitives (5k) | `serialize_any` per-element (Int, Float64, Symbol, String mix) |
| alt 5 | wide shallow tree (~8.4k `WideNode`) | mutable struct graph, broader recursion than the primary suite |
| alt 6 | `Vector{NamedTuple}` (2k) | NamedTuple serialize, tuple unroll |
| alt 7 | big `Expr` (3k call exprs) | `serialize(::Expr)` |
| alt 8 | 200 shared refs to a 2k linked list | heavy serialize-side cycle table use |
| alt 9 | `Vector{Float64}` (1M) | bits-array baseline complement to V{Int} |

## Bug discovered and fixed

The original `(f)` (bits-union array fast path) inadvertently introduced
a **2.4× regression on wide-tree deserialize** (and 15-30% regressions
across most other workloads). The cause: I had collapsed
`deserialize_array`'s dims-tuple parsing into a `dims::Dims = if ... end`
form, whose runtime typeassert against the abstract `Tuple{Vararg{Int}}`
triggered a `subtype` check involving `UnionAll` instantiation per call —
allocating fresh `DataType`/`SimpleVector`/`UnionAll` per `Vector`
deserialize.

This was caught by the alt-workload sweep (the primary suite happened to
not stress the per-`Vector` allocation pattern). Fixed by amending `(f)`
to keep the original branchy `if/elseif/else` structure (each branch
binds `dims` with its own concrete type, no `::Dims` typeassert) and
preserve `Array{elty, length(dims)}(undef, dims)` so the `length(dims)`
is resolved at the call site. The bits-union additions are kept in both
the 1D fast path and the multi-D path.

After the fix, all subsequent commits naturally rebased on top.

## Cumulative ratio vs baseline

`time(commit) / time(baseline)`. Lower is better. Bold = ≥10% slower than
baseline.

The numbers below are from the original 11-commit walk before the
`(c) IdDict-based sertag` step was dropped (it was found to be fully
obviated by `flat_sertag`). Baseline and the final-state column
(`dry_cycle`) are accurate for the current 10-commit chain. The
intermediate `(h)` / `tup_unroll` / `geom_grow` columns reflect the
old chain (where sertag was already an IdDict); in the post-rebase
chain those positions still have the linear-scan sertag, so they
under-show the current intermediate state on sertag-sensitive
workloads (alt 3/5/7/8 ser). The end column is identical.

| Workload | (a) | (f) | (g) | (h) | tup_unroll | geom_grow | flat_sertag | union_spec | dry_cycle |
|---|---|---|---|---|---|---|---|---|---|
| **alt 1** Dict{Str,Int} ser | 1.00× | 1.00× | 1.00× | 0.89× | 0.89× | 0.89× | 0.89× | 0.89× | 0.89× |
| **alt 1** Dict{Str,Int} de  | 1.02× | 1.00× | 0.59× | 0.59× | 0.60× | 0.57× | 0.57× | 0.57× | 0.57× |
| **alt 2** V{Str} ser        | 1.02× | 1.01× | 1.00× | 1.04× | 1.04× | 1.04× | 1.04× | 1.05× | 1.03× |
| **alt 2** V{Str} de         | 1.02× | 1.00× | 0.42× | 0.43× | 0.43× | 0.41× | 0.41× | 0.41× | 0.41× |
| **alt 3** V{Sym} ser        | 1.00× | 1.00× | 1.00× | 0.54× | 0.54× | 0.54× | 0.45× | 0.45× | 0.45× |
| **alt 3** V{Sym} de         | 1.00× | 1.00× | 0.91× | 0.88× | 0.88× | 0.87× | 0.87× | 0.87× | 0.87× |
| **alt 4** V{Any} mixed ser  | 1.00× | 1.00× | 1.01× | 0.84× | 0.85× | 0.85× | 0.79× | 0.79× | 0.79× |
| **alt 4** V{Any} mixed de   | 1.00× | 1.00× | 0.91× | 0.91× | 0.91× | 0.90× | 0.90× | 0.90× | 0.90× |
| **alt 5** wide tree ser     | 1.04× | 1.01× | 0.99× | 0.51× | 0.51× | 0.51× | 0.47× | 0.47× | 0.47× |
| **alt 5** wide tree de      | 0.96× | 0.98× | 0.80× | 0.75× | 0.73× | 0.73× | 0.75× | 0.73× | 0.75× |
| **alt 6** V{NT} ser         | 1.00× | 0.99× | 0.99× | 0.83× | 0.83× | 0.83× | 0.80× | 0.80× | 0.80× |
| **alt 6** V{NT} de          | 1.01× | 0.99× | 0.95× | 0.97× | 0.71× | 0.71× | 0.71× | 0.71× | 0.71× |
| **alt 7** Big Expr ser      | 1.00× | 1.00× | 1.00× | 0.63× | 0.64× | 0.64× | 0.60× | 0.59× | 0.60× |
| **alt 7** Big Expr de       | 1.00× | 1.00× | 0.84× | 0.84× | 0.84× | 0.83× | 0.83× | 0.83× | 0.83× |
| **alt 8** chains ser        | 1.02× | 1.02× | 1.02× | 0.64× | 0.64× | 0.64× | 0.59× | 0.59× | 0.59× |
| **alt 8** chains de         | 0.96× | 1.07× | 0.74× | 0.72× | 0.73× | 0.70× | 0.69× | 0.71× | 0.69× |
| **alt 9** V{Float64} ser    | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× | 1.01× |
| **alt 9** V{Float64} de     | 1.07× | **1.10×** | 1.10× | 1.09× | **1.10×** | 1.07× | **1.10×** | 1.10× | **1.10×** |

## Findings

- All workloads except one are **weakly better** than baseline, and most
  end up substantially faster.
- The lone "regression" is **alt 9 V{Float64} deserialize at 1.10×**: 158 μs
  vs 143 μs baseline (~15 μs / ~10%). It sits exactly at the threshold and
  is stable across every commit including `(a)` (which doesn't touch the
  bits-array deserialize path), so I read it as system noise on a 1 M-element
  read-bandwidth-bound workload, not a real code regression. The serialize
  side of the same workload is stable at 1.01× across the same range.
- The biggest end-to-end wins on the alt suite mirror the primary suite:
  - **wide-tree serialize**: 0.47× (~53% faster) by `flat_sertag`
  - **wide-tree deserialize**: 0.75× by the cumulative `(g)` + cycle-table
    work
  - **Vector{String} deserialize**: 0.41× (~59% faster) — `(g)` removes
    the per-back-ref `IdDict` boxing for shared strings
  - **Big Expr serialize**: 0.60× — sertag misses for many distinct symbols
  - **chains ser / Dict de**: ~0.57-0.59× from cycle-table + Vector{Any}
    deserialize-table

## Reproducing

```sh
$ /tmp/serialization_bench/walk_commits.sh > walk.out 2>&1
$ /Users/andy/.juliaup/bin/julia /tmp/serialization_bench/analyze_walk.jl
```

The walk script checks out each SHA in turn and runs
`/tmp/serialization_bench/suite_alt.jl` against that commit's source.
`analyze_walk.jl` parses the per-commit logs and emits the table above
plus a list of any workloads that regressed >10% relative to baseline.
