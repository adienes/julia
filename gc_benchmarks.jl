#!/usr/bin/env julia
# gc_benchmarks.jl - Comprehensive GC benchmark suite for finding regressions.
#
# Each benchmark is tagged by category and exercises a different facet of GC
# behavior: allocation patterns, reference graphs, mutation, generational
# tenure, library-style workloads, and pathological cases.
#
# Usage:
#   julia gc_benchmarks.jl [output.tsv]
#
# Environment:
#   GCBENCH_RUNS=N        runs per benchmark (default 5)
#   GCBENCH_WARMUP=N      warmup runs (default 1)
#   GCBENCH_FILTER=regex  only run benchmarks whose name matches
#
# Output: TSV with columns
#   category bench time_ms_median time_ms_min time_ms_max gc_ms_median max_heap_mb fulls runs
#
# Designed to be cross-version compatible (Julia 1.9+); falls back gracefully
# if `Base.full_sweep_reasons` is unavailable.

using Statistics
using Printf
using Random

const RUNS = parse(Int, get(ENV, "GCBENCH_RUNS", "5"))
const WARMUP = parse(Int, get(ENV, "GCBENCH_WARMUP", "1"))
const FILTER = get(ENV, "GCBENCH_FILTER", "")

const BENCHMARKS = Tuple{String, String, Function}[]

# Register a benchmark with a category and name.
register!(category::String, name::String, f) = push!(BENCHMARKS, (category, name, f))

# Run one benchmark, return aggregated stats. `@nospecialize f` is critical:
# without it, the compiler specializes this function for each concrete lambda
# type and may inline-and-constant-fold pure-looking benchmark bodies (e.g.
# allocate-a-tree-then-walk-it returning the same Int every time), reporting
# ~0 ns even when the body actually allocates hundreds of MB.
function run_one(name::String, @nospecialize(f))
    # Warmup
    for _ in 1:WARMUP
        f()
    end
    GC.gc(true); GC.gc(true)

    times = Float64[]
    gc_times = Float64[]
    max_heap_bytes = Int64(0)
    fulls_total = 0

    for _ in 1:RUNS
        gc1 = Base.gc_num()
        t0 = time_ns()
        Base.invokelatest(f)  # invokelatest also blocks specialization
        t = time_ns() - t0
        gc2 = Base.gc_num()
        push!(times, t / 1e6)  # ms
        push!(gc_times, (gc2.total_time - gc1.total_time) / 1e6)
        max_heap_bytes = max(max_heap_bytes, gc2.max_memory)
        fulls_total += gc2.full_sweep - gc1.full_sweep
    end
    return (
        time_median = median(times),
        time_min = minimum(times),
        time_max = maximum(times),
        gc_median = median(gc_times),
        max_heap_mb = max_heap_bytes / (1 << 20),
        fulls = fulls_total,
        runs = RUNS,
    )
end

# ----------------------------------------------------------------------------
# Category: allocation-patterns
# ----------------------------------------------------------------------------

# Many small short-lived allocations: should never trigger fulls.
register!("alloc-patterns", "small-short-lived", function ()
    s = 0.0
    for i in 1:1_000_000
        a = [Float64(i), Float64(i+1), Float64(i+2)]
        s += a[1] + a[2] + a[3]
    end
    s
end)

# Many small long-lived allocations: builds up old-gen fast.
register!("alloc-patterns", "small-long-lived", function ()
    keep = Vector{Vector{Int}}()
    for i in 1:200_000
        push!(keep, collect(i:i+5))
    end
    sum(length, keep)
end)

# Few large short-lived bigvals: stresses bigval sweep.
register!("alloc-patterns", "big-short-lived", function ()
    s = 0.0
    for _ in 1:50
        a = Vector{Float64}(undef, 1_000_000)
        fill!(a, 0.5)
        s += sum(a)
    end
    s
end)

# Few large long-lived bigvals: builds up old-gen bigval list.
register!("alloc-patterns", "big-long-lived", function ()
    keep = Vector{Vector{Float64}}()
    for _ in 1:50
        push!(keep, randn(100_000))
    end
    sum(sum, keep)
end)

# Steady-state: allocate at a constant rate, drop oldest.
register!("alloc-patterns", "rolling-window", function ()
    window = Vector{Vector{Int}}()
    for i in 1:200_000
        push!(window, collect(1:rand(1:50)))
        length(window) > 1000 && popfirst!(window)
    end
    length(window)
end)

# Bursty: alternating heavy and light phases.
register!("alloc-patterns", "bursty", function ()
    s = 0
    for phase in 1:20
        if phase % 2 == 0
            # heavy phase
            arr = [randn(1000) for _ in 1:1000]
            s += sum(sum, arr)
        else
            # light phase
            for _ in 1:1000
                s += rand(Int)
            end
        end
    end
    s
end)

# Growing then shrinking heap.
register!("alloc-patterns", "grow-shrink", function ()
    cache = Vector{Vector{Int}}()
    for i in 1:50_000
        push!(cache, collect(1:50))
    end
    # drop most
    cache = cache[1:1000]
    sum(length, cache)
end)

# ----------------------------------------------------------------------------
# Category: reference-graphs
# ----------------------------------------------------------------------------

# Linked list: pointer-chain stress, many small mutable.
mutable struct LNode; v::Int; next::Union{LNode,Nothing}; end
register!("ref-graphs", "linked-list", function ()
    head = LNode(0, nothing)
    for i in 1:5_000_000
        head = LNode(i, head)
    end
    s = 0; cur = head
    while cur !== nothing
        s += cur.v
        cur = cur.next
    end
    s
end)

# Binary tree.
mutable struct TNode; l::Union{TNode,Nothing}; r::Union{TNode,Nothing}; v::Int; end
function build_tree(depth)
    depth == 0 && return TNode(nothing, nothing, depth)
    return TNode(build_tree(depth-1), build_tree(depth-1), depth)
end
function walk_tree(t::Union{TNode,Nothing})
    t === nothing && return 0
    walk_tree(t.l) + walk_tree(t.r) + t.v
end
register!("ref-graphs", "binary-tree", () -> walk_tree(build_tree(22)))

# Wide fan-out: one shared object referenced 10M times.
register!("ref-graphs", "wide-fan-out", function ()
    shared = Ref(42)
    arr = Vector{Ref{Int}}(undef, 10_000_000)
    fill!(arr, shared)
    sum(r[] for r in arr[1:1000])  # actual use to keep alive
end)

# Sparse pointer array (mostly null).
register!("ref-graphs", "sparse-pointers", function ()
    arr = Vector{Union{Nothing,Vector{Int}}}(nothing, 5_000_000)
    for i in 1:1000:length(arr)
        arr[i] = collect(i:i+9)
    end
    sum(sum(x) for x in arr if x !== nothing)
end)

# Dense parametric tuple chain.
register!("ref-graphs", "tuple-chain", function ()
    cur = nothing
    for i in 1:200_000
        cur = (cur, i, [i, i*2])
    end
    s = 0; node = cur
    while node !== nothing
        s += node[2]
        node = node[1]
    end
    s
end)

# ----------------------------------------------------------------------------
# Category: mutation
# ----------------------------------------------------------------------------

# Old → young writes: write-barrier stress.
register!("mutation", "intergen-writes", function ()
    parents = [Ref{Any}(0) for _ in 1:500_000]
    GC.gc(true)  # promote parents to old
    for (i, p) in enumerate(parents)
        p[] = (i, [i, i*2])  # young value into old ref
    end
    sum(p[][1] for p in parents)
end)

# Many small mutable struct mutations.
mutable struct MutSlot; v::Int; n::Int; end
register!("mutation", "mutable-fields", function ()
    cache = [MutSlot(0, 0) for _ in 1:500_000]
    for _ in 1:5
        for s in cache
            s.v += 1
            s.n = s.v * 2
        end
    end
    sum(s.v + s.n for s in cache)
end)

# Append-heavy (vector growth).
register!("mutation", "vector-append", function ()
    v = Int[]
    for i in 1:5_000_000
        push!(v, i)
    end
    sum(v)
end)

# Random updates (no allocation per update).
register!("mutation", "random-updates", function ()
    arr = zeros(Int, 1_000_000)
    rng = Random.Xoshiro(42)
    for _ in 1:5_000_000
        arr[rand(rng, 1:length(arr))] += 1
    end
    sum(arr)
end)

# ----------------------------------------------------------------------------
# Category: generational
# ----------------------------------------------------------------------------

# All-young (everything dies fast).
register!("generational", "all-young", function ()
    s = 0
    for i in 1:1_000_000
        a = [i, i+1, i+2]
        s += a[1] + a[2] + a[3]
    end
    s
end)

# All-old (everything persists).
register!("generational", "all-old", function ()
    keep = Vector{Vector{Int}}()
    for i in 1:50_000
        push!(keep, collect(i:i+99))
    end
    sum(length, keep)
end)

# Mixed: persistent backbone + transient churn.
register!("generational", "mixed-persistent-transient", function ()
    persistent = [Dict{String,Int}() for _ in 1:1000]
    for d in persistent
        for k in 1:50
            d[string(k)] = k
        end
    end
    s = 0
    for _ in 1:1000
        # transient
        a = randn(10_000)
        s += sum(a)
    end
    s + length(persistent)
end)

# Generational pressure: rolling cache that surely promotes.
register!("generational", "rolling-promote", function ()
    cache = Vector{Vector{Int}}()
    for i in 1:100_000
        push!(cache, collect(1:rand(1:30)))
        length(cache) > 100 && popfirst!(cache)
    end
    sum(length, cache)
end)

# Sliding window pattern (where aging really helps).
register!("generational", "sliding-window-aging", function ()
    window = Vector{Vector{Float64}}()
    for i in 1:200_000
        push!(window, randn(100))
        length(window) > 50 && popfirst!(window)
    end
    sum(sum, window)
end)

# ----------------------------------------------------------------------------
# Category: real-world-ish
# ----------------------------------------------------------------------------

# Dict with churning string keys (Pkg-load-style).
register!("real-world", "dict-string-keys", function ()
    d = Dict{String,Int}()
    rng = Random.Xoshiro(0)
    for i in 1:500_000
        k = string("key_", rand(rng, 1:100_000))
        d[k] = get(d, k, 0) + 1
    end
    length(d)
end)

# Build a large hash table.
register!("real-world", "hash-build", function ()
    d = Dict{Int,Vector{Float64}}()
    for i in 1:50_000
        d[i] = randn(rand(1:50))
    end
    sum(length, values(d))
end)

# Set operations.
register!("real-world", "set-ops", function ()
    s1 = Set{Int}()
    s2 = Set{Int}()
    for i in 1:200_000
        push!(s1, rand(1:100_000))
        push!(s2, rand(1:100_000))
    end
    length(intersect(s1, s2)) + length(union(s1, s2))
end)

# String concatenation.
register!("real-world", "string-concat", function ()
    s = ""
    for i in 1:5000
        s *= "_" * string(i)
    end
    length(s)
end)

# IOBuffer string building.
register!("real-world", "iobuffer-build", function ()
    io = IOBuffer()
    for i in 1:200_000
        write(io, "_")
        write(io, string(i))
    end
    length(take!(io))
end)

# Symbol churn (interned).
register!("real-world", "symbol-churn", function ()
    syms = Symbol[]
    for i in 1:300_000
        push!(syms, Symbol("s_", rand(1:50_000)))
    end
    length(syms)
end)

# Closure-heavy.
register!("real-world", "closure-build", function ()
    cbs = Function[]
    for i in 1:500_000
        push!(cbs, let i=i; () -> i*2 end)
    end
    sum(c() for c in cbs[1:1000])
end)

# Tasks / channels.
register!("real-world", "task-spawn", function ()
    chs = Channel{Int}[]
    for i in 1:5_000
        ch = Channel{Int}(1)
        @async put!(ch, i)
        push!(chs, ch)
    end
    sum(take!(ch) for ch in chs)
end)

# WeakRef churn.
register!("real-world", "weakref-churn", function ()
    refs = WeakRef[]
    for i in 1:1_000_000
        x = Ref(i)
        push!(refs, WeakRef(x))
    end
    n_alive = count(r -> r.value !== nothing, refs)
    refs = WeakRef[]  # drop
    GC.gc(false)
    n_alive
end)

# Finalizer churn.
mutable struct WithFin; v::Int; end
const FIN_COUNT = Ref(0)
register!("real-world", "finalizer-churn", function ()
    FIN_COUNT[] = 0
    for _ in 1:50_000
        x = WithFin(rand(Int))
        finalizer(_ -> (FIN_COUNT[] += 1; nothing), x)
    end
    GC.gc(true)
    FIN_COUNT[]
end)

# Linear algebra (mostly bigval).
using LinearAlgebra
register!("real-world", "matrix-mul", function ()
    s = 0.0
    for _ in 1:50
        A = randn(200, 200)
        B = randn(200, 200)
        C = A * B
        s += sum(diag(C))
    end
    s
end)

# Sorting (allocates many comparisons + buffer).
register!("real-world", "sort-strings", function ()
    rng = Random.Xoshiro(0)
    strs = [string("k_", rand(rng, 1:1_000_000)) for _ in 1:200_000]
    sort!(strs)
    last(strs)
end)

# Graph BFS (adjacency list, lots of small vectors).
register!("real-world", "graph-bfs", function ()
    n = 100_000
    adj = [Int[] for _ in 1:n]
    rng = Random.Xoshiro(0)
    for _ in 1:600_000
        u, v = rand(rng, 1:n, 2)
        push!(adj[u], v)
        push!(adj[v], u)
    end
    visited = falses(n)
    queue = [1]
    visited[1] = true
    count = 0
    while !isempty(queue)
        u = popfirst!(queue)
        count += 1
        for v in adj[u]
            if !visited[v]
                visited[v] = true
                push!(queue, v)
            end
        end
    end
    count
end)

# ----------------------------------------------------------------------------
# Category: pathological
# ----------------------------------------------------------------------------

# Issue #53018 MWE — the regression we fixed.
register!("pathological", "issue-53018-mwe", function ()
    res = 0.0
    for _ in 1:10
        res += sum(sum.([rand(rand(1:100)) for _ in 1:100_000]))
    end
    res
end)

# Larger Groebner-style MWE.
register!("pathological", "groebner-style", function ()
    for _ in 1:5
        polys = [rand(UInt32, rand(1:100)) for _ in 1:100_000]
        s = 0
        for poly in polys
            new_poly = similar(poly)
            new_poly[1] = one(eltype(poly))
            for i in 2:length(poly)
                new_poly[i] = convert(eltype(poly), i)^3 - new_poly[i-1]
            end
            s += sum(new_poly)
        end
    end
end)

# Forced full-GC churn.
register!("pathological", "explicit-full-gc", function ()
    s = 0.0
    for _ in 1:30
        a = randn(50_000)
        s += sum(a)
        GC.gc(true)
    end
    s
end)

# Forced incremental-GC churn.
register!("pathological", "explicit-incr-gc", function ()
    s = 0.0
    for _ in 1:200
        a = randn(50_000)
        s += sum(a)
        GC.gc(false)
    end
    s
end)

# ----------------------------------------------------------------------------
# Category: many-types (compiler/dispatch stress)
# ----------------------------------------------------------------------------

struct Boxed{T}; v::T; end
register!("many-types", "parametric-mix", function ()
    s = 0.0
    for T in (Int8, Int16, Int32, Int64, Float32, Float64, UInt8, UInt16, UInt32, UInt64)
        for _ in 1:50_000
            x = Boxed{T}(rand(T))
            s += Float64(x.v)
        end
    end
    s
end)

# Any-typed array (boxes everything).
register!("many-types", "any-array", function ()
    arr = Any[]
    for i in 1:500_000
        push!(arr, i)  # autoboxes
    end
    sum(x::Int for x in arr)
end)

# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

function should_run(cat, name)
    isempty(FILTER) && return true
    full = "$cat/$name"
    return occursin(Regex(FILTER), full)
end

function main(args)
    println("=" ^ 76)
    println("Julia GC benchmarks  v$(VERSION)  RUNS=$RUNS  WARMUP=$WARMUP")
    println("=" ^ 76)

    out = stdout
    output_path = isempty(args) ? nothing : args[1]
    if output_path !== nothing
        out = open(output_path, "w")
    end

    println(out, join(["category", "bench", "time_ms_med", "time_ms_min",
                       "time_ms_max", "gc_ms_med", "max_heap_mb", "fulls", "runs"], '\t'))
    flush(out)

    @printf("\n%-18s  %-30s  %-9s  %-9s  %-7s  %-9s  %-6s\n",
            "category", "bench", "time(ms)", "gc(ms)", "fulls", "max_heap_MB", "min..max")
    println("-" ^ 100)

    for (cat, name, f) in BENCHMARKS
        should_run(cat, name) || continue
        try
            r = run_one(name, f)
            @printf("%-18s  %-30s  %9.1f  %9.1f  %7d  %9.0f  %.0f..%.0f\n",
                    cat, name, r.time_median, r.gc_median, r.fulls,
                    r.max_heap_mb, r.time_min, r.time_max)
            print(out, join([cat, name,
                round(r.time_median, digits=2), round(r.time_min, digits=2),
                round(r.time_max, digits=2), round(r.gc_median, digits=2),
                round(r.max_heap_mb, digits=1), r.fulls, r.runs], '\t'))
            println(out)
            flush(out)
        catch e
            @printf("%-18s  %-30s  ERROR: %s\n", cat, name, e)
            println(out, join([cat, name, "ERROR", string(e), "", "", "", "", ""], '\t'))
            flush(out)
        end
    end

    if output_path !== nothing
        close(out)
        println("\nResults written to: $output_path")
    end
    println()
end

if !isinteractive()
    main(ARGS)
end
