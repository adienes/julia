# TP_ABLATE= MODE=save: store every result; MODE=diff: compare with the stored results
using Serialization, Random
include(joinpath(@__DIR__, get(ENV, "TPFILE", "TwoPhase.jl")))
include(joinpath(@__DIR__, "census_pairs.jl"))
include(joinpath(@__DIR__, "..", "test", "testhelpers", "IntersectionWitnesses.jl"))
include(joinpath(@__DIR__, "classify.jl"))
const IW = IntersectionWitnesses
pairs = Any[]
pairsfile = joinpath(@__DIR__, "quickpairs.jls")
if get(ENV, "QUICK", "") != "" && isfile(pairsfile)
    using Test, LinearAlgebra, SparseArrays, Statistics, Dates, Printf
    append!(pairs, deserialize(pairsfile))
else
    quick = get(ENV, "QUICK", "") != ""
    for (i, (k, a, b)) in enumerate(collect_sigs())
        quick && i % 8 != 0 && continue
        push!(pairs, (a, b, Any[])); push!(pairs, (b, a, Any[]))
    end
    for seed in 1:3
        rng = Xoshiro(seed)
        for _ in 1:2000
            A = IW.gentype(rng, (:T, :S)[1:rand(rng, 1:2)])
            seeds = filter(w -> w <: A, IW.instances(rng, A, 4))
            isempty(seeds) && continue
            B = IW.generalization(rng, seeds[1])
            W = filter(w -> w <: A && w <: B, vcat(seeds, IW.instances(rng, A, 10), IW.instances(rng, B, 10)))
            push!(pairs, (A, B, W)); push!(pairs, (B, A, W))
        end
    end
    if get(ENV, "QUERIES", "") != ""
        using Test, LinearAlgebra, SparseArrays, Statistics, Dates, Printf
        for (i, (a, b)) in enumerate(deserialize(joinpath(@__DIR__, ENV["QUERIES"])))
            quick && i > 10000 && break
            push!(pairs, (a, b, Any[]))
        end
    end
    get(ENV, "QUICK", "") != "" && serialize(pairsfile, pairs)
end
function run1(a, b)
    try
        return TwoPhase.intersect(a, b)
    catch err
        return err isa TwoPhase.Unsupported ? :unsupported : :error
    end
end
t0 = time()
res = Any[run1(a, b) for (a, b, _) in pairs]
el = round(time() - t0, digits = 1)
file = joinpath(@__DIR__, get(ENV, "BASEFILE", "ablate_base.jls"))
if get(ENV, "MODE", "diff") == "save"
    serialize(file, res)
    println("saved ", length(res), " results in ", el, " s")
    exit()
end
base = deserialize(file)
c = Dict{Symbol,Int}(); ex = Dict{Symbol,Vector{Any}}()
bump(k) = c[k] = get(c, k, 0) + 1
note(k, x) = (v = get!(ex, k, Any[]); length(v) < 3 && push!(v, x))
rng = Xoshiro(5)
for (i, (a, b, W)) in enumerate(pairs)
    r, s = base[i], res[i]
    (r === s || r isa Type && s isa Type && r == s) && continue
    if r isa Symbol || s isa Symbol
        k = Symbol(r isa Symbol ? r : :ok, :_to_, s isa Symbol ? s : :ok); bump(k); note(k, (a, b, r, s)); continue
    end
    k = r <: s ? :looser : s <: r ? :tighter : :incomparable
    if k !== :looser
        cands = vcat(W, Any[r, typeintersect(a, b), typeintersect(b, a)], class_samples(rng, r, 30), class_samples(rng, typeintersect(a, b), 30))
        any(w -> w <: a && w <: b && !(w <: s), cands) && (k = Symbol(k, :_unsound))
    end
    bump(k); note(k, (a, b, r, s))
end
println("ablate=", get(ENV, "TP_ABLATE", ""), "  time ", el, " s  changed: ", isempty(c) ? "none" : join(["$k=$v" for (k, v) in sort!(collect(c))], " "))
for (k, v) in ex
    println("  --- ", k)
    foreach(x -> println("    A=", first(string(x[1]), 200), "\n    B=", first(string(x[2]), 200), "\n    base=", first(string(x[3]), 200), "\n    ablated=", first(string(x[4]), 200)), v)
end
