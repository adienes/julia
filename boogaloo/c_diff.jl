using Serialization, Test, LinearAlgebra, SparseArrays, Statistics, Dates, Printf
include(joinpath(@__DIR__, "TwoPhase.jl"))
cmeet(a, b) = ccall(:jl_type_meet, Any, (Any, Any), a, b)
pairs = Any[(p[1], p[2]) for p in deserialize(joinpath(@__DIR__, "quickpairs.jls"))]
if isfile(joinpath(@__DIR__, "queries60k.jls"))
    for (a, b) in deserialize(joinpath(@__DIR__, "queries60k.jls"))
        (Base.has_free_typevars(a) || Base.has_free_typevars(b)) && push!(pairs, (a, b))
    end
end
println(length(pairs), " pairs"); flush(stdout)
c = Dict{Symbol,Int}(); ex = Dict{Symbol,Vector{Any}}()
bump(k) = c[k] = get(c, k, 0) + 1
note(k, x) = (v = get!(ex, k, Any[]); length(v) < 8 && push!(v, x))
tt = tc = 0.0
for (i, (a, b)) in enumerate(pairs)
    global tt, tc
    t0 = time(); t = try TwoPhase.intersect(a, b) catch err; err isa TwoPhase.Unsupported ? :unsupported : :error end; tt += time() - t0
    t0 = time(); r = try cmeet(a, b) catch err; (:cerror, sprint(showerror, err)[1:min(end, 200)]) end; tc += time() - t0
    k = r === nothing ? (t === :unsupported ? :both_unsupported : :c_unsupported) :
        r isa Tuple ? :c_error :
        !(t isa Type) ? :tp_not_type :
        r == t ? :same : r <: t ? :c_tighter : t <: r ? :c_looser : :incomparable
    bump(k); k in (:same, :both_unsupported) || note(k, (a, b, t, r))
    i % 5000 == 0 && (println(i, " tp ", round(tt, digits = 1), " c ", round(tc, digits = 1), " ", sort!(collect(c))); flush(stdout))
end
println(sort!(collect(c)))
println("time tp ", round(tt, digits = 1), " c ", round(tc, digits = 1))
for (k, v) in ex
    println("--- ", k)
    for x in v
        println("  A=", first(string(x[1]), 250), "\n  B=", first(string(x[2]), 250), "\n  TP=", first(string(x[3]), 250), "\n  C =", first(string(x[4]), 250))
    end
end
