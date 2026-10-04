using Serialization, Test, LinearAlgebra, SparseArrays, Statistics, Dates, Printf
pairs = Any[(p[1], p[2]) for p in deserialize(joinpath(@__DIR__, "quickpairs.jls"))]
for (a, b) in deserialize(joinpath(@__DIR__, "queries.jls")); push!(pairs, (a, b)); end
mode = haskey(ENV, "JULIA_TYPE_MEET") ? "port" : "old"
for rnd in 1:3
    t0 = time(); for (a, b) in pairs; typeintersect(a, b); end; t1 = time() - t0
    t0 = time(); for (a, b) in pairs; ccall(:jl_type_intersection_with_env, Any, (Any, Any), a, b); end; t2 = time() - t0
    println(mode, " round $rnd over ", length(pairs), " pairs: typeintersect ", round(t1, digits = 3), " s, with env ", round(t2, digits = 3), " s")
end
