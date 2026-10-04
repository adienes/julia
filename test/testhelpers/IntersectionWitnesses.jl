# This file is a part of Julia. License is MIT: https://julialang.org/license

# Random pairs of types for checking that `typeintersect(A, B)` contains every type
# that lies in both `A` and `B`. `B` generalizes a type in `A`, so pairs usually
# intersect, and witnesses are drawn by instantiating the variables of both.
module IntersectionWitnesses

using Random

const LEAVES = Any[Int, Int8, Float64, String, Nothing, Missing, Integer, Real, Number, Signed, Any,
                   Union{Missing,Int}, Union{Nothing,Int}, Union{Int,String}, Union{Missing,Float64}]
const VALUES = Any[Int, Int8, Float64, String, Nothing, Missing, Union{Missing,Int}, Union{Nothing,Int},
                   Union{Int,String}, Union{Missing,Float64}, Integer, Real, Number, Any, Tuple{Int,Int},
                   Union{Nothing,Tuple{Int,Int}}, Union{Missing,Nothing}, Signed, Complex{Int}]
const BOUNDS = Any[Any, Integer, Real, Number, Signed, Union{Missing,Int}, Union{Nothing,Integer},
                   (Union{Missing,U} where U<:Number), (Union{Nothing,U} where U<:Tuple{Int,Int}),
                   Union{Int,String}, Union{Missing,Nothing,Int}]

function gen(rng, depth, vars)
    r = rand(rng, 1:(depth <= 0 ? 3 : 11))
    if r == 1 || (isempty(vars) && r <= 3)
        return rand(rng, LEAVES)
    elseif r <= 3
        return rand(rng, vars)
    elseif r == 4
        return Union{rand(rng, (Missing, Nothing, String)), gen(rng, depth - 1, vars)}
    elseif r <= 6
        return Ref{gen(rng, depth - 1, vars)}
    elseif r == 7
        return Vector{gen(rng, depth - 1, vars)}
    elseif r == 8
        return Tuple{gen(rng, depth - 1, vars), gen(rng, depth - 1, vars)}
    elseif r == 9
        return Vector{Union{rand(rng, (Missing, Nothing)), gen(rng, depth - 1, vars)}}
    elseif r == 10
        p = gen(rng, depth - 1, vars)
        return p isa TypeVar || p <: Real ? Complex{p} : Ref{p}
    else
        return Tuple{isempty(vars) ? Int : rand(rng, vars), Ref{gen(rng, depth - 1, vars)}}
    end
end

function gentype(rng, names)
    vars = TypeVar[]
    for n in names
        r = rand(rng, 1:8)
        ub = r <= 5 ? rand(rng, BOUNDS) : r == 6 && !isempty(vars) ? rand(rng, vars) : rand(rng, LEAVES)
        lb = rand(rng, 1:6) == 1 ? rand(rng, (Int, Int8, Missing, Nothing)) : Union{}
        push!(vars, TypeVar(n, lb, ub))
    end
    return foldr(UnionAll, vars; init = Tuple{gen(rng, 2, vars), gen(rng, 2, vars)})
end

function instances(rng, t, k)
    n = 0
    u = t
    while u isa UnionAll
        n += 1
        u = u.body
    end
    out = Any[]
    for _ in 1:k
        w = try n == 0 ? t : t{(rand(rng, VALUES) for _ in 1:n)...} catch; nothing end
        w === nothing || push!(out, w)
    end
    return out
end

const NAMES = (:X, :Y, :Z, :V)

function generalize(rng, t, cov, env, vars)
    if length(vars) < length(NAMES) && rand(rng) < 0.3
        haskey(env, t) && rand(rng) < 0.7 && return env[t]
        lb = rand(rng) < 0.15 && t isa DataType && isconcretetype(t) ? t : Union{}
        v = TypeVar(NAMES[length(vars) + 1], lb, rand(rng, filter(u -> t <: u, BOUNDS)))
        push!(vars, v)
        env[t] = v
        return v
    end
    if cov && rand(rng) < 0.15
        return rand(rng) < 0.5 ? Union{t, Nothing} : (t isa DataType && t !== Any ? supertype(t) : t)
    end
    t isa Union && return Union{(generalize(rng, m, cov, env, vars) for m in Base.uniontypes(t))...}
    if t isa DataType && !isempty(t.parameters) && !(t <: Type)
        tuple = t <: Tuple
        ps = Any[p isa Type ? generalize(rng, p, tuple && cov, env, vars) : p for p in t.parameters]
        return try t.name.wrapper{ps...} catch; t end
    end
    return t
end

function generalization(rng, w)
    vars = TypeVar[]
    b = generalize(rng, w, true, IdDict{Any,Any}(), vars)
    return foldr(UnionAll, vars; init = b)
end

# The `(A, B, W, order)` for which `W <: A`, `W <: B` but `W` is missing from the intersection.
function missed_witnesses(seed, n)
    rng = Xoshiro(seed)
    missed = Any[]
    for _ in 1:n
        A = gentype(rng, (:T, :S)[1:rand(rng, 1:2)])
        seeds = filter(w -> w <: A, instances(rng, A, 4))
        isempty(seeds) && continue
        B = generalization(rng, seeds[1])
        W = vcat(seeds, instances(rng, A, 10), instances(rng, B, 10))
        ab = typeintersect(A, B)
        ba = typeintersect(B, A)
        for w in W
            (w <: A && w <: B) || continue
            w <: ab || push!(missed, (A, B, w, :AB))
            w <: ba || push!(missed, (A, B, w, :BA))
        end
    end
    return missed
end

end
