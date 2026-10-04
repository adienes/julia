# Attribute a difference between the prototype's result `t` and C's `c` for `a ∩ b` to a rule.
using Random
const CLASS_VALUES = Any[Int, Int8, Int64, UInt8, Float64, Float32, BigInt, Bool, Char, String, Symbol, Nothing,
    Missing, Union{Missing,Int}, Union{Nothing,Int}, Integer, Signed, Real, AbstractFloat, Number, Any, Union{},
    Vector{Int}, Tuple{}, Tuple{Int}, DataType, Type{Int}, Core.TypeEgal{Int}, Complex{Int}, Rational{Int},
    AbstractVector{Int}, Matrix{Float64}, UnitRange{Int}, Base.OneTo{Int}, Dict{Int,Int}, SubString{String}, 0, 1, 2, 3]
function class_samples(rng, @nospecialize(t), k)
    t isa Union && return reduce(vcat, (class_samples(rng, m, k) for m in Base.uniontypes(t)); init = Any[])
    t isa UnionAll || return Any[t]
    out = Any[]
    for _ in 1:k
        u = t
        while u isa UnionAll
            v = u.var
            cands = Any[v.lb, v.ub]
            for x in CLASS_VALUES
                (x isa Type || x isa Int) || continue
                x isa Type && (v.lb isa Type && v.ub isa Type) && !(v.lb <: x && x <: v.ub) && continue
                push!(cands, x)
            end
            u = try u{rand(rng, cands)} catch; nothing end
        end
        u isa Type && push!(out, u)
    end
    return out
end
# `t` with each `where` variable narrowed by the closed declared bounds of the parameters it
# fills directly
function declared_narrow(@nospecialize t)
    t isa UnionAll || return t
    v = t.var
    body = declared_narrow(t.body)
    ub = v.ub
    function visit(@nospecialize x)
        if x isa DataType && !(x <: Tuple)
            qs = Base.unwrap_unionall(x.name.wrapper).parameters
            for (p, q) in zip(x.parameters, qs)
                p === v && q isa TypeVar && !Base.has_free_typevars(q.ub) && (ub = typeintersect(ub, q.ub))
                visit(p)
            end
        elseif x isa DataType || x isa Union
            foreach(visit, x isa Union ? Base.uniontypes(x) : x.parameters)
        elseif x isa UnionAll
            visit(x.var.ub); visit(x.body)
        end
    end
    visit(body)
    ub === v.ub && return UnionAll(v, body)
    nv = TypeVar(v.name, v.lb, ub)
    return UnionAll(nv, ccall(:jl_instantiate_unionall, Any, (Any, Any), UnionAll(v, body), nv))
end

function classify(rng, @nospecialize(a), @nospecialize(b), @nospecialize(t), @nospecialize(c))
    t == c && return :same
    for w in class_samples(rng, c, 40)
        w <: a && w <: b && !(w <: t) && return :tp_unsound
    end
    if c <: t
        valid = class_samples(rng, t, 60)
        for w in valid
            w <: a && w <: b && !(w <: c) && return :looser_c_unsound
        end
        dt = try declared_narrow(t) catch; t end
        dt <: c && return :looser_declared
        !isempty(valid) && all(w -> w <: c, valid) && return :looser_nowitness
        return :looser
    end
    t <: c && return :tighter
    tp_only = [w for w in class_samples(rng, t, 80) if !(w <: c)]
    any(w -> w <: a && w <: b, tp_only) && return :incomparable_c_unsound
    isempty(tp_only) && return :tighter_nowitness
    c_only = [w for w in class_samples(rng, c, 80) if !(w <: t)]
    isempty(c_only) && return :looser_nowitness
    return :incomparable
end
