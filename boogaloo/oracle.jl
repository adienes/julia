# A finite-universe semantics for type intersection.
#
# A type denotes the set of concrete types below it. Over a finite universe of concrete types
# built from the fuzzer's building blocks, `members(T)` enumerates that set by instantiating
# variables with values from `PARAMS` and covariant abstract slots with types from `CONCRETE`;
# membership is the runtime's `<:` on concrete types. For inputs `A`, `B` and a result `R`:
#
#   missing(R) = concrete types in both inputs that are not below R   (unsound if nonempty)
#   excess(R)  = concrete types below R that are not in both inputs    (imprecise if nonempty)
module Oracle

const LEAVES = Any[Int, Int8, Float64, String, Nothing, Missing, Bool]
const ABSTRACT = Any[Integer, Signed, Real, Number, Any]
const UNIONS = Any[Union{Missing,Int}, Union{Nothing,Int}, Union{Int,String}, Union{Missing,Float64},
    Union{Missing,Nothing}, Union{Nothing,Integer}, Union{Missing,Nothing,Int}, Union{Missing,Number},
    Union{Nothing,Real}, Union{Nothing,Tuple{Int,Int}}, Union{Missing,Integer}]
const SMALL = Any[Int, Real, Any, Missing, Nothing, String, Union{Missing,Int}, Union{Nothing,Int}, Integer]

# values a variable can take: any type, abstract and union ones included
const PARAMS = let ps = Any[LEAVES; ABSTRACT; UNIONS]
    for p in SMALL
        push!(ps, Vector{p}, Ref{p}, Vector{Union{Nothing,p}}, Vector{Union{Missing,p}})
    end
    append!(ps, Any[Tuple{Int,Int}, Complex{Int}, ComplexF64, Tuple{Int,Ref{Int}}, Tuple{Missing,Int}])
    unique(ps)
end

# concrete types for covariant slots (`Ref` is abstract: its members are `Base.RefValue`s)
const CONCRETE = let cs = Any[LEAVES; Complex{Int}; ComplexF64]
    for p in PARAMS
        push!(cs, Base.RefValue{p}, Vector{p})
    end
    for x in Any[Int, Int8, Float64, Missing, Nothing, String], y in Any[Int, Missing, Nothing, Base.RefValue{Int}, Base.RefValue{Union{Missing,Int}}]
        push!(cs, Tuple{x, y})
    end
    filter!(isconcretetype, unique(cs))
end

const VALUES = unique(Any[PARAMS; CONCRETE])

const LIMIT = 3_000

struct TooBig <: Exception end

nvars(@nospecialize t) = t isa UnionAll ? 1 + nvars(t.var.ub) + nvars(t.body) : t isa Union ? nvars(t.a) + nvars(t.b) :
    t isa DataType ? sum(nvars, t.parameters; init = 0) : 0

# The values a variable takes: those in its bounds, evenly thinned to `n`. A variable can be
# any type, so concrete ones (the members of covariant slots) are values too.
function values_for(v::TypeVar, n::Int)
    ps = Any[p for p in VALUES if v.lb <: p && p <: v.ub]
    length(ps) <= n && return ps
    return ps[round.(Int, range(1, length(ps), length = n))]
end

# The concrete types below `t`, as far as the universe reaches. A type with several variables
# instantiates each with fewer values: every member is still a concrete type below `t`.
function members(@nospecialize(t), acc = Set{Any}(), n = max(4, floor(Int, 2000^(1 / max(1, nvars(t))))))
    length(acc) > LIMIT && throw(TooBig())
    time() > get(task_local_storage(), :deadline, Inf) && throw(TooBig())
    if t === Union{}
    elseif t isa Union
        members(t.a, acc, n); members(t.b, acc, n)
    elseif t isa UnionAll
        for p in values_for(t.var, n)
            i = try t{p} catch; continue end
            members(i, acc, n)
        end
    elseif t isa DataType && t <: Tuple
        ps = collect(Any, t.parameters)
        va = !isempty(ps) && Base.isvarargtype(ps[end]) ? pop!(ps) : nothing
        # a Vararg contributes 0 to 2 elements (or its fixed length)
        lengths = va === nothing ? (0:0) : isdefined(va, :N) ? (va.N isa Int ? (va.N:va.N) : (0:2)) : (0:2)
        for k in lengths
            el = va === nothing || !isdefined(va, :T) ? Any : va.T
            elts = Any[ps; fill(el, k)]
            slots = [collect(members(p, Set{Any}(), n)) for p in elts]
            prod(length, slots; init = 1) > LIMIT && throw(TooBig())
            for combo in Iterators.product(slots...)
                push!(acc, Tuple{combo...})
            end
        end
    elseif t isa DataType && isconcretetype(t)
        push!(acc, t)
    elseif t isa DataType || t isa TypeVar
        for c in CONCRETE
            c <: t && push!(acc, c)
        end
    end
    return acc
end

const MEMBERS = IdDict{Any,Any}()
const MEMBERS_LOCK = ReentrantLock()
function cached_members(@nospecialize t)
    r = lock(() -> get(MEMBERS, t, nothing), MEMBERS_LOCK)
    r === nothing || (r === :toobig ? throw(TooBig()) : return r)
    r = try members(t) catch err; err isa TooBig ? :toobig : rethrow() end
    lock(() -> (MEMBERS[t] = r), MEMBERS_LOCK)
    r === :toobig && throw(TooBig())
    return r
end

# `(missing, excess, inboth)` for result `r` of `a ∩ b`, or nothing past the universe limit
function judge(@nospecialize(a), @nospecialize(b), @nospecialize(r); budget = 20.0)
    task_local_storage(:deadline, time() + budget)
    try
        cand = union(cached_members(a), cached_members(b))
        inboth = Set{Any}(g for g in cand if g <: a && g <: b)
        miss = Any[g for g in inboth if !(g <: r)]
        exc = Any[g for g in cached_members(r) if !(g <: a && g <: b)]
        return (miss, exc, inboth)
    catch err
        err isa TooBig && return nothing
        rethrow()
    end
end

end
