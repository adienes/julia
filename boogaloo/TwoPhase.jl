# A two-phase type intersection.
#
# Every variable the inputs bind is an unknown, and so is every `where` in a covariant position;
# a variable free in the inputs is opaque.
# The intersection is a union of cases: a covariant union is split, a `Vararg` that may be
# empty is dropped or kept, a length variable is pinned to 0 or left free. Each case is solved
# in two phases, and the cases share one `where`.
#
# Phase 1 derives necessary facts from the two types meeting in the slots every member of the
# result has: equalities from invariant positions, subtype relations from bounds and from
# covariant slots holding a type with one member, and emptiness checks from declared parameter
# bounds and slot pairs. `meet!`, `sub!` and `equate!` are the three relations; each handles
# variables, unions and `where`s and hands the remaining structure to `decompose!`. Subtype
# facts are propagated to a fixpoint (`solve!`). Omitting a fact only loosens the result.
#
# Phase 2 builds the meet of the two types from the solved facts (`build`), solving a union
# member or a `where` that cannot move to the top on its own (`build_local`), and closes the
# result over its remaining variables (`wrap`).
#
# Results are sound relative to the runtime's `<:` (a type in both inputs is in the result),
# do not depend on argument order, and use forms `<:` decides; TYPE_PLAN.md states the rules.

module TwoPhase

# ----------------------------------------------------------------------------------------
# Types as data

struct Unsupported <: Exception
    site::Symbol
end
struct Unsat <: Exception end
const TypeEq = typeof(Type{Int})
isvararg(@nospecialize t) = Base.isvarargtype(t)
va_T(@nospecialize v) = isdefined(v, :T) ? v.T : Any
va_N(@nospecialize v) = isdefined(v, :N) ? v.N : nothing
make_va(@nospecialize(T), @nospecialize(N)) = N === nothing ? Vararg{T} : Vararg{T, N}

# the fixed elements of a tuple type and its trailing Vararg, if any
function tuple_parts(t::DataType)
    ps = collect(Any, t.parameters)
    if !isempty(ps) && isvararg(ps[end])
        return ps[1:end-1], ps[end]
    end
    return ps, nothing
end

has_vararg(@nospecialize t) = t isa DataType && t <: Tuple && !isempty(t.parameters) && isvararg(t.parameters[end])
istypeeq(@nospecialize t) = t isa TypeEq
isegal(@nospecialize t) = t isa Core.TypeEgal
# The variables free in the inputs. Each is opaque, as in `<:`: one type, equal only to itself,
# above `Union{}` alone and below only `Any` and the unions that have it; its bounds play no part.
const OPAQUE = TypeVar[]
isopaque(@nospecialize t) = t isa TypeVar && any(o -> o === t, OPAQUE)

# the runtime's `<:`, which also takes an opaque variable (the builtin takes only types)
function (<:)(@nospecialize(a), @nospecialize(b))
    return ccall(:jl_subtype, Cint, (Any, Any), a, b) != 0
end

# no unknowns: no variables but opaque ones
isclosed(@nospecialize t) = !Base.has_free_typevars(t) || !isempty(OPAQUE) && isempty(free_vars!(TypeVar[], t))
is_tuple_type(@nospecialize t) = t isa DataType && t <: Tuple

function unwrap(@nospecialize t)
    vars = TypeVar[]
    while t isa UnionAll
        push!(vars, t.var)
        t = t.body
    end
    return vars, t
end

# the free variables of `t` but the opaque ones
function free_vars!(acc::Vector{TypeVar}, @nospecialize(t), scope::Vector{TypeVar} = TypeVar[])
    if t isa TypeVar
        t in scope || t in acc || isopaque(t) || push!(acc, t)
    elseif t isa Union
        free_vars!(acc, t.a, scope); free_vars!(acc, t.b, scope)
    elseif t isa UnionAll
        free_vars!(acc, t.var.lb, scope); free_vars!(acc, t.var.ub, scope)
        free_vars!(acc, t.body, TypeVar[scope; t.var])
    elseif istypeeq(t) || isegal(t)
        free_vars!(acc, t.T, scope)
    elseif isvararg(t)
        free_vars!(acc, va_T(t), scope); free_vars!(acc, va_N(t), scope)
    elseif t isa DataType
        foreach(p -> free_vars!(acc, p, scope), t.parameters)
    end
    return acc
end

# substitution with the runtime's checking that ignores a declared bound's invalid union member
function inst(@nospecialize(x), v::TypeVar, @nospecialize(val))
    x === v && return val
    Base.has_typevar(x, v) || return x
    r = ccall(:jl_type_meet_inst, Ptr{Cvoid}, (Any, Any, Any), x, v, val)
    r == C_NULL && throw(ArgumentError("invalid instantiation"))
    return unsafe_pointer_to_objref(r)
end

# covariant and invariant occurrence counts of `v` in `t`
function occurrences(@nospecialize(t), v::TypeVar, inv::Bool = false, acc = [0, 0])
    if t === v
        acc[inv ? 2 : 1] += 1
    elseif t isa Union
        foreach(m -> occurrences(m, v, inv, acc), Base.uniontypes(t))
    elseif istypeeq(t)
        occurrences(t.T, v, true, acc)
    elseif isvararg(t)
        occurrences(va_T(t), v, inv, acc)
        occurrences(va_T(t), v, inv, acc)
        va_N(t) === v && (acc[2] += 1)
    elseif t isa DataType
        tuple = t <: Tuple
        foreach(p -> occurrences(p, v, inv || !tuple, acc), t.parameters)
    elseif t isa UnionAll
        occurrences(t.var.lb, v, true, acc); occurrences(t.var.ub, v, true, acc)
        t.var === v || occurrences(t.body, v, inv, acc)
    end
    return acc
end

# occurrences in the body and, invariantly, in the bounds of the other variables
function occurrences(vars::Vector{TypeVar}, @nospecialize(t), v::TypeVar)
    acc = occurrences(t, v)
    for w in vars
        w === v && continue
        occurrences(w.lb, v, true, acc)
        occurrences(w.ub, v, true, acc)
    end
    return acc
end

function is_diagonal(vars::Vector{TypeVar}, @nospecialize(t), v::TypeVar)
    cov, inv = occurrences(vars, t, v)
    return cov > 1 && inv == 0
end

is_concrete_leaf(@nospecialize t) = isopaque(t) || t isa DataType && isconcretetype(t) && !(t <: Type) &&
    (!(t <: Tuple) || all(is_concrete_leaf, t.parameters))

head(@nospecialize t) = istypeeq(t) ? TypeEq : t isa DataType ? t.name : nothing

# `lo` seen at `hi`'s head through its supertypes, or nothing
function common_head(lo::DataType, hi::DataType)
    s = lo
    while s !== Any && s.name !== hi.name
        s = supertype(s)
    end
    return s === Any && hi !== Any ? nothing : s
end

# the variables of `t` and those their bounds mention
function deps(@nospecialize t)
    acc = free_vars!(TypeVar[], t)
    i = 1
    while i <= length(acc)
        free_vars!(acc, acc[i].lb)
        free_vars!(acc, acc[i].ub)
        i += 1
    end
    return acc
end

# The fixed slots of two tuples where splitting unions into cases separates nothing: closed in
# both; not a tuple on either side and with variables that occur in no other slot (a tuple's
# elements can share a variable); or closed against a diagonal variable, which is one of its
# members whether split or not.
function independent_slots(@nospecialize(a), @nospecialize(b), diagonal)
    is_tuple_type(a) && is_tuple_type(b) || return Int[]
    pa, pb = a.parameters, b.parameters
    n = min(length(tuple_parts(a)[1]), length(tuple_parts(b)[1]))
    ds = [TypeVar[i <= length(pa) ? deps(pa[i]) : TypeVar[]; i <= length(pb) ? deps(pb[i]) : TypeVar[]]
          for i in 1:max(length(pa), length(pb))]
    has_tuple(@nospecialize t) = any(m -> Base.unwrap_unionall(m) isa DataType && Base.unwrap_unionall(m) <: Tuple, Base.uniontypes(t))
    diag(@nospecialize t) = t isa TypeVar && get(diagonal, t, false)
    return Int[i for i in 1:n if isempty(ds[i]) ||
               isclosed(pa[i]) && diag(pb[i]) || isclosed(pb[i]) && diag(pa[i]) ||
               !has_tuple(pa[i]) && !has_tuple(pb[i]) && all(j -> j == i || !any(v -> v in ds[j], ds[i]), eachindex(ds))]
end

# the bounds of `T` in a closed `Type{X}` (`X` itself) or `Type{T} where lb<:T<:ub`, or nothing
function type_range(@nospecialize t)
    isclosed(t) || return nothing
    istypeeq(t) && return (t.T, t.T)
    t isa UnionAll && istypeeq(t.body) && t.body.T === t.var || return nothing
    return (t.var.lb, t.var.ub)
end

# the declared upper bound of a variable, through bounds that are variables
upper(@nospecialize t) = (while t isa TypeVar; t = t.ub; end; t)

# nominal types share a member only along one supertype chain
function nominally_disjoint(@nospecialize(a), @nospecialize(b))
    a, b = Base.unwrap_unionall(a), Base.unwrap_unionall(b)
    a isa Union && return nominally_disjoint(a.a, b) && nominally_disjoint(a.b, b)
    b isa Union && return nominally_disjoint(a, b.a) && nominally_disjoint(a, b.b)
    a isa DataType && b isa DataType && !(a <: Tuple) && !(b <: Tuple) || return false
    return common_head(a, b) === nothing && common_head(b, a) === nothing
end

function check_supported(@nospecialize(t), vars)
    if t isa TypeVar
        check_supported(t.lb, vars); check_supported(t.ub, vars)
    elseif t isa Union
        foreach(m -> check_supported(m, vars), Base.uniontypes(t))
    elseif t isa UnionAll
        check_supported(t.var.lb, vars)
        check_supported(t.var.ub, vars)
        check_supported(t.body, TypeVar[vars; t.var])
    elseif istypeeq(t)
        check_supported(t.T, vars)
    elseif isegal(t)
    elseif t isa DataType
        for (i, p) in enumerate(t.parameters)
            if isvararg(p)
                (t <: Tuple && i == length(t.parameters)) || throw(Unsupported(:vararg_position))
                check_supported(va_T(p), vars)
                n = va_N(p)
                n === nothing || n isa Int || n isa TypeVar || throw(Unsupported(:vararg_length))
                continue
            end
            (p isa Type || p isa TypeVar) && check_supported(p, vars)
        end
    elseif t !== Union{}
        throw(Unsupported(:not_a_type))
    end
end

# ----------------------------------------------------------------------------------------
# Unknowns

mutable struct Binding
    lb::Any
    ub::Any          # declared or derived bound, possibly mentioning other variables
    eq::Any          # nothing, or a type this variable equals
end

mutable struct Env
    b::IdDict{TypeVar,Binding}
    subs::Vector{Tuple{Any,Any}}
    changed::Bool
    diagonal::IdDict{TypeVar,Bool}
    locals::Vector{TypeVar}             # never the representative of an equality with an outer one
    depth::Int                          # nesting of local solves
    owner::IdDict{TypeVar,TypeVar}      # a variable occurring only in the bounds of one other
    checks::Vector{Tuple{Any,Any,Any}}  # declared parameter bounds (lb, parameter, ub)
    slots::Vector{Tuple{Any,Any}}       # covariant slot pairs that every member has
    meets::Vector{Tuple{Any,Any}}       # pairs that share a member, applied again every round
    lifts::Vector{Tuple{Symbol,Any,Any,Any}}  # `(rel, t, u, body)`: `u` lifted for the fact `t rel u`
    version::Int                        # number of binding changes
    met::Int                            # `version` when the meets were last applied
    split::Bool                         # a kept Vararg has elements: the case was split on it
    uses::IdDict{TypeVar,Tuple{Int,Int}}  # covariant and invariant occurrences of each input variable
    lengths::Vector{TypeVar}            # input variables used as a Vararg length
    mins::IdDict{TypeVar,Int}           # a length variable is at least this
    offsets::IdDict{TypeVar,Tuple{Any,Int}}  # a length variable is another one plus this
    inhabited::Vector{TypeVar}          # variables that are a tuple element every member has
end

# a fresh environment
Env(diagonal, owner = IdDict{TypeVar,TypeVar}()) =
    Env(IdDict{TypeVar,Binding}(), Tuple{Any,Any}[], false, diagonal, TypeVar[], 0, owner,
        Tuple{Any,Any,Any}[], Tuple{Any,Any}[], Tuple{Any,Any}[], Tuple{Symbol,Any,Any,Any}[], 0, 0, true,
        IdDict{TypeVar,Tuple{Int,Int}}(), TypeVar[], IdDict{TypeVar,Int}(), IdDict{TypeVar,Tuple{Any,Int}}(), TypeVar[])

local_env(env) = Env(IdDict{TypeVar,Binding}(k => Binding(b.lb, b.ub, b.eq) for (k, b) in env.b),
                     copy(env.subs), false, env.diagonal, TypeVar[], env.depth + 1, env.owner,
                     copy(env.checks), Tuple{Any,Any}[], Tuple{Any,Any}[], copy(env.lifts), 0, 0, env.split, env.uses,
                     env.lengths, copy(env.mins), copy(env.offsets), env.inhabited)

new_vars(sub, env) = TypeVar[x for x in keys(sub.b) if !haskey(env.b, x)]
bound(env, @nospecialize t) = t isa TypeVar && haskey(env.b, t)

function resolve(env, @nospecialize t)
    n = 0
    while bound(env, t) && env.b[t].eq !== nothing && n < 64
        t = env.b[t].eq
        n += 1
    end
    return t
end

# Replace bound variables by what they are known to equal. A union member whose parameters
# break their declared bounds is empty, unless the union is inside an invariant parameter.
function subst(env, @nospecialize(t), inv::Bool = false)
    isclosed(t) && return t
    if t isa TypeVar
        r = resolve(env, t)
        r === t && return t
        return subst(env, r, inv)
    elseif t isa Union
        ms = Any[]
        for m in Base.uniontypes(t)
            r = try subst(env, m, inv) catch err; err isa Unsat ? nothing : rethrow() end
            # a member that is not a type, or whose parameters break their declared bounds, is empty
            r isa Type || r isa TypeVar || (inv ? throw(Unsat()) : continue)
            push!(ms, r)
        end
        return Union{ms...}
    elseif istypeeq(t)
        return Type{subst(env, t.T, true)}
    elseif isvararg(t)
        n = va_N(t)
        T, n = subst(env, va_T(t), inv), n === nothing ? nothing : subst(env, n, true)
        return try make_va(T, n) catch; throw(Unsat()) end
    elseif t isa DataType
        ps = Any[subst(env, p, inv || !(t <: Tuple)) for p in t.parameters]
        # the runtime's checking without throwing: a declared bound's invalid union member is ignored
        r = ccall(:jl_type_meet_apply, Ptr{Cvoid}, (Any, Any), t, Core.svec(ps...))
        r == C_NULL && throw(Unsat())
        return unsafe_pointer_to_objref(r)
    elseif t isa UnionAll
        lb, ub = subst(env, t.var.lb, inv), subst(env, t.var.ub, inv)
        lb === t.var.lb && ub === t.var.ub && return UnionAll(t.var, subst(env, t.body, inv))
        nv = TypeVar(t.var.name, lb, ub)
        return UnionAll(nv, subst(env, inst(t.body, t.var, nv), inv))
    end
    return t
end

occurs(v::TypeVar, @nospecialize t) = t !== v && Base.has_typevar(t, v)

# a binding changed
touched!(env) = (env.changed = true; env.version += 1; nothing)

# record `lo <: hi`; a new fact is a change
function fact!(env, @nospecialize(lo), @nospecialize(hi))
    f = (lo, hi)
    f in env.subs && return
    push!(env.subs, f)
    env.changed = true
end

# the representative of an equality between variables has the highest rank
rank(env, v::TypeVar) = v in env.locals ? 0 : haskey(env.owner, v) ? 1 : 2

# A preference between two types the facts make equal, independent of argument order: the
# closed one, then the one with fewer bare variables in unions, then the one whose variables are
# tied to more of the inputs (invariant occurrences, then all occurrences: its correlations with
# the rest of the type are what the facts could not express, and a variable occurring once
# covariantly becomes its bound), then the smaller one, then the first in `type_order`.
function canonical(env, @nospecialize(x), @nospecialize(y))
    isclosed(x) && return x
    isclosed(y) && return y
    (ix, ox), (iy, oy) = shared(env, x), shared(env, y)
    kx = (bare_vars(x), -ix, -ox, type_size(x))
    ky = (bare_vars(y), -iy, -oy, type_size(y))
    kx != ky && return kx < ky ? x : y
    return type_order(x, y) <= 0 ? x : y
end

# the invariant and the total occurrences in the inputs of the variables of `t`
function shared(env, @nospecialize t)
    inv = tot = 0
    for v in free_vars!(TypeVar[], t)
        c, i = get(env.uses, v, (0, 0))
        inv += i
        tot += c + i
    end
    return inv, tot
end

bare_vars(@nospecialize t) = t isa Union ? sum(m -> m isa TypeVar ? 1 : bare_vars(m), Base.uniontypes(t)) :
    t isa DataType ? sum(bare_vars, t.parameters; init = 0) : 0

function type_size(@nospecialize t)
    t isa Union && return 1 + type_size(t.a) + type_size(t.b)
    t isa UnionAll && return 1 + type_size(t.var.ub) + type_size(t.body)
    t isa DataType && return 1 + sum(type_size, t.parameters; init = 0)
    (istypeeq(t) || isegal(t)) && return 1 + type_size(t.T)
    isvararg(t) && return 1 + type_size(va_T(t))
    return 1
end

# A total order on types by structure: kind, then name, then parts.
function type_order(@nospecialize(x), @nospecialize(y))
    rank(@nospecialize t) = t isa TypeVar ? 0 : t isa DataType ? 1 : t isa Union ? 2 : t isa UnionAll ? 3 :
        isvararg(t) ? 4 : 5
    rx, ry = rank(x), rank(y)
    rx != ry && return cmp(rx, ry)
    if x isa TypeVar
        c = cmp(x.name, y.name)
        c == 0 || return c
        c = type_order(x.lb, y.lb)
        return c == 0 ? type_order(x.ub, y.ub) : c
    elseif x isa DataType
        c = cmp((string(x.name.module), x.name.name), (string(y.name.module), y.name.name))
        c == 0 || return c
        c = cmp(length(x.parameters), length(y.parameters))
        c == 0 || return c
        for (p, q) in zip(x.parameters, y.parameters)
            c = type_order(p, q)
            c == 0 || return c
        end
        return 0
    elseif x isa Union
        c = type_order(x.a, y.a)
        return c == 0 ? type_order(x.b, y.b) : c
    elseif x isa UnionAll
        c = type_order(x.var, y.var)
        return c == 0 ? type_order(x.body, y.body) : c
    elseif isvararg(x)
        c = type_order(va_T(x), va_T(y))
        return c == 0 ? type_order(something(va_N(x), -1), something(va_N(y), -1)) : c
    end
    x == y && return 0
    return cmp(repr(x), repr(y))
end

function bind_eq!(env, v::TypeVar, @nospecialize t)
    t = subst(env, t)
    # a union that has `v` as a member equals `v` when its other members are below `v`
    if t isa Union && any(m -> m === v, Base.uniontypes(t))
        fact!(env, Union{(m for m in Base.uniontypes(t) if m !== v)...}, v)
        return
    end
    occurs(v, t) && throw(Unsat())
    m = get(env.mins, v, 0)
    m > 0 && (t isa Int || t isa TypeVar) && at_least!(env, t, m)
    bnd = env.b[v]
    bnd.eq = t
    touched!(env)
    fact!(env, bnd.lb, t)
    fact!(env, t, bnd.ub)
    add_param_bounds!(env, t)
end

function set_ub!(env, v, @nospecialize t)
    bnd = env.b[v]
    nu = isclosed(bnd.ub) ? meet_closed(bnd.ub, t) : closedize(env, bnd.ub, true) <: t ? bnd.ub : t
    # a variable that is a tuple element every member has is not empty
    nu === Union{} && v in env.inhabited && throw(Unsat())
    if nu != bnd.ub
        bnd.ub = nu
        touched!(env)
    end
end

function set_lb!(env, v, @nospecialize t)
    bnd = env.b[v]
    nl = bnd.lb === Union{} ? t : Union{bnd.lb, t}
    if nl != bnd.lb
        bnd.lb = nl
        touched!(env)
    end
end

same_type(@nospecialize(x), @nospecialize(y)) = x === y || x isa Type && y isa Type && x == y

# The body of the `where` `u` lifted for the fact `t rel u` (`:sub`: `t <: u`; `:meet`: they
# share a member). One fact has one witness for `u`'s variable, however often it is applied.
function lifted!(env, rel::Symbol, @nospecialize(t), u::UnionAll)
    for (r, l, h, body) in env.lifts
        r === rel && same_type(l, t) && same_type(h, u) && return body
    end
    body = lift!(env, u)
    push!(env.lifts, (rel, t, u, body))
    return body
end

# An existential `where` as a fresh unknown of the environment; returns the body.
function lift!(env, u::UnionAll)
    u = Base.rename_unionall(u)
    v = u.var
    env.b[v] = Binding(v.lb, v.ub, nothing)
    cov, inv = occurrences(u.body, v)
    cov > 1 && inv == 0 && (env.diagonal[v] = true)
    v.lb === Union{} || fact!(env, v.lb, v)
    v.ub === Any || fact!(env, v, v.ub)
    return u.body
end

# Declared parameter bounds of a datatype are necessary for it to exist. They only decide
# emptiness: C's subtype ignores them, so a result narrowed by them is not comparable with
# the same type spelled without them.
function add_param_bounds!(env, @nospecialize t)
    t isa DataType || return
    isclosed(t) && return
    if !(t <: Tuple)
        qs = Base.unwrap_unionall(t.name.wrapper).parameters
        for (p, q) in zip(t.parameters, qs)
            q isa TypeVar || continue
            (isclosed(q.ub) && q.ub !== Any || isclosed(q.lb) && q.lb !== Union{}) &&
                push!(env.checks, (isclosed(q.lb) ? q.lb : Union{}, p, isclosed(q.ub) ? q.ub : Any))
        end
    end
    foreach(p -> add_param_bounds!(env, p), t.parameters)
end

# ----------------------------------------------------------------------------------------
# Closed approximations

# `t` with its variables as `where`s over their closed bounds, keeping their correlation;
# nothing where that would make a variable diagonal that is not
function rewrap_closed(env, @nospecialize(t), depth)
    # a type whose parameters break its declared bounds is empty
    r = try subst(env, t) catch err; err isa Unsat ? (return Union{}) : rethrow() end
    for w in free_vars!(TypeVar[], r)
        cov, inv = occurrences(r, w)
        cov > 1 && inv == 0 && !get(env.diagonal, w, false) && return nothing
        lb = bound(env, w) ? closedize(env, w, false, depth + 1) : closedize(env, w.lb, false, depth + 1)
        ub = bound(env, w) ? closedize(env, w, true, depth + 1) : closedize(env, w.ub, true, depth + 1)
        lb <: ub || return Union{}
        nv = TypeVar(w.name, lb, ub)
        r = UnionAll(nv, inst(r, w, nv))
    end
    return r
end

# A closed over-approximation (upper) or under-approximation (lower) of `t`.
function closedize(env, @nospecialize(t), upper::Bool, depth = 0)
    isclosed(t) && return t
    depth > 32 && return upper ? Any : Union{}
    if t isa TypeVar
        r = resolve(env, t)
        r !== t && return closedize(env, r, upper, depth + 1)
        # a variable of an enclosing `where` stands for some type within its bounds
        bound(env, t) || return closedize(env, upper ? t.ub : t.lb, upper, depth + 1)
        return closedize(env, upper ? env.b[t].ub : env.b[t].lb, upper, depth + 1)
    elseif t isa Union
        ms = Any[]
        for m in Base.uniontypes(t)
            c = closedize(env, m, upper, depth + 1)
            c === nothing && return nothing
            # a member that is a value, not a type, is empty
            (c isa Type || c isa TypeVar) && push!(ms, c)
        end
        return Union{ms...}
    elseif istypeeq(t) || t isa DataType && !(t <: Tuple)
        # invariant positions: the variables range over their closed bounds together
        return upper ? rewrap_closed(env, t, depth) : Union{}
    elseif has_vararg(t)
        fixed, va = tuple_parts(t)
        fs = Any[closedize(env, p, upper, depth + 1) for p in fixed]
        any(==(Union{}), fs) && return Union{}
        n = va_N(va)
        n isa TypeVar && (n = resolve(env, n))
        # a negative length is no tuple
        n isa Int && n < 0 && return Union{}
        T = closedize(env, va_T(va), upper, depth + 1)
        if upper
            return Tuple{fs..., make_va(T, n isa Int ? n : nothing)}
        elseif n === nothing
            return Tuple{fs...}
        elseif n isa Int
            n > 0 && T === Union{} && return Union{}
            return Tuple{fs..., Vararg{T, n}}
        end
        return Union{}
    elseif t isa DataType && t <: Tuple
        ps = Any[closedize(env, p, upper, depth + 1) for p in t.parameters]
        any(==(Union{}), ps) && return Union{}
        return Tuple{ps...}
    elseif t isa UnionAll
        upper || return Union{}
        r = rewrap_closed(env, t, depth)
        return r === nothing ? closedize(env, t.body, true, depth + 1) : r
    end
    return upper ? Any : Union{}
end

# The meet of two closed types; `a` when the meet is already being computed (it is a
# superset of the meet).
function meet_closed(@nospecialize(a), @nospecialize(b))
    a === Any && return b
    b === Any && return a
    r = meet_or_nothing(a, b)
    return r === nothing ? a : r
end

# false when the meet is already being computed: the check adds nothing then
disjoint(@nospecialize(a), @nospecialize(b)) = meet_or_nothing(a, b) === Union{}

# A nominal type, `Type{X}` and an opaque variable are never empty, and a tuple type is empty
# only when one of its fixed elements is.
never_empty(@nospecialize t) = istypeeq(t) || isegal(t) ||
    t isa DataType && (!(t <: Tuple) || all(p -> isvararg(p) || never_empty(p), t.parameters)) ||
    isclosed(t) && t !== Union{}

# the kind of every value of a type expression, or nothing for a variable (an opaque one too)
kind_of(@nospecialize t) = isopaque(t) ? nothing : isclosed(t) ? typeof(t) : istypeeq(t) ? TypeEq :
    isegal(t) ? Core.TypeEgal : t isa DataType ? DataType : t isa Union ? Union : t isa UnionAll ? UnionAll : nothing

# `Type{X}` (any type object equal to `X`) or `TypeEgal{X}` (`X` itself) has a value of kind `k`;
# an opaque variable is a type of any kind but `Union{}`'s
function in_kind(@nospecialize(s), @nospecialize(k))
    X = s.T
    isopaque(k) && return false
    isopaque(X) && return k <: Type && k !== Core.TypeofBottom
    return istypeeq(s) ? meets_kind(X, k) : X isa k
end
meets_kind(@nospecialize(X), @nospecialize(k)) = ccall(:jl_typeeq_meets_kind, Cint, (Any, Any), X, k) != 0

const KINDS = Any[DataType, UnionAll, Union, Core.TypeofBottom, TypeEq, Core.TypeEgal]

# the kinds of the members of a closed `Type{X}`, `TypeEgal{X}` or union of them, or nothing
function member_kinds(@nospecialize t)
    ms = Base.uniontypes(t)
    all(m -> istypeeq(m) || isegal(m), ms) || return nothing
    ks = Any[]
    for m in ms, k in KINDS
        in_kind(m, k) && !(k in ks) && push!(ks, k)
    end
    return Union{ks...}
end

# A type object equal to an instance of the open type `X` has a kind in `k`: one equal to a
# data type is a data type or a `where` pinning its parameters, to a tuple also a union of
# tuples, to `Type{Y}` a `TypeEq` or `TypeofBottom`; a variable can be anything.
function open_kind_meets(@nospecialize(X), @nospecialize(k))
    ks = istypeeq(X) ? (TypeEq, DataType) :
        X isa DataType ? (X <: Tuple ? (DataType, UnionAll, Union) : (DataType, UnionAll)) :
        X isa Union || X isa UnionAll ? (DataType, UnionAll, Union) : nothing
    ks === nothing && return true
    for kind in ks
        kind <: k && return true
    end
    return false
end

# a closed type with no structure the solve could take apart
is_atom(@nospecialize t) = istypeeq(t) || isegal(t) || t isa DataType && !(t <: Tuple) || isopaque(t)

# a bound C's subtype can decide: no union with a bare variable member
adoptable(@nospecialize t) = t isa DataType || t isa UnionAll ||
    t isa Union && all(m -> !(m isa TypeVar) && adoptable(m), Base.uniontypes(t))

# ----------------------------------------------------------------------------------------
# Phase 1: facts
#
# Three relations between types: `meet!` (the two share a member in a slot every member of the
# result has), `sub!` and `equate!`. Each handles variables, unions and `where`s in its own way
# and hands the remaining structure to `decompose!`. Equalities bind variables at once; subtype
# facts are kept and applied again on every round of `solve!`.

# The parts two types relate through once neither is a variable, a union or a `where`: the
# parameter of `Type{T}` and the one value of `TypeEgal{X}`, tuple elements, and the
# parameters (invariant) at the head two nominal types share.
function decompose!(env, rel::Symbol, @nospecialize(x), @nospecialize(y), lift::Bool)
    if isopaque(x) || isopaque(y)
        # an opaque variable is in no nominal type, tuple or `Type{X}`, and only an empty type is
        # below it
        rel === :sub && isopaque(y) && !never_empty(x) || throw(Unsat())
    elseif istypeeq(x) && istypeeq(y)
        equate!(env, x.T, y.T)
    elseif isegal(x) && istypeeq(y) && rel !== :eq || isegal(y) && istypeeq(x) && rel === :meet
        # the one value of `TypeEgal{X}` is `X`
        equate!(env, x.T, y.T)
    elseif istypeeq(x) || istypeeq(y) || isegal(x) || isegal(y)
        rel === :eq && throw(Unsat())
        # `Type{X}` meets a nominal type only if that is above `Type`, or a kind `X` is in
        (t, n) = istypeeq(x) || isegal(x) ? (x, y) : (y, x)
        if n isa DataType && !(Type <: n)
            cn = closedize(env, n, true)
            cn <: Type || throw(Unsat())
            X = resolve(env, t.T)
            if istypeeq(t)
                (isclosed(X) && !isopaque(X) ? meets_kind(X, cn) : open_kind_meets(X, cn)) || throw(Unsat())
            else
                k = kind_of(X)
                k === nothing || k <: cn || throw(Unsat())
            end
        end
    elseif !(x isa DataType && y isa DataType)
    elseif x <: Tuple && y <: Tuple
        rel === :meet ? meet_tuples!(env, x, y, lift) :
        rel === :sub ? sub_tuples!(env, x, y) : equate_tuples!(env, x, y)
    elseif x <: Tuple || y <: Tuple
        # a tuple type and another nominal type share no member
        throw(Unsat())
    else
        h = rel === :eq ? (x.name === y.name ? (x, y) : nothing) :
            (s = common_head(x, y)) !== nothing ? (s, y) :
            rel === :meet && (s = common_head(y, x)) !== nothing ? (s, x) : nothing
        h === nothing && throw(Unsat())
        for (p, q) in zip(h[1].parameters, h[2].parameters)
            equate!(env, p, q)
        end
    end
end

# `x` and `y` share a member in a slot every member of the result has (a `Vararg` element
# included: a case keeping a `Vararg` assumes it has elements). `lift`: a `where` here may
# become an unknown of the whole type.
function meet!(env, @nospecialize(x), @nospecialize(y), lift::Bool = true)
    x = resolve(env, x)
    y = resolve(env, y)
    (x === y || x === Any || y === Any) && return
    # a slot holding a value other than a type has no members
    (x isa Type || x isa TypeVar) && (y isa Type || y isa TypeVar) || throw(Unsat())
    # closed types where one contains the other meet in the smaller, constraining nothing
    isclosed(x) && isclosed(y) && (x <: y || y <: x) && return
    if lift && (x isa UnionAll || y isa UnionAll) && !bound(env, x) && !bound(env, y)
        x isa UnionAll ? meet!(env, lifted!(env, :meet, y, x), y) : meet!(env, x, lifted!(env, :meet, x, y))
        return
    end
    # a tuple slot is empty exactly when one of its element slots is
    is_tuple_type(x) && is_tuple_type(y) || (x, y) in env.slots || push!(env.slots, (x, y))
    if bound(env, x) || bound(env, y)
        (v, t) = bound(env, x) ? (x, y) : (y, x)
        if bound(env, x) && bound(env, y) && !get(env.diagonal, v, false)
            (v, t) = (t, v)
        end
        if get(env.diagonal, v, false)
            # a diagonal variable is one concrete type: it meets a type by being below it, or
            # by being it when that type is concrete too
            if is_concrete_leaf(t) || get(env.diagonal, t, false)
                equate!(env, v, t)
            else
                # a diagonal variable meeting `Type{X}` is the kind of one of its members
                ct = closedize(env, t, true)
                if ct <: Type && !(ct isa DataType)
                    ks = member_kinds(ct)
                    ks === nothing || fact!(env, v, ks)
                else
                    fact!(env, v, t)
                end
            end
        elseif is_concrete_leaf(t) || isegal(t)
            # a type with one member meets a variable by being in it
            fact!(env, t, v)
        end
    elseif x isa Union || y isa Union
        isclosed(x) && isclosed(y) || meet_union!(env, x isa Union ? y : x, x isa Union ? x : y)
    elseif !(x isa UnionAll || y isa UnionAll)
        decompose!(env, :meet, x, y, lift)
    elseif !(isclosed(x) && isclosed(y))
        meet_where!(env, x isa UnionAll ? x : y, x isa UnionAll ? y : x)
    end
end

# a `where` that stays in place meets `t`: what a local solve implies for the other unknowns holds
function meet_where!(env, u::UnionAll, @nospecialize(t))
    (env.depth >= 3 || exhausted()) && return
    r = local_meet(env, u, t, true)
    r === nothing && throw(Unsat())
    commit!(env, Env[r[1]], Returns(true))
end

function meet_tuples!(env, x::DataType, y::DataType, lift::Bool)
    fx, vx = live_parts(env, x)
    fy, vy = live_parts(env, y)
    pairs, _ = tuple_pairs(fx, vx, fy, vy)
    for (i, (p, q)) in enumerate(pairs)
        # closed elements constrain nothing; they share a member when their meet is not empty
        if isclosed(p) && isclosed(q)
            p <: q || q <: p || meet_closed(p, q) !== Union{} || throw(Unsat())
        else
            meet!(env, p, q, lift && i <= min(length(fx), length(fy)))
        end
    end
    # the trailing elements meet only where the Varargs have elements
    vx !== nothing && vy !== nothing && env.split && meet!(env, va_T(vx), va_T(vy), false)
    length_equal!(env, tuple_length(env, fx, vx), tuple_length(env, fy, vy))
end

# `x` meets a union in a slot every member has: no feasible member empties the slot, one makes
# it that member, and several commit what all of them imply
function meet_union!(env, @nospecialize(x), u::Union)
    (env.depth >= 3 || exhausted()) && return
    feasible = Tuple{Any,Env}[]
    for m in Base.uniontypes(u)
        r = local_meet(env, m, x, false)
        r === nothing || push!(feasible, (m, r[1]))
    end
    isempty(feasible) && throw(Unsat())
    if length(feasible) == 1
        meet!(env, x, feasible[1][1])
        return
    end
    commit!(env, Env[sub for (_, sub) in feasible], Returns(true))
end

# `lo <: hi`, applied again on every round of propagation
function sub!(env, @nospecialize(lo), @nospecialize(hi))
    lo = resolve(env, lo)
    hi = resolve(env, hi)
    (lo === hi || lo === Union{} || hi === Any) && return
    hi === Union{} && never_empty(lo) && throw(Unsat())
    isclosed(lo) && !isclosed(hi) && !(lo <: closedize(env, hi, true)) && throw(Unsat())
    if lo isa UnionAll && !(isclosed(lo) && isclosed(hi))
        # every instance is below `hi`, the extreme ones in particular
        for x in (lo.var.lb, lo.var.ub)
            i = try inst(lo.body, lo.var, x) catch; nothing end
            i === nothing || sub!(env, i, hi)
        end
    elseif isclosed(lo) && isclosed(hi)
        lo <: hi || throw(Unsat())
    elseif lo isa Union
        for m in Base.uniontypes(lo)
            sub!(env, m, hi)
        end
    elseif bound(env, lo)
        sub_var!(env, lo, hi)
    elseif bound(env, hi)
        set_lb!(env, hi, closedize(env, lo, false))
        # lo <: hi <: ub
        ub = closedize(env, hi, true)
        ub === Any || sub!(env, lo, ub)
    elseif hi isa UnionAll
        # `lo <: (body where v)` has one `v`, however often the fact is applied
        sub!(env, lo, lifted!(env, :sub, lo, hi))
    elseif hi isa Union
        sub_union!(env, lo, hi)
    else
        decompose!(env, :sub, lo, hi, false)
    end
end

function sub_var!(env, v::TypeVar, @nospecialize hi)
    if bound(env, hi)
        sub!(env, v, closedize(env, hi, true))
        sub!(env, closedize(env, v, false), hi)
    elseif isclosed(hi)
        set_ub!(env, v, hi)
        # lb <: v <: hi, which the meet of the bounds may not show
        lb = closedize(env, v, false)
        lb === Union{} || sub!(env, lb, hi)
    else
        bnd = env.b[v]
        # implied by the current bound: it holds with the unknowns held fixed
        implied = bnd.ub !== Any && (bnd.ub <: hi || subst(env, bnd.ub) <: subst(env, hi))
        m = implied || !adoptable(hi) ? nothing :
            bnd.ub === Any ? (occurs(v, subst(env, hi)) ? nothing : hi) :
            meet_bound(env, v, bnd.ub, hi)
        if implied
        elseif m !== nothing && !isclosed(m)
            if m != bnd.ub
                bnd.ub = m
                touched!(env)
            end
        else
            set_ub!(env, v, m === nothing ? closedize(env, hi, true) : m)
        end
        # lb <: v <: hi
        lb = closedize(env, v, false)
        lb === Union{} || sub!(env, lb, hi)
    end
end

# `lo` below a union: below the members it can meet, and below the only one when one is left
function sub_union!(env, @nospecialize(lo), u::Union)
    ms = Base.uniontypes(u)
    closed = Union{filter(isclosed, ms)...}
    isclosed(lo) && lo <: closed && return
    lub = closedize(env, lo, true)
    candidates = Any[]
    for c in ms
        cub = closedize(env, c, true)
        (cub isa Type ? !disjoint(lub, cub) : true) && push!(candidates, c)
    end
    isempty(candidates) && throw(Unsat())
    length(candidates) == 1 && sub!(env, lo, candidates[1])
end

function sub_tuples!(env, lo::DataType, hi::DataType)
    fl, vl = live_parts(env, lo)
    fh, vh = live_parts(env, hi)
    # each element `lo` has at a position `hi` has elements at
    for i in 1:max(length(fl), length(fh))
        p = i <= length(fl) ? fl[i] : vl === nothing ? nothing : va_T(vl)
        q = i <= length(fh) ? fh[i] : vh === nothing ? nothing : va_T(vh)
        (p === nothing || q === nothing) && throw(Unsat())
        sub!(env, p, q)
    end
    length_equal!(env, tuple_length(env, fl, vl), tuple_length(env, fh, vh))
end

function equate!(env, @nospecialize(x), @nospecialize(y))
    x = resolve(env, x)
    y = resolve(env, y)
    x === y && return
    # lengths related by an offset are equal through their roots
    if x isa TypeVar && haskey(env.offsets, x) || y isa TypeVar && haskey(env.offsets, y)
        relate!(env, x, y, 0)
        return
    end
    if bound(env, x) && bound(env, y)
        rx, ry = rank(env, x), rank(env, y)
        rx != ry ? (rx < ry ? bind_eq!(env, x, y) : bind_eq!(env, y, x)) :
            canonical(env, x, y) === x ? bind_eq!(env, y, x) : bind_eq!(env, x, y)
    elseif bound(env, x) || bound(env, y)
        bound(env, x) ? bind_eq!(env, x, y) : bind_eq!(env, y, x)
    elseif isclosed(x) && isclosed(y)
        x == y || throw(Unsat())
    elseif x isa Union || y isa Union || x === Union{} || y === Union{}
        # equal types are below each other
        fact!(env, x, y)
        fact!(env, y, x)
    elseif x isa UnionAll || y isa UnionAll
        hx, hy = head(Base.unwrap_unionall(x)), head(Base.unwrap_unionall(y))
        hx !== nothing && hy !== nothing && hx !== hy && throw(Unsat())
        (u, d) = x isa UnionAll ? (x, y) : (y, x)
        # a `where` over a variable in an invariant position denotes several types unless its
        # bounds pin it, and a data type is one type
        if d isa DataType && u isa UnionAll && u.var.lb !== u.var.ub && occurrences(u.body, u.var)[2] > 0
            throw(Unsat())
        end
        fact!(env, x, y)
        fact!(env, y, x)
    else
        decompose!(env, :eq, x, y, false)
    end
end

# `x == y` for tuples, possibly with a trailing Vararg
function equate_tuples!(env, x::DataType, y::DataType)
    fx, vx = live_parts(env, x)
    fy, vy = live_parts(env, y)
    if vx === nothing && vy === nothing
        length(fx) == length(fy) || throw(Unsat())
        for (p, q) in zip(fx, fy)
            equate!(env, p, q)
        end
        return
    end
    if vx === nothing || vy === nothing
        (f, v, g) = vx === nothing ? (fy, vy, fx) : (fx, vx, fy)
        T, n = va_T(v), va_N(v)
        n isa TypeVar && (n = resolve(env, n))
        length(g) >= length(f) || throw(Unsat())
        # a Vararg of any length never equals fixed elements: `<:` does not take its element
        # to be `Union{}`
        if n isa Int
            length(g) == length(f) + n || throw(Unsat())
        elseif !bound(env, n)
            throw(Unsat())
        end
        for i in eachindex(f)
            equate!(env, f[i], g[i])
        end
        for i in length(f)+1:length(g)
            equate!(env, T, g[i])
        end
        n isa Int || equate!(env, n, length(g) - length(f))
        return
    end
    pairs, _ = tuple_pairs(fx, vx, fy, vy)
    for (p, q) in pairs
        equate!(env, p, q)
    end
    equate!(env, va_T(vx), va_T(vy))
    nx, ny = va_N(vx), va_N(vy)
    nx === nothing && ny === nothing && return
    # nor one of some length
    (nx === nothing || ny === nothing) && throw(Unsat())
    length_equal!(env, tuple_length(env, fx, vx), tuple_length(env, fy, vy))
end

# the trailing Vararg of a tuple, or nothing when absent or known to be empty
function live_parts(env, t::DataType)
    f, v = tuple_parts(t)
    v === nothing && return f, v
    n = va_N(v)
    n isa TypeVar && (n = resolve(env, n))
    return f, n === 0 ? nothing : v
end

# element pairs of two tuples aligned position by position, padding with vararg elements
function tuple_pairs(fx, vx, fy, vy)
    (vx === nothing && length(fy) > length(fx)) && throw(Unsat())
    (vy === nothing && length(fx) > length(fy)) && throw(Unsat())
    k = vx === nothing ? length(fx) : vy === nothing ? length(fy) : max(length(fx), length(fy))
    el(f, v, i) = i <= length(f) ? f[i] : va_T(v)
    return [(el(fx, vx, i), el(fy, vy, i)) for i in 1:k], k
end

# the length of a tuple part: fixed count plus the vararg length (0, :any, Int or a variable)
function tuple_length(env, fixed, va)
    va === nothing && return (length(fixed), 0)
    n = va_N(va)
    n === nothing && return (length(fixed), :any)
    return (length(fixed), n isa TypeVar ? resolve(env, n) : n)
end

# require kx + nx == ky + ny
function length_equal!(env, (kx, nx), (ky, ny))
    if nx isa TypeVar
        nx, d = len_root(env, nx)
        kx += d
    end
    if ny isa TypeVar
        ny, d = len_root(env, ny)
        ky += d
    end
    (nx === :any || ny === :any) && begin
        (nx === :any && ny isa Int && ky + ny < kx) && throw(Unsat())
        (ny === :any && nx isa Int && kx + nx < ky) && throw(Unsat())
        nx === :any && bound(env, ny) && at_least!(env, ny, kx - ky)
        ny === :any && bound(env, nx) && at_least!(env, nx, ky - kx)
        return
    end
    if nx isa Int && ny isa Int
        kx + nx == ky + ny || throw(Unsat())
    elseif nx isa Int || ny isa Int
        (n, k, other) = nx isa Int ? (kx + nx, ky, ny) : (ky + ny, kx, nx)
        n - k >= 0 || throw(Unsat())
        bound(env, other) && equate!(env, other, n - k)
    elseif bound(env, nx) && bound(env, ny)
        relate!(env, nx, ny, ky - kx)
    end
end

# a length variable as a root length and an offset, or its integer value when known
function len_root(env, @nospecialize(v))
    v = resolve(env, v)
    d = 0
    while v isa TypeVar && haskey(env.offsets, v)
        u, e = env.offsets[v]
        v = resolve(env, u)
        d += e
    end
    return v isa Int ? (v + d, 0) : (v, d)
end

# the lengths `x` and `y` satisfy `x == y + d`
function relate!(env, @nospecialize(x), @nospecialize(y), d::Int)
    rx, dx = len_root(env, x)
    ry, dy = len_root(env, y)
    e = dy + d - dx
    if rx isa Int && ry isa Int
        rx == ry + e || throw(Unsat())
    elseif rx isa Int
        rx - e >= 0 || throw(Unsat())
        bound(env, ry) && equate!(env, ry, rx - e)
    elseif ry isa Int
        ry + e >= 0 || throw(Unsat())
        bound(env, rx) && equate!(env, rx, ry + e)
    elseif rx === ry
        e == 0 || throw(Unsat())
    elseif !(bound(env, rx) && bound(env, ry))
    elseif e == 0
        equate!(env, rx, ry)
    else
        (hi, lo, off) = e > 0 ? (rx, ry, e) : (ry, rx, -e)
        env.offsets[hi] = (lo, off)
        env.changed = true
        at_least!(env, lo, get(env.mins, hi, 0) - off)
    end
end

# the length variable `v` is at least `n`
function at_least!(env, @nospecialize(v), n::Int)
    r, d = len_root(env, v)
    r isa Int && (r >= n || throw(Unsat()); return)
    bound(env, r) || return
    n - d > get(env.mins, r, 0) || return
    env.mins[r] = n - d
    env.changed = true
end

# The meet of a variable's bound with another, at least one of them open. A bound only matters
# when its variable is not empty, so the local solve commits what it implies for the variables
# that occur only in this variable's bounds.
const BOUNDS_ACTIVE = Tuple{Any,Any}[]

function meet_bound(env, v, @nospecialize(a), @nospecialize(b))
    (env.depth >= 3 || exhausted()) && return nothing
    # a local solve inherits the facts that started it: the same meet inside it adds nothing
    any(p -> p[1] === a && p[2] === b, BOUNDS_ACTIVE) && return nothing
    push!(BOUNDS_ACTIVE, (a, b))
    try
        r = local_meet(env, a, b, true)
        r === nothing && return Union{}
        sub, a, b = r
        m = try wrap(sub, build(sub, b, a), new_vars(sub, env)) catch err; err isa Unsat ? Union{} : rethrow() end
        # ownership follows equality: once `v` equals another variable, the variables only in either's
        # bounds are only in the bounds of the one they now are
        # a variable every member has is not empty, so what its bounds imply holds for all variables
        commit!(env, [sub], v in env.inhabited ? Returns(true) :
            w -> (o = get(env.owner, w, nothing); o !== nothing && resolve(env, o) === resolve(env, v)))
        return occurs(v, subst(env, m)) ? nothing : m
    finally
        pop!(BOUNDS_ACTIVE)
    end
end

# Commit to `env`, as facts, what every one of the solved alternatives `alts` implies about the
# variables `keep` selects: an equality, or else the bounds, they all agree on. A fact that
# mentions a variable local to the alternatives stays local.
function commit!(env, alts::Vector{Env}, keep)
    function agreed(f)
        x = f(alts[1])
        for a in alts
            same_type(f(a), x) || return nothing
        end
        return x
    end
    function local_to(@nospecialize t)
        for w in keys(alts[1].b)
            !haskey(env.b, w) && Base.has_typevar(t, w) && return true
        end
        return false
    end
    for k in collect(keys(env.b))
        (resolve(env, k) === k && keep(k)) || continue
        eq = agreed(a -> subst(a, k))
        if eq !== nothing && eq !== k
            local_to(eq) || equate!(env, k, eq)
        elseif all(a -> resolve(a, k) === k, alts)
            ub = agreed(a -> subst(a, a.b[k].ub))
            ub === nothing || local_to(ub) || same_type(ub, env.b[k].ub) || fact!(env, k, ub)
            lb = agreed(a -> subst(a, a.b[k].lb))
            lb === nothing || local_to(lb) || same_type(lb, env.b[k].lb) || fact!(env, lb, k)
        end
    end
end

# `b` meeting `a` in a copy of `env` where `a`'s `where`s (and `b`'s, when `liftb`) are local
# unknowns: the solved copy and the unwrapped types, or nothing when they cannot meet. A local
# solve is one approximation; the enclosing fixpoint evaluates it again when its state changes.
# A budget on the local solves of one intersection; past it the solves that only add facts are
# skipped and the ones that build fall back to closed approximations.
const WORK = Ref(0)
const BUDGET = 2000
exhausted() = WORK[] > BUDGET

function local_meet(env, @nospecialize(a), @nospecialize(b), liftb::Bool)
    nominally_disjoint(a, b) && return nothing
    WORK[] += 1
    sub = local_env(env)
    while a isa UnionAll
        a = lift!(sub, a)
    end
    while liftb && b isa UnionAll
        b = lift!(sub, b)
    end
    append!(sub.locals, env.locals, new_vars(sub, env))
    try
        meet!(sub, b, a, false)
        solve!(sub)
    catch err
        err isa Unsat || rethrow()
        return nothing
    end
    return sub, a, b
end

# Apply the subtype and meet facts until nothing changes, checking each round that every variable's
# bounds and every declared parameter bound can still hold; at the end, every slot pair must
# still be able to share a member.
function solve!(env)
    for _ in 1:32
        env.changed = false
        for (lo, hi) in copy(env.subs)
            sub!(env, lo, hi)
        end
        # a meet yields something new only once a binding has changed
        if env.met != env.version
            env.met = env.version
            for (x, y) in copy(env.meets)
                meet!(env, x, y, false)
            end
        end
        for (v, bnd) in env.b
            lb, ub = closedize(env, v, false), closedize(env, v, true)
            # a length is an integer, not a type with members
            v in env.lengths && (lb isa Type && lb !== Union{} || ub isa Type && ub !== Any) && throw(Unsat())
            m = get(env.mins, v, 0)
            m > 0 && (r = resolve(env, v); r isa Int && r < m && throw(Unsat()))
            (lb isa Type && ub isa Type) || continue
            lb <: ub || throw(Unsat())
            # a diagonal variable is concrete, and the concrete type containing a concrete type
            # is that type
            get(env.diagonal, v, false) && bnd.eq === nothing && is_concrete_leaf(lb) && equate!(env, v, lb)
        end
        for (dlb, p, dub) in env.checks
            p isa Type || p isa TypeVar || continue
            lo = closedize(env, p, false)
            lo isa Type && !(lo <: dub) && throw(Unsat())
            hi = closedize(env, p, true)
            hi isa Type && !(dlb <: hi) && throw(Unsat())
        end
        # a slot pair whose closed forms share one concrete type holds that type: both sides
        # contain it
        for (x, y) in unique(env.slots)
            isclosed(x) && isclosed(y) && continue
            cx, cy = closedize(env, x, true), closedize(env, y, true)
            (cx isa Type && cy isa Type) || continue
            m = meet_or_nothing(cx, cy)
            m === Union{} && throw(Unsat())
            if m isa Type && is_concrete_leaf(m)
                isclosed(x) || fact!(env, m, x)
                isclosed(y) || fact!(env, m, y)
            end
        end
        env.changed || break
    end
end

# ----------------------------------------------------------------------------------------
# Phase 2: the result

# The meet of `x` and `y` in a covariant position, from the solved facts.
function build(env, @nospecialize(x), @nospecialize(y))
    x = subst(env, x)
    y = subst(env, y)
    x === y && return x
    x === Any && return y
    y === Any && return x
    (x === Union{} || y === Union{}) && return Union{}
    if isclosed(x) && isclosed(y)
        x <: y && return x
        y <: x && return y
    end
    if istypeeq(x) && istypeeq(y)
        return Type{canonical(env, x.T, y.T)}
    end
    # `Type{T}` meeting a kind: the members of `Type{T}` that the kind has, which no condition
    # on `T` describes; `Type{T}` is the tighter of the two over-approximations
    for (t, k) in ((x, y), (y, x))
        istypeeq(t) && !istypeeq(k) || continue
        ck = closedize(env, k, true)
        ck <: Type || continue
        return ck <: closedize(env, t, true) ? k : t
    end
    if x isa Union || y isa Union
        # each member meets the other side on its own
        (a, b) = x isa Union ? (x, y) : (y, x)
        ms = Any[]
        for m in Base.uniontypes(a)
            push!(ms, isclosed(m) && isclosed(b) ? build(env, m, b) : build_local(env, m, b))
        end
        return Union{ms...}
    end
    if bound(env, x) || bound(env, y)
        (v, t) = bound(env, x) ? (x, y) : (y, x)
        bound(env, t) && get(env.diagonal, t, false) && !get(env.diagonal, v, false) && ((v, t) = (t, v))
        get(env.diagonal, v, false) && return v
        vub = closedize(env, v, true)
        tub = closedize(env, t, true)
        if vub <: tub && tub <: vub && bound(env, t)
            return canonical(env, v, t)
        end
        vub <: tub && return v
        tub <: vub && return t
        isclosed(t) && return meet_closed(vub, t)
        return meet_closed(vub, tub)
    end
    if x isa DataType && y isa DataType
        if x <: Tuple && y <: Tuple
            fx, vx = live_parts(env, x)
            fy, vy = live_parts(env, y)
            pairs, k = try tuple_pairs(fx, vx, fy, vy) catch err; err isa Unsat ? (nothing, 0) : rethrow() end
            pairs === nothing && return Union{}
            ps = Any[isclosed(p) && isclosed(q) ? meet_closed(p, q) : build(env, p, q) for (p, q) in pairs]
            any(==(Union{}), ps) && return Union{}
            (vx === nothing || vy === nothing) && return Tuple{ps...}
            T = build(env, va_T(vx), va_T(vy))
            # a length variable carries over when its fixed part already has length k
            n = nothing
            aligned = Any[]
            for (f, v) in ((fx, vx), (fy, vy))
                m = va_N(v)
                m isa TypeVar && (m = resolve(env, m))
                if m isa Int
                    n = m - (k - length(f))
                    n < 0 && return Union{}
                    break
                elseif m !== nothing && length(f) == k
                    push!(aligned, m)
                end
            end
            if n === nothing && !isempty(aligned)
                n = length(aligned) == 1 ? aligned[1] : canonical(env, aligned[1], aligned[2])
            end
            T === Union{} && return n isa Int && n > 0 ? Union{} : Tuple{ps...}
            return Tuple{ps..., make_va(T, n)}
        end
        x.name === y.name && return canonical(env, x, y)
        common_head(x, y) === nothing || return x
        common_head(y, x) === nothing || return y
    end
    x isa UnionAll && return build_local(env, x, y)
    y isa UnionAll && return build_local(env, y, x)
    return meet_closed(closedize(env, x, true), closedize(env, y, true))
end

# `u ∩ t` for a union member or a `where` that cannot move to the top. The local solve decides
# emptiness and resolves its own variables; an outer one may be specialized to a closed value,
# but not re-expressed through another outer variable, and a diagonal one not at all, so the
# rest of the type keeps its correlations.
function build_local(env, @nospecialize(u), @nospecialize(t))
    exhausted() && return meet_closed(closedize(env, u, true), closedize(env, t, true))
    r = local_meet(env, u, t, true)
    r === nothing && return Union{}
    sub, u, t = r
    for (k, b) in env.b
        k in sub.locals && continue
        get(env.diagonal, k, false) || continue
        sub.b[k] = Binding(b.lb, b.ub, b.eq)
    end
    return try wrap(sub, build(sub, t, u), new_vars(sub, env)) catch err; err isa Unsat ? Union{} : rethrow() end
end

function wrap(env, @nospecialize(t), only = nothing)
    t === Union{} && return t
    ours(w) = only === nothing || w in only
    direct = TypeVar[v for v in keys(env.b) if ours(v) && Base.has_typevar(t, v)]
    isempty(direct) && return t
    # the variables of the result: those in `t` and those in the bounds of these, each after
    # the variables its bounds mention
    vars = TypeVar[]
    seen = IdSet{TypeVar}()
    function visit(v)
        v in seen && return
        push!(seen, v)
        lb, ub = subst(env, env.b[v].lb), subst(env, env.b[v].ub)
        for w in keys(env.b)
            ours(w) && w !== v && (Base.has_typevar(ub, w) || Base.has_typevar(lb, w)) && visit(w)
        end
        push!(vars, v)
    end
    foreach(visit, direct)
    fresh = IdDict{TypeVar,Any}()
    order = TypeVar[]
    # a bound may mention a result variable created before it
    function bound_of(v, upper)
        b = subst(env, upper ? env.b[v].ub : env.b[v].lb)
        if !isclosed(b)
            for w in keys(env.b)
                ours(w) && Base.has_typevar(b, w) && !(haskey(fresh, w) && fresh[w] isa TypeVar) && return closedize(env, v, upper)
            end
            for w in order
                b = inst(b, w, fresh[w])
            end
        end
        return b
    end
    for v in vars
        lb = closedize(env, v, false)
        ub = closedize(env, v, true)
        if !(lb isa Type && ub isa Type)
            # one value (an integer, an opaque variable) stands for the variable
            if lb !== nothing && lb === ub
                fresh[v] = ub
                continue
            end
            lb isa Type || (lb = Union{})
            ub isa Type || (ub = Any)
        end
        lb <: ub || return Union{}
        cov, inv = occurrences(t, v)
        # only covariant here: the variable denotes nothing beyond its upper bound,
        # unless it was diagonal in its input
        covariant_only = inv == 0 && !(cov > 1 && get(env.diagonal, v, false)) && v in direct
        if lb == ub || covariant_only
            fresh[v] = ub
        else
            fresh[v] = TypeVar(v.name, bound_of(v, false), bound_of(v, true))
            push!(order, v)
        end
    end
    body = t
    for v in vars
        # a variable that is empty in a tuple element slot empties the tuple
        body = try inst(body, v, fresh[v]) catch; fresh[v] === Union{} ? (return Union{}) : rethrow() end
    end
    return foldr((v, b) -> UnionAll(fresh[v], b), order; init = body)
end

# ----------------------------------------------------------------------------------------
# Cases

# Move `where`s in covariant positions (tuple elements, union members) to the top, except in
# the element slots `other` has identically or that are closed in both.
function lift_covariant(@nospecialize(t), vars, @nospecialize(other = nothing))
    if t isa UnionAll
        u = Base.rename_unionall(t)
        push!(vars, u.var)
        return lift_covariant(u.body, vars)
    elseif t isa Union
        return Union{(lift_covariant(m, vars) for m in Base.uniontypes(t))...}
    elseif t isa DataType && t <: Tuple
        fixed, va = tuple_parts(t)
        same(i) = is_tuple_type(other) && i <= length(other.parameters) && !isvararg(other.parameters[i]) &&
            (other.parameters[i] === fixed[i] || isclosed(fixed[i]) && isclosed(other.parameters[i]))
        ps = Any[same(i) ? fixed[i] : lift_covariant(fixed[i], vars) for i in eachindex(fixed)]
        va === nothing || push!(ps, va)
        return Tuple{ps...}
    end
    return t
end

# variables used as a Vararg length
function length_vars!(acc, @nospecialize t)
    if isvararg(t)
        n = va_N(t)
        n isa TypeVar && !(n in acc) && push!(acc, n)
        length_vars!(acc, va_T(t))
    elseif t isa Union
        foreach(m -> length_vars!(acc, m), Base.uniontypes(t))
    elseif t isa DataType || istypeeq(t)
        foreach(p -> length_vars!(acc, p), istypeeq(t) ? Any[t.T] : t.parameters)
    elseif t isa UnionAll
        length_vars!(acc, t.body)
    end
    return acc
end

# The cases of `t`: its covariant unions split, and a Vararg that may be empty dropped or kept.
# Past `limit` cases `t` stays whole and `split[]` is cleared.
function expand(@nospecialize(t), limit, split::Ref{Bool}, @nospecialize(other = nothing), keep = Int[])
    if t isa Union
        out = Any[]
        for m in Base.uniontypes(t)
            append!(out, expand(m, limit, split, other, keep))
        end
        return out
    elseif t isa DataType && t <: Tuple && !isempty(t.parameters)
        fixed, va = tuple_parts(t)
        ofixed = is_tuple_type(other) ? tuple_parts(other)[1] : Any[]
        out = Any[Tuple{}]
        for (i, p) in enumerate(fixed)
            ps = i in keep ? Any[p] : expand(p, limit, split)
            # a member nominally disjoint from the other slot has no case
            i <= length(ofixed) && filter!(q -> !nominally_disjoint(upper(q), upper(ofixed[i])), ps)
            next = Any[]
            for o in out, q in ps
                push!(next, Tuple{o.parameters..., q})
            end
            out = next
            # past the budget the union stays whole; it is then met member by member
            length(out) > limit && (split[] = false; return Any[t])
        end
        va === nothing && return out
        # element constraints hold only when the Vararg has elements
        keep = Any[Tuple{o.parameters..., va} for o in out]
        (va_N(va) !== nothing || va_T(va) === Any) && return keep
        return Any[keep; out]
    end
    return Any[t]
end

# `v` occurs in `t` only as a Vararg length
only_length(@nospecialize(t), v::TypeVar) = t === v ? false :
    t isa Union ? only_length(t.a, v) && only_length(t.b, v) :
    t isa UnionAll ? only_length(t.var.lb, v) && only_length(t.var.ub, v) && only_length(t.body, v) :
    istypeeq(t) ? only_length(t.T, v) :
    isvararg(t) ? only_length(va_T(t), v) :
    t isa DataType ? all(p -> only_length(p, v), t.parameters) : true

# `t` with each `Vararg{T, v}` written as `m` elements `T` and `Vararg{T, w}`
function peel(@nospecialize(t), v::TypeVar, m::Int, w::TypeVar)
    t isa Union && return Union{peel(t.a, v, m, w), peel(t.b, v, m, w)}
    t isa UnionAll && return UnionAll(t.var, peel(t.body, v, m, w))
    istypeeq(t) && return Type{peel(t.T, v, m, w)}
    t isa DataType && Base.has_typevar(t, v) || return t
    ps = Any[peel(p, v, m, w) for p in t.parameters]
    if t <: Tuple && !isempty(ps) && isvararg(ps[end]) && va_N(ps[end]) === v
        T = va_T(pop!(ps))
        append!(ps, fill(T, m))
        push!(ps, Vararg{T, w})
    end
    return t <: Tuple ? Tuple{ps...} : t.name.wrapper{ps...}
end

# A length variable known to be at least `m` and used only as a length is `m` more elements
# and a fresh length.
function peel_lengths!(env, @nospecialize(a), @nospecialize(b))
    a, b = subst(env, a), subst(env, b)
    only(v) = only_length(a, v) && only_length(b, v) &&
        all(bnd -> only_length(bnd.lb, v) && only_length(bnd.ub, v) && (bnd.eq === nothing || bnd.eq === v || only_length(bnd.eq, v)), values(env.b))
    function rewrite(v, m, w)
        a, b = peel(a, v, m, w), peel(b, v, m, w)
        for bnd in values(env.b)
            bnd.lb = peel(bnd.lb, v, m, w)
            bnd.ub = peel(bnd.ub, v, m, w)
            bnd.eq === nothing || (bnd.eq = peel(bnd.eq, v, m, w))
        end
    end
    # a length that is a root plus an offset is that many more elements and the root
    for v in collect(keys(env.offsets))
        resolve(env, v) === v || continue
        r, d = len_root(env, v)
        if r isa Int
            env.b[v].eq = r
        elseif d > 0 && only(v)
            rewrite(v, d, r)
        end
    end
    # a root length known to be at least `m` is `m` more elements and a fresh length
    for (v, m) in collect(env.mins)
        m > 0 && resolve(env, v) === v && !haskey(env.offsets, v) && only(v) || continue
        w = TypeVar(v.name)
        env.b[w] = Binding(Union{}, Any, nothing)
        rewrite(v, m, w)
    end
    return a, b
end

# the variables that are, directly or through tuple elements, a fixed element of the tuple `t`
function inhabited!(acc, @nospecialize t)
    is_tuple_type(t) || return acc
    fixed, _ = tuple_parts(t)
    for p in fixed
        p isa TypeVar ? (p in acc || push!(acc, p)) : inhabited!(acc, p)
    end
    return acc
end

function intersect_case(avars, a, bvars, b, diagonal, empty, owner, split, uses, lengths)
    env = Env(diagonal, owner)
    env.split = split
    env.uses = uses
    env.lengths = lengths
    inhabited!(env.inhabited, a)
    inhabited!(env.inhabited, b)
    for v in Iterators.flatten((avars, bvars))
        env.b[v] = Binding(v.lb, v.ub, nothing)
        v.lb === Union{} || fact!(env, v.lb, v)
        v.ub === Any || fact!(env, v, v.ub)
    end
    add_param_bounds!(env, a)
    add_param_bounds!(env, b)
    try
        foreach(n -> bind_eq!(env, n, 0), empty)
        meet!(env, a, b)
        push!(env.meets, (a, b))
        env.met = env.version
        solve!(env)
        a, b = peel_lengths!(env, a, b)
        body = build(env, a, b)
        r = wrap(env, body)
        r === Union{} && return nothing
        return (env, body, r)
    catch err
        err isa Unsat && return nothing
        rethrow()
    end
end

# Union states solved separately share one `where`: a variable free in several states
# gets the join of its bounds over them.
function merge_cases(cases)
    if length(cases) <= 64
        keep = Any[]
        for i in eachindex(cases)
            ri = cases[i][3]
            covered = false
            for j in eachindex(cases)
                j == i && continue
                rj = cases[j][3]
                if ri <: rj && (j < i || !(rj <: ri))
                    covered = true
                    break
                end
            end
            covered || push!(keep, cases[i])
        end
        cases = keep
    end
    length(cases) == 1 && return cases[1][3]
    # a variable is shared when every state leaves it free with the same bounds over shared
    # variables; otherwise each state gets its own copy
    free = IdDict{TypeVar,Vector{Int}}()
    for (i, (env, _, _)) in enumerate(cases), v in keys(env.b)
        resolve(env, v) === v && push!(get!(free, v, Int[]), i)
    end
    lbs(v, i) = subst(cases[i][1], cases[i][1].b[v].lb)
    ubs(v, i) = subst(cases[i][1], cases[i][1].b[v].ub)
    shared = IdDict{TypeVar,Bool}()
    for (v, is) in free
        shared[v] = same(Any[lbs(v, i) for i in is]) && same(Any[ubs(v, i) for i in is])
    end
    changed = true
    while changed
        changed = false
        for (v, is) in free
            shared[v] || continue
            b = Any[lbs(v, is[1]), ubs(v, is[1])]
            for w in keys(free)
                if !shared[w] && (Base.has_typevar(b[1], w) || Base.has_typevar(b[2], w))
                    shared[v] = false
                    changed = true
                    break
                end
            end
        end
    end
    copies = [IdDict{TypeVar,TypeVar}() for _ in cases]
    for (v, is) in free, i in is
        shared[v] || (copies[i][v] = TypeVar(v.name))
    end
    rn(i, @nospecialize t) = (for (v, c) in copies[i]; t = inst(t, v, c); end; t)
    menv = Env(IdDict{TypeVar,Bool}())
    menv.uses = copy(cases[1][1].uses)
    for (v, is) in free, i in is
        shared[v] && i != is[1] && continue
        env = cases[i][1]
        c = shared[v] ? v : copies[i][v]
        menv.b[c] = Binding(rn(i, lbs(v, i)), rn(i, ubs(v, i)), nothing)
        menv.diagonal[c] = shared[v] ? all(j -> get(cases[j][1].diagonal, v, false), is) : get(env.diagonal, v, false)
        shared[v] || (menv.uses[c] = get(env.uses, v, (0, 0)))
    end
    return wrap(menv, Union{Any[rn(i, cases[i][2]) for i in eachindex(cases)]...})
end

function same(xs)
    for x in xs
        x == xs[1] || return false
    end
    return true
end

# ----------------------------------------------------------------------------------------
# Entry points

const ACTIVE = Tuple{Any,Any}[]

function intersect(@nospecialize(A), @nospecialize(B))
    WORK[] = 0
    n = length(OPAQUE)
    free_vars!(OPAQUE, A)
    free_vars!(OPAQUE, B)
    try
        # a `where` of one input is not the variable free in the other
        any(isopaque, unwrap(A)[1]) && (A = Base.rename_unionall(A))
        any(isopaque, unwrap(B)[1]) && (B = Base.rename_unionall(B))
        r = meet_or_nothing(A, B)
        r === nothing && throw(Unsupported(:cycle))
        return r
    finally
        resize!(OPAQUE, n)
    end
end

# `A ∩ B`, or nothing when it is already being computed
function meet_or_nothing(@nospecialize(A), @nospecialize(B))
    A <: B && return A
    B <: A && return B
    nominally_disjoint(A, B) && return Union{}
    # `Type{T}` with `T` over a closed range meets another such range in the meet of the ranges
    ra, rb = type_range(A), type_range(B)
    if ra !== nothing && rb !== nothing
        l, u = Union{ra[1], rb[1]}, meet_closed(ra[2], rb[2])
        l <: u || return Union{}
        u <: l && return Type{l}
        v = TypeVar((A isa UnionAll ? A : B).var.name, l, u)
        return UnionAll(v, Type{v})
    end
    # closed atoms share a member only by one containing the other: invariant parameters are
    # equal or disjoint, nominal types share a member only along one supertype chain, and
    # `Type{X}` and `TypeEgal{X}`, whose one member is `X`, meet a kind that `X` is in
    if isclosed(A) && isclosed(B) && is_atom(A) && is_atom(B)
        for (s, k) in ((A, B), (B, A))
            (istypeeq(s) || isegal(s)) && !(istypeeq(k) || isegal(k)) && return in_kind(s, k) ? s : Union{}
        end
        return Union{}
    end
    for (x, y) in ACTIVE
        if same_type(x, A) && same_type(y, B) || same_type(x, B) && same_type(y, A)
            CUT[] += 1
            return nothing
        end
    end
    closed = isclosed(A) && isclosed(B)
    closed && haskey(MEMO, (A, B)) && return MEMO[(A, B)]
    cut = CUT[]
    push!(ACTIVE, (A, B))
    r = try
        intersect_open(A, B)
    finally
        pop!(ACTIVE)
    end
    # a meet of closed types is a function of them, unless it was computed with a cycle cut short
    closed && CUT[] == cut && (MEMO[(A, B)] = r)
    return r
end

const MEMO = IdDict{Any,Any}()
const CUT = Ref(0)

function intersect_open(@nospecialize(A), @nospecialize(B))
    avars, a = unwrap(A)
    for v in avars
        if Base.has_typevar(B, v)
            B = Base.rename_unionall(B)
            break
        end
    end
    bvars, b = unwrap(B)
    a, b = lift_covariant(a, avars, b), lift_covariant(b, bvars, a)
    check_supported(a, avars)
    check_supported(b, bvars)
    diagonal = IdDict{TypeVar,Bool}()
    owner = IdDict{TypeVar,TypeVar}()
    uses = IdDict{TypeVar,Tuple{Int,Int}}()
    for (vars, body) in ((avars, a), (bvars, b)), v in vars
        diagonal[v] = is_diagonal(vars, body, v)
        uses[v] = Tuple(occurrences(vars, body, v))
        occurrences(body, v) == [0, 0] || continue
        os = TypeVar[w for w in vars if w !== v && (Base.has_typevar(w.lb, v) || Base.has_typevar(w.ub, v))]
        length(os) == 1 && (owner[v] = os[1])
    end
    lens = TypeVar[]
    length_vars!(lens, a)
    length_vars!(lens, b)
    filter!(n -> n in avars || n in bvars, lens)
    lengths = copy(lens)
    split = Ref(length(lens) <= 4)
    split[] || empty!(lens)
    keep = independent_slots(a, b, diagonal)
    as, bs = expand(a, 128, split, b, keep), expand(b, 128, split, a, keep)
    cases = Any[]
    for mask in 0:(1 << length(lens)) - 1, ai in as, bi in bs
        empty = TypeVar[lens[i] for i in eachindex(lens) if (mask >> (i - 1)) & 1 == 1]
        # diagonal only if diagonal both in the input and in this branch of its unions
        diag = IdDict{TypeVar,Bool}()
        for v in avars; diag[v] = diagonal[v] && is_diagonal(avars, ai, v); end
        for v in bvars; diag[v] = diagonal[v] && is_diagonal(bvars, bi, v); end
        r = intersect_case(avars, ai, bvars, bi, diag, empty, owner, split[], uses, lengths)
        r === nothing || push!(cases, r)
    end
    isempty(cases) && return Union{}
    return merge_cases(cases)
end

# The intersection for method matching, with the values of `B`'s variables: a single tuple
# when both inputs are tuples (`B` itself when the cases do not merge into one), and the
# variable values read off matching the result against `B`.
function intersect_env(@nospecialize(A), @nospecialize(B))
    t = intersect(A, B)
    t === Union{} && return t, Core.svec()
    tuples = Base.unwrap_unionall(A) isa DataType && Base.unwrap_unionall(A) <: Tuple &&
             Base.unwrap_unionall(B) isa DataType && Base.unwrap_unionall(B) <: Tuple
    tuples && !(Base.unwrap_unionall(t) isa DataType) && (t = B)
    env = ccall(:jl_type_intersection_with_env, Any, (Any, Any), t <: B ? t : B, B)[2]
    return t, env
end

end
