# This file is a part of Julia. License is MIT: https://julialang.org/license

## Functions to compute the reduced shape

# --- Reduction planning (array-generic) ---------------------------------------

# Main plan struct that feeds all kernels with pre-computed decisions
struct MRPlan{N,T}
    reduce_mask  :: NTuple{N,Bool}   # dims to reduce
    keep_mask    :: NTuple{N,Bool}   # !reduce_mask
    sizes        :: NTuple{N,Int}    # size(A,d)
    inner_prefix :: Int              # largest k with reduce_mask[1:k] all true
    blocksize    :: Int              # pairwise blocksize (heuristic)
    tile_B       :: Int              # remainder-lane batch
    unroll_U     :: Int              # 8 or 16 (commutative fast path)
    fast_linear  :: Bool             # has_fast_linear_indexing(A)
    commutative  :: Bool             # type-aware result
    widening     :: Bool             # from _mapreduce_might_widen
    stream_ok    :: Bool             # see streaming kernel section
    strides      :: NTuple{N,Int}    # linear strides (if fast_linear)
    firsts       :: NTuple{N,Int}    # axes(A,d).first
    T_el         :: Type{T}          # eltype(A)
end

"""
    _use_pairwise_reduction(plan, contig_len, rem_len) :: Bool

Decide if we should use the naive pairwise reduction instead of the blocked
remainder processing. When the remainder dimension count is large, it's more
efficient to do a single pairwise sweep over all inner dimensions.
"""
@inline function _use_pairwise_reduction(plan::MRPlan, contig_len::Int, rem_len::Int)
    # The blocked path is good when remainder is small. For large remainders,
    # one call per lane is too expensive versus a single big pairwise.
    # If the remainder exceeds the pairwise blocksize, naive is better
    return rem_len > max(plan.blocksize, Base.MIN_BLOCK)
end


# --- Linear-index helpers (array-generic, no pointer math) --------------------
# We use column-major linearization with offsets from axes' first indices.

@inline _axes_first(A) = ntuple(d->first(axes(A,d)), ndims(A))
@inline _axes_len(A)   = ntuple(d->length(axes(A,d)), ndims(A))

# Build linear strides s where s[1]=1, s[d]=s[d-1]*len[d-1] using compile-time recursion
@inline _lin_strides_from_lengths(::Tuple{}) = ()
@inline _lin_strides_from_lengths(len::Tuple{Int}) = (1,)
@inline function _lin_strides_from_lengths(len::NTuple{N,Int}) where {N}
    (1, _lin_strides_tail(first(len), tail(len))...)
end

@inline _lin_strides_tail(prev::Int, ::Tuple{}) = ()
@inline function _lin_strides_tail(prev::Int, len::Tuple)
    (prev, _lin_strides_tail(prev * first(len), Base.tail(len))...)
end


# Compute the *base* linear index for (outer=o, remainder=r) with the first k dims fixed
# to their axis starts; dims>k take from r if reduced, else from o.
@inline function _lin_base_for(o::CartesianIndex{N}, r::CartesianIndex{N},
                               reduce_mask::NTuple{N,Bool}, k::Int,
                               firsts::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    acc = 1
    # Hoist the offset computation for first k dims (they're always at firsts)
    # No contribution since (firsts[d] - firsts[d]) * strides[d] = 0
    @inbounds for d in (k+1):N
        idd = reduce_mask[d] ? r.I[d] : o.I[d]
        acc += (idd - firsts[d]) * strides[d]
    end
    return acc
end

# Fast path check (reuses your trait)
@inline _fast_linear(A) = has_fast_linear_indexing(A)

@inline _contiguous_prefix(mask::NTuple{N,Bool}) where {N} = begin
    k = 0
    @inbounds for d in 1:N
        if mask[d] && d == k + 1
            k += 1
        else
            break
        end
    end
    k
end

# Helper to determine unroll factor based on type and commutativity
@inline function _pick_unroll(mask::NTuple{N,Bool}, ::Type{T}, commutative::Bool) where {N,T}
    if commutative && isbitstype(T) && sizeof(T) ≤ 4 && any(mask)
        return 16
    else
        return 8
    end
end

# Identity element traits for virtual padding
has_identity(::typeof(+), ::Type) = true
has_identity(::typeof(Base.add_sum), ::Type) = true
has_identity(::typeof(*), ::Type) = true
has_identity(::typeof(Base.mul_prod), ::Type) = true
has_identity(::typeof(min), ::Type{T}) where T = isbitstype(T)
has_identity(::typeof(max), ::Type{T}) where T = isbitstype(T)
has_identity(::typeof(&), ::Type{Bool}) = true
has_identity(::typeof(|), ::Type{Bool}) = true
has_identity(::Any, ::Type) = false

identity_element(::typeof(+), ::Type{T}) where T = zero(T)
identity_element(::typeof(Base.add_sum), ::Type{T}) where T = zero(T <: BitSignedSmall ? Int : T <: BitUnsignedSmall ? UInt : T)
identity_element(::typeof(*), ::Type{T}) where T = one(T)
identity_element(::typeof(Base.mul_prod), ::Type{T}) where T = one(T <: BitSignedSmall ? Int : T <: BitUnsignedSmall ? UInt : T)
identity_element(::typeof(min), ::Type{T}) where T = typemax(T)
identity_element(::typeof(max), ::Type{T}) where T = typemin(T)
identity_element(::typeof(&), ::Type{Bool}) = true
identity_element(::typeof(|), ::Type{Bool}) = false

# Helper to pick tile size for remainder processing
@inline function _pick_tile_B(f, op, ::Type{T}) where T
    return sizeof(T) > 4 ? 8 : 16
end

# Check if streaming kernel should be used based on cost analysis
@inline function _should_use_streaming_plan(mask::NTuple{N,Bool}, sizes::NTuple{N,Int}, k::Int) where N
    # Only consider streaming if first dimension is reduced
    mask[1] || return false

    # Calculate the contiguous block length
    block_len = k == 0 ? 0 : prod(sizes[1:k])
    block_len > 0 || return false

    # Calculate costs
    output_size = prod(ifelse(mask[d], 1, sizes[d]) for d in 1:N)
    total_reduced = prod(ifelse(mask[d], sizes[d], 1) for d in 1:N)

    # Standard path cost: output_size * (total_reduced/block_len) non-sequential hits
    standard_cost = output_size * (total_reduced ÷ block_len)

    # Streaming cost: length(A)/block_len sequential runs
    total_size = prod(sizes)
    streaming_cost = total_size ÷ block_len

    # Use bias factor of 8 to prefer standard unless streaming is clearly better
    return streaming_cost * 8 < standard_cost
end

# Main planner function that builds MRPlan
function plan_mapreducedim(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink) where {F,OP,T,N}
    mask = ntuple(d -> d in dims, Val(N))
    sizes = ntuple(d -> size(A,d), Val(N))
    k = _contiguous_prefix(mask)  # inner_prefix

    # Compute traits
    fl = has_fast_linear_indexing(A)
    com = is_commutative_op(op, T)
    wid = _mapreduce_might_widen(f, op, A, init, sink)

    # Compute heuristics
    U = _pick_unroll(mask, T, com)
    B = _pick_tile_B(f, op, T)
    bs = Base.pairwise_blocksize(f, op)

    # Compute strides and firsts if fast linear
    str = fl ? _lin_strides_from_lengths(sizes) : ntuple(_ -> 0, Val(N))
    fst = fl ? _axes_first(A) : ntuple(_ -> 0, Val(N))

    # Check if streaming is beneficial
    s_ok = _should_use_streaming_plan(mask, sizes, k)

    MRPlan(mask, map(!, mask), sizes, k, bs, B, U, fl, com, wid, s_ok, str, fst, T)
end


# Build trailing scalar coordinates (dims k+1..N) with type-stable arity.
# For d > k: pick from `r` if that dim is reduced, otherwise from `o`.
@inline function _trailing(o::CartesianIndex{N}, r::CartesianIndex{N},
                          reduce_mask::NTuple{N,Bool}, ::Val{k}) where {N,k}
    ntuple(Val(N-k)) do j
        d = k + j
        reduce_mask[d] ? r.I[d] : o.I[d]
    end
end

# Initialize result array with first reduction slice
@inline function _seed_result!(sink, R, f, op, A, init, is_inner_dim, contiguous_inner, outer)
    for o in outer
        v = mapreduce_pairwise(f, op, A, init, mergeindices(is_inner_dim, contiguous_inner, o))
        mapreduce_set!(sink, R, o, v)
    end
    return R
end

# Note: _process_commutative_tile! and _process_rowmajor_tile! removed
# These functions were causing allocations due to tile_buf Vector allocation
# Functionality has been inlined directly into the calling functions to avoid allocations

# Helper to select axes for inner/outer dimensions
@inline select_outer_dimensions(A, is_inner_dim) =
    ntuple(d -> is_inner_dim[d] ? axes(A, d) : reduced_index(axes(A, d)), ndims(A))

# for reductions that expand 0 dims to 1
reduced_index(i::OneTo{T}) where {T} = OneTo(one(T))
reduced_index(i::Union{Slice, IdentityUnitRange}) = oftype(i, first(i):first(i))
reduced_index(i::AbstractUnitRange) =
    throw(ArgumentError("no reduced_index for $(typeof(i))"))
reduced_indices(a::AbstractArrayOrBroadcasted, region) = reduced_indices(axes(a), region)

# for reductions that keep 0 dims as 0
reduced_indices0(a::AbstractArray, region) = reduced_indices0(axes(a), region)

function reduced_indices(axs::Indices{N}, region) where N
    _check_valid_region(region)
    ntuple(d -> d in region ? reduced_index(axs[d]) : axs[d], Val(N))
end

function reduced_indices0(axs::Indices{N}, region) where N
    _check_valid_region(region)
    ntuple(d -> d in region && !isempty(axs[d]) ? reduced_index(axs[d]) : axs[d], Val(N))
end

function _check_valid_region(region)
    for d in region
        isa(d, Integer) || throw(ArgumentError("reduced dimension(s) must be integers"))
        Int(d) < 1 && throw(ArgumentError("region dimension(s) must be ≥ 1, got $d"))
    end
end

###### Generic reduction functions #####

# Given two indices or ranges, merge them by dimension akin to a broadcasted `ifelse` over the dims
_sliceall(x) = x[begin]:x[end] # avoid instabilities with OneTos and offset axes
_ifelseslice(b,x,y) = ifelse(b, _sliceall(x), _sliceall(y))
mergeindices(b::NTuple{N,Bool}, x::CartesianIndices{N}, y::CartesianIndices{N}) where {N} =
    CartesianIndices(map(_ifelseslice, b, x.indices, y.indices))
mergeindices(b::NTuple{N,Bool}, x::CartesianIndex{N}, y::CartesianIndices{N}) where {N} =
    CartesianIndices(map(_ifelseslice, b, x.I, y.indices))
mergeindices(b::NTuple{N,Bool}, x::CartesianIndices{N}, y::CartesianIndex{N}) where {N} =
    CartesianIndices(map(_ifelseslice, b, x.indices, y.I))
mergeindices(b::NTuple{N,Bool}, x::CartesianIndex{N}, y::CartesianIndex{N}) where {N} =
    CartesianIndex(map((b,x,y)->ifelse(b, x, y), b, x.I, y.I))

keep_first_trues(::Tuple{}) = ()
keep_first_trues(t) = t[1] ? (true, keep_first_trues(tail(t))...) : ntuple(Returns(false), length(t))

# These functions aren't used in the implementation here, but are used widely in the ecosystem
promote_union(T::Union) = promote_type(promote_union(T.a), promote_union(T.b))
promote_union(T) = T
_realtype(::Type{<:Complex}) = Real
_realtype(::Type{Complex{T}}) where T<:Real = T
_realtype(T::Type) = T
_realtype(::Union{typeof(abs),typeof(abs2)}, T) = _realtype(T)
_realtype(::Any, T) = T

mapreduce_similar(A, e, dims) = similar(A, typeof(e), dims)


"""
    _might_widen(f, op, A, init) -> Bool

Checks if the reduction might widen the type during computation.
Uses a safe approach with conservative fallback for custom containers.
"""
@inline function _might_widen(f, op, A, init)
    # identity fast-path: same-type reductions almost never widen
    f === identity && return false
    T = Base._empty_eltype(A)
    try
        z = _mapreduce_start(f, op, A, init, Base.zero(T))
        return typeof(z) !== typeof(op(z, z))  # quick probe stays in-T if closed
    catch
        return true  # be conservative on error
    end
end

# Sink API for mapreduce - handles both allocation and in-place updates
abstract type _MRSink end

# Forward declare in-place sink for type dispatch
struct _MRInPlaceSink{A,OP,UpdateVal} <: _MRSink
    A::A
    op::OP
end

_mapreduce_might_widen(f, op, A, init, sink::_MRInPlaceSink) = false
_mapreduce_might_widen(f, op, A, init, _) = _might_widen(f, op, A, init)

"Allocate a fresh destination with given element value and axes."
mapreduce_allocate(::_MRSink, e, axes_tuple) = error("not implemented")

"Set first value at index `I`."
mapreduce_set!(::_MRSink, R, I, val) = error("not implemented")

"Accumulate subsequent value at index `I`."
mapreduce_accum!(::_MRSink, R, I, val) = error("not implemented")

"Optionally finalize (usually a no-op)."
mapreduce_finish(::_MRSink, R) = R

# _MRAllocSink is already defined in reduce.jl for bootstrap ordering
# Add methods for it here
mapreduce_allocate(s::_MRAllocSink, e, axs) =
    mapreduce_similar(s.proto, e, axs)

@inline @propagate_inbounds mapreduce_set!(::_MRAllocSink, R, I, v) = (R[I] = v)
@inline @propagate_inbounds mapreduce_accum!(::_MRAllocSink, R, I, v) = (R[I] = v)
mapreduce_finish(::_MRAllocSink, R) = R

# Simplified in-place sink implementation
_MRInPlaceSink(A, op; update::Bool=false) =
    _MRInPlaceSink{typeof(A), typeof(op), Val(update)}(A, op)

mapreduce_allocate(s::_MRInPlaceSink, e, axs) = s.A

@inline @propagate_inbounds function mapreduce_set!(s::_MRInPlaceSink{<:Any,<:Any,Val{update}}, R, I, v) where {update}
    if update
        r_val = R[I]
        result = s.op(r_val, v)
        R[I] = result
    else
        R[I] = v
    end
end
@inline @propagate_inbounds function mapreduce_accum!(s::_MRInPlaceSink, R, I, v)
    r_val = R[I]
    result = s.op(r_val, v)
    R[I] = result
end
mapreduce_finish(::_MRInPlaceSink, R) = R


# When performing dimensional reductions over arrays with singleton dimensions, we have
# a choice as to whether that singleton dimenion should be a part of the reduction or not;
# it does not affect the output. It's advantageous to additionally consider _leading_ singleton
# dimensions as part of the reduction as that allows more cases to be considered contiguous;
# but once we've broken contiguity it's more helpful to _ignore_ those cases.
compute_inner_dims(flagged_dims, source_size) =
    flagged_dims[1] || source_size[1] == 1 ?
        (true, compute_inner_dims(tail(flagged_dims), tail(source_size))...) :
        (false, map((flag,sz)->flag && sz != 1, tail(flagged_dims), tail(source_size))...)
compute_inner_dims(::Tuple{}, ::Tuple{}) = ()

# Method for Colon dims - reduce over all dimensions
function mapreducedim(f::F, op::OP, A::AbstractArray{T,N}, init, ::Colon, sink=_MRAllocSink(A)) where {F, OP, T, N}
    return mapreduce_pairwise(f, op, A, init)
end

# One dispatcher for everything - clean, decision-based architecture
function mapreducedim(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink=_MRAllocSink(A)) where {F, OP, T, N}

    # Build the plan once with all decisions
    P = plan_mapreducedim(f, op, A, init, dims, sink)

    # 1) Must use naive pairwise for small arrays?
    if length(A) ≤ P.blocksize
        return reduce_whole_inner(f, op, A, init, dims, sink, P)
    end

    # 2) Widening or "inner is fully contiguous" → naïve blocked
    if P.widening || P.inner_prefix == count(P.reduce_mask)
        return reduce_naive_blocked(f, op, A, init, dims, sink, P)
    end

    # 3) Streaming in memory order if it wins
    if P.stream_ok
        return reduce_streaming(f, op, A, init, dims, sink, P)
    end

    # 4) Otherwise: col-major or row-major kernel by whether dim1 is reduced
    if P.reduce_mask[1]
        return reduce_colmajor(f, op, A, init, dims, sink, P)
    else
        return reduce_rowmajor(f, op, A, init, dims, sink, P)
    end
end

# Kernel 5.a: Whole-inner pairwise (fallback & small cases)
function reduce_whole_inner(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink, P::MRPlan{N,T}) where {F,OP,T,N}
    # For small arrays, just use mapreduce_pairwise over the appropriate indices
    outer = CartesianIndices(reduced_indices(A, dims))
    inner = CartesianIndices(select_outer_dimensions(A, P.reduce_mask))

    # Need to get an initial value for allocation
    # For empty case, this will be handled by mapreduce_pairwise
    if !isempty(outer) && !isempty(inner)
        # Get a sample value for allocation
        first_inds = mergeindices(P.reduce_mask, inner, first(outer))
        v0 = mapreduce_pairwise(f, op, A, init, first_inds)
    else
        # Empty array - use standard approach
        v0 = _mapreduce_start(f, op, A, init)
    end
    R = mapreduce_allocate(sink, v0, axes(outer))

    for o in outer
        inds = mergeindices(P.reduce_mask, inner, o)
        v = mapreduce_pairwise(f, op, A, init, inds)
        mapreduce_set!(sink, R, o, v)
    end

    return mapreduce_finish(sink, R)
end

# Kernel 5.b: Naïve blocked when inner is fully contiguous or widening is likely
function reduce_naive_blocked(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink, P::MRPlan{N,T}) where {F,OP,T,N}
    outer = CartesianIndices(reduced_indices(A, dims))

    # When inner is fully contiguous, use linear ranges
    if P.inner_prefix > 0 && P.fast_linear
        block_len = prod(P.sizes[1:P.inner_prefix])
        i0 = first(LinearIndices(A))

        v0 = _mapreduce_start(f, op, A, init)
        R = mapreduce_allocate(sink, v0, axes(outer))

        for (idx, o) in enumerate(outer)
            off = block_len * (idx - 1)
            v = mapreduce_pairwise(f, op, A, init, (i0 + off):(i0 + off + block_len - 1))
            mapreduce_set!(sink, R, o, v)
        end

        return mapreduce_finish(sink, R)
    else
        # Fallback: use CartesianIndices
        inner = CartesianIndices(select_outer_dimensions(A, P.reduce_mask))

        v0 = _mapreduce_start(f, op, A, init)
        R = mapreduce_allocate(sink, v0, axes(outer))

        for o in outer
            v = mapreduce_pairwise(f, op, A, init, mergeindices(P.reduce_mask, inner, o))
            mapreduce_set!(sink, R, o, v)
        end

        return mapreduce_finish(sink, R)
    end
end

# Kernel 5.c: Col-major (dim 1 reduced, but remainder dims exist)
function reduce_colmajor(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink, P::MRPlan{N,T}) where {F,OP,T,N}
    outer = CartesianIndices(reduced_indices(A, dims))
    inner = CartesianIndices(select_outer_dimensions(A, P.reduce_mask))

    # Split into contiguous prefix and discontiguous remainder
    is_contiguous_inner = keep_first_trues(P.reduce_mask)
    contiguous_inner = mergeindices(is_contiguous_inner, inner, first(inner))
    discontiguous_inner = mergeindices(is_contiguous_inner, first(inner), inner)

    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, axes(outer))

    # Seed R with first remainder slice
    if P.fast_linear && P.inner_prefix > 0
        block_len = prod(P.sizes[1:P.inner_prefix])
        r0 = first(discontiguous_inner)

        # Pre-compute r0 contribution for dimensions > k (hoisted)
        r0_contrib = 1
        @inbounds for d in (P.inner_prefix+1):N
            if P.reduce_mask[d]
                r0_contrib += (r0.I[d] - P.firsts[d]) * P.strides[d]
            end
        end

        for o in outer
            # Compute outer contribution and combine with hoisted r0 contribution
            base = r0_contrib
            @inbounds for d in (P.inner_prefix+1):N
                if !P.reduce_mask[d]
                    base += (o.I[d] - P.firsts[d]) * P.strides[d]
                end
            end
            v = mapreduce_pairwise(f, op, A, init, base:base+block_len-1)
            mapreduce_set!(sink, R, o, v)
        end

        # Process remainder lanes
        if P.commutative && length(discontiguous_inner) > 1
            # Commutative: tile the remainder processing
            tile_buffer = Vector{CartesianIndex{N}}(undef, P.tile_B)

            for dci in Iterators.drop(discontiguous_inner, 1)
                # Process in tiles - simplified version for now
                for o in outer
                    # Compute inner contribution
                    inner_base = 1
                    @inbounds for d in (P.inner_prefix+1):N
                        if P.reduce_mask[d]
                            inner_base += (dci.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    # Combine with outer contribution
                    base = inner_base
                    @inbounds for d in (P.inner_prefix+1):N
                        if !P.reduce_mask[d]
                            base += (o.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    val = mapreduce_pairwise(f, op, A, init, base:base+block_len-1)
                    mapreduce_accum!(sink, R, o, val)
                end
            end
        else
            # Non-commutative: strict left-to-right
            for dci in Iterators.drop(discontiguous_inner, 1)
                for o in outer
                    # Similar computation but maintaining order
                    inner_base = 1
                    @inbounds for d in (P.inner_prefix+1):N
                        if P.reduce_mask[d]
                            inner_base += (dci.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    base = inner_base
                    @inbounds for d in (P.inner_prefix+1):N
                        if !P.reduce_mask[d]
                            base += (o.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    val = mapreduce_pairwise(f, op, A, init, base:base+block_len-1)
                    r_val = @inbounds R[o]
                    result = op(r_val, val)
                    @inbounds R[o] = result
                end
            end
        end
    else
        # Fallback for non-fast-linear arrays
        for o in outer
            v = mapreduce_pairwise(f, op, A, init, mergeindices(is_contiguous_inner, contiguous_inner, o))
            mapreduce_set!(sink, R, o, v)
        end

        for dci in Iterators.drop(discontiguous_inner, 1)
            for o in outer
                i = mergeindices(P.reduce_mask, dci, o)
                val = mapreduce_pairwise(f, op, A, init, mergeindices(is_contiguous_inner, contiguous_inner, i))
                r_val = @inbounds R[o]
                result = op(r_val, val)
                @inbounds R[o] = result
            end
        end
    end

    return mapreduce_finish(sink, R)
end

# Kernel 5.d: Row-major (dim 1 kept)
function reduce_rowmajor(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink, P::MRPlan{N,T}) where {F,OP,T,N}
    outer = CartesianIndices(reduced_indices(A, dims))
    inner = CartesianIndices(select_outer_dimensions(A, P.reduce_mask))

    # Initialize with first inner index
    i1 = first(inner)
    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, axes(outer))

    if P.fast_linear
        # Pre-compute i1 inner contribution
        i1_contrib = 1
        @inbounds for d in 1:N
            if P.reduce_mask[d]
                i1_contrib += (i1.I[d] - P.firsts[d]) * P.strides[d]
            end
        end

        for o in outer
            # Build linear index for (o, i1)
            acc = i1_contrib
            @inbounds for d in 1:N
                if !P.reduce_mask[d]
                    acc += (o.I[d] - P.firsts[d]) * P.strides[d]
                end
            end
            a_val = @inbounds A[acc]
            v = _mapreduce_start(f, op, A, init, f(a_val))
            mapreduce_set!(sink, R, o, v)
        end

        # Process remaining inner indices
        if P.commutative
            # Commutative: tile the processing
            tile_buffer = Vector{CartesianIndex{N}}(undef, P.tile_B)

            for i in Iterators.drop(inner, 1)
                # Simplified tiling for now
                for o in outer
                    # Compute indices
                    inner_acc = 1
                    @inbounds for d in 1:N
                        if P.reduce_mask[d]
                            inner_acc += (i.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    outer_acc = 1
                    @inbounds for d in 1:N
                        if !P.reduce_mask[d]
                            outer_acc += (o.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end

                    acc = outer_acc + inner_acc - 1
                    a_val = @inbounds A[acc]
                    v = f(a_val)
                    mapreduce_accum!(sink, R, o, v)
                end
            end
        else
            # Non-commutative: strict left-to-right recurrence
            for i in Iterators.drop(inner, 1)
                # Pre-compute inner index contribution once per i
                inner_acc = 1
                @inbounds for d in 1:N
                    if P.reduce_mask[d]
                        inner_acc += (i.I[d] - P.firsts[d]) * P.strides[d]
                    end
                end

                for o in outer
                    outer_acc = 1
                    @inbounds for d in 1:N
                        if !P.reduce_mask[d]
                            outer_acc += (o.I[d] - P.firsts[d]) * P.strides[d]
                        end
                    end
                    acc = outer_acc + inner_acc - 1
                    a_val = @inbounds A[acc]
                    r_val = @inbounds R[o]
                    result = op(r_val, f(a_val))
                    @inbounds R[o] = result
                end
            end
        end
    else
        # Fallback for non-fast-linear arrays
        for o in outer
            idx = mergeindices(P.reduce_mask, i1, o)
            a_val = @inbounds A[idx]
            v = _mapreduce_start(f, op, A, init, f(a_val))
            mapreduce_set!(sink, R, o, v)
        end

        for i in Iterators.drop(inner, 1)
            for o in outer
                iA = mergeindices(P.reduce_mask, i, o)
                a_val = @inbounds A[iA]
                val = f(a_val)
                r_val = @inbounds R[o]
                result = op(r_val, val)
                @inbounds R[o] = result
            end
        end
    end

    return mapreduce_finish(sink, R)
end

# Kernel 6: Streaming kernel (memory order) with cost-based selection
function reduce_streaming(f::F, op::OP, A::AbstractArray{T,N}, init, dims, sink, P::MRPlan{N,T}) where {F,OP,T,N}
    # Build and initialize output
    output_axes = reduced_indices(A, dims)
    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, output_axes)

    # Initialize output elements
    for I in CartesianIndices(R)
        mapreduce_set!(sink, R, I, v0)
    end

    # Process in memory order
    if P.fast_linear && has_fast_linear_indexing(R)
        output_shape = size(R)
        output_strides = _lin_strides_from_lengths(output_shape)

        # Find the largest contiguous block we can process at once
        contiguous_block = 1
        first_non_reduced_dim = N + 1
        for d in 1:N
            if P.reduce_mask[d]
                contiguous_block *= P.sizes[d]
            else
                first_non_reduced_dim = d
                break
            end
        end

        # Pre-compute stride products for non-reduced dimensions
        output_stride_multipliers = ntuple(d ->
            d < first_non_reduced_dim ? 0 :
            (P.reduce_mask[d] ? 0 : output_strides[d]), Val(N))

        # Process the array in memory order
        if contiguous_block > 1
            # We can process contiguous blocks efficiently
            n_blocks = length(A) ÷ contiguous_block

            # Iterate through all blocks
            @inbounds for block_idx in 0:(n_blocks-1)
                # Compute the starting linear index for this block
                base_idx = block_idx * contiguous_block + 1

                # Convert block index to coordinates in the non-reduced dimensions
                remaining = block_idx
                out_idx = 1
                dim_idx = 1

                # Skip the contiguous reduced dimensions at the start
                for d in 1:N
                    if P.reduce_mask[d]
                        dim_idx = d + 1
                    else
                        break
                    end
                end

                # Compute output index based on remaining dimensions using hoisted multipliers
                for d in dim_idx:N
                    coord = (remaining % P.sizes[d]) + 1
                    remaining ÷= P.sizes[d]
                    out_idx += (coord - 1) * output_stride_multipliers[d]
                end

                # Process the entire contiguous block and accumulate to single output position
                for offset in 0:(contiguous_block-1)
                    val = f(A[base_idx + offset])
                    R[out_idx] = op(R[out_idx], val)
                end
            end
        else
            # No contiguous blocks to exploit, process element by element
            @inbounds for lin_idx in 1:length(A)
                val = f(A[lin_idx])

                # Map to output index using hoisted multipliers
                remaining = lin_idx - 1
                out_idx = 1
                for d in 1:N
                    coord = (remaining % P.sizes[d]) + 1
                    remaining ÷= P.sizes[d]
                    out_idx += (coord - 1) * output_stride_multipliers[d]
                end

                R[out_idx] = op(R[out_idx], val)
            end
        end
    else
        # Fallback for arrays without fast linear indexing
        @inbounds for I_in in CartesianIndices(A)
            I_out = CartesianIndex(ntuple(d -> P.reduce_mask[d] ? 1 : I_in[d], Val(N)))
            val = f(A[I_in])
            R[I_out] = op(R[I_out], val)
        end
    end

    return mapreduce_finish(sink, R)
end




"""
    mapreduce!(f, op, R::AbstractArray, A::AbstractArrayOrBroadcasted; [init], update=false)

Compute `mapreduce(f, op, A; init, dims)` where `dims` are the singleton dimensions of `R`, storing the result into `R`.

# Arguments
- `update::Bool=false`: If `true`, combines the reduction result with existing values in `R` using `op`.
                         If `false` (default), overwrites `R` with the reduction result.

# Examples
```jldoctest
julia> R = zeros(1, 4);

julia> A = [1 2 3 4; 5 6 7 8];

julia> mapreduce!(identity, +, R, A);  # update=false overwrites R

julia> R
1×4 Matrix{Float64}:
 6.0  8.0  10.0  12.0

julia> mapreduce!(identity, +, R, A; update=true);  # update=true combines with existing R

julia> R
1×4 Matrix{Float64}:
 12.0  16.0  20.0  24.0
```

!!! note
    When `update=false`, the previous values in `R` are completely ignored and overwritten.
"""
function mapreduce!(f, op, R::AbstractArray, A::AbstractArrayOrBroadcasted; init=_InitialValue(), update=false)
    # Check dimension compatibility
    if ndims(R) > ndims(A) || !all(d->size(R, d) == 1 || axes(R, d) == axes(A, d), 1:max(ndims(R), ndims(A)))
        throw(DimensionMismatch())
    end
    # Compute dims from R - the dimensions where R has size 1 are the reduction dimensions
    # We need to consider up to ndims(A), not just ndims(R)
    dims = if ndims(R) < ndims(A)
        # R has fewer dims, so dims beyond ndims(R) are implicitly reduced
        tuple((d for d in 1:ndims(A) if d > ndims(R) || size(R, d) == 1)...)
    else
        # R has same or more dims as A
        tuple((d for d in 1:ndims(A) if size(R, d) == 1)...)
    end
    return mapreducedim(f, op, A, init, dims, _MRInPlaceSink(R, op; update=update))
end

"""
    reduce!(op, R::AbstractArray, A::AbstractArrayOrBroadcasted; [init], update=false)

Compute `reduce(op, A; init, dims)` where `dims` are the singleton dimensions of `R`, storing the result into `R`.

# Arguments
- `update::Bool=false`: If `true`, combines the reduction result with existing values in `R` using `op`.
                         If `false` (default), overwrites `R` with the reduction result.

!!! note
    When `update=false`, the previous values in `R` are completely ignored and overwritten.
"""
reduce!(op, R::AbstractArray, A::AbstractArrayOrBroadcasted; init=_InitialValue(), update=false) = mapreduce!(identity, op, R, A; init, update)

##### Specific reduction functions #####

"""
    count([f=identity,] A::AbstractArray; dims=:)

Count the number of elements in `A` for which `f` returns `true` over the given
dimensions.

!!! compat "Julia 1.5"
    `dims` keyword was added in Julia 1.5.

!!! compat "Julia 1.6"
    `init` keyword was added in Julia 1.6.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> count(<=(2), A, dims=1)
1×2 Matrix{Int64}:
 1  1

julia> count(<=(2), A, dims=2)
2×1 Matrix{Int64}:
 2
 0
```
"""
count(A::AbstractArrayOrBroadcasted; dims=:, init=0) = count(identity, A; dims, init)
count(f, A::AbstractArrayOrBroadcasted; dims=:, init=0) = _count(f, A, dims, init)

_count(f, A::AbstractArrayOrBroadcasted, dims::Colon, init) = _simple_count(f, A, init)
_count(f, A::AbstractArrayOrBroadcasted, dims, init) = mapreduce(_bool(f), add_sum, A; dims, init)

"""
    count!([f=identity,] r, A)

Count the number of elements in `A` for which `f` returns `true` over the
singleton dimensions of `r`, writing the result into `r` in-place.

$(_DOCS_ALIASING_WARNING)

!!! compat "Julia 1.5"
    inplace `count!` was added in Julia 1.5.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> count!(<=(2), [1 1], A)
1×2 Matrix{Int64}:
 1  1

julia> count!(<=(2), [1; 1], A)
2-element Vector{Int64}:
 2
 0
```
"""
count!(r::AbstractArray, A::AbstractArrayOrBroadcasted; init::Bool=true) = count!(identity, r, A; init=init)
count!(f, r::AbstractArray, A::AbstractArrayOrBroadcasted; init::Bool=true) =
    mapreduce!(_bool(f), add_sum, r, A; update=!init)

"""
    sum(A::AbstractArray; dims)

Sum elements of an array over the given dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> sum(A, dims=1)
1×2 Matrix{Int64}:
 4  6

julia> sum(A, dims=2)
2×1 Matrix{Int64}:
 3
 7
```
"""
sum(A::AbstractArray; dims)

"""
    sum(f, A::AbstractArray; dims)

Sum the results of calling function `f` on each element of an array over the given
dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> sum(abs2, A, dims=1)
1×2 Matrix{Int64}:
 10  20

julia> sum(abs2, A, dims=2)
2×1 Matrix{Int64}:
  5
 25
```
"""
sum(f, A::AbstractArray; dims)

"""
    sum!(r, A)

Sum elements of `A` over the singleton dimensions of `r`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> sum!([1; 1], A)
2-element Vector{Int64}:
 3
 7

julia> sum!([1 1], A)
1×2 Matrix{Int64}:
 4  6
```
"""
sum!(r, A)

"""
    prod(A::AbstractArray; dims)

Multiply elements of an array over the given dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> prod(A, dims=1)
1×2 Matrix{Int64}:
 3  8

julia> prod(A, dims=2)
2×1 Matrix{Int64}:
  2
 12
```
"""
prod(A::AbstractArray; dims)

"""
    prod(f, A::AbstractArray; dims)

Multiply the results of calling the function `f` on each element of an array over the given
dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> prod(abs2, A, dims=1)
1×2 Matrix{Int64}:
 9  64

julia> prod(abs2, A, dims=2)
2×1 Matrix{Int64}:
   4
 144
```
"""
prod(f, A::AbstractArray; dims)

"""
    prod!(r, A)

Multiply elements of `A` over the singleton dimensions of `r`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> prod!([1; 1], A)
2-element Vector{Int64}:
  2
 12

julia> prod!([1 1], A)
1×2 Matrix{Int64}:
 3  8
```
"""
prod!(r, A)

"""
    maximum(A::AbstractArray; dims)

Compute the maximum value of an array over the given dimensions. See also the
[`max(a,b)`](@ref) function to take the maximum of two or more arguments,
which can be applied elementwise to arrays via `max.(a,b)`.

See also: [`maximum!`](@ref), [`extrema`](@ref), [`findmax`](@ref), [`argmax`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> maximum(A, dims=1)
1×2 Matrix{Int64}:
 3  4

julia> maximum(A, dims=2)
2×1 Matrix{Int64}:
 2
 4
```
"""
maximum(A::AbstractArray; dims)

"""
    maximum(f, A::AbstractArray; dims)

Compute the maximum value by calling the function `f` on each element of an array over the given
dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> maximum(abs2, A, dims=1)
1×2 Matrix{Int64}:
 9  16

julia> maximum(abs2, A, dims=2)
2×1 Matrix{Int64}:
  4
 16
```
"""
maximum(f, A::AbstractArray; dims)

"""
    maximum!(r, A)

Compute the maximum value of `A` over the singleton dimensions of `r`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> maximum!([1; 1], A)
2-element Vector{Int64}:
 2
 4

julia> maximum!([1 1], A)
1×2 Matrix{Int64}:
 3  4
```
"""
maximum!(r, A)

"""
    minimum(A::AbstractArray; dims)

Compute the minimum value of an array over the given dimensions. See also the
[`min(a,b)`](@ref) function to take the minimum of two or more arguments,
which can be applied elementwise to arrays via `min.(a,b)`.

See also: [`minimum!`](@ref), [`extrema`](@ref), [`findmin`](@ref), [`argmin`](@ref).

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> minimum(A, dims=1)
1×2 Matrix{Int64}:
 1  2

julia> minimum(A, dims=2)
2×1 Matrix{Int64}:
 1
 3
```
"""
minimum(A::AbstractArray; dims)

"""
    minimum(f, A::AbstractArray; dims)

Compute the minimum value by calling the function `f` on each element of an array over the given
dimensions.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> minimum(abs2, A, dims=1)
1×2 Matrix{Int64}:
 1  4

julia> minimum(abs2, A, dims=2)
2×1 Matrix{Int64}:
 1
 9
```
"""
minimum(f, A::AbstractArray; dims)

"""
    minimum!(r, A)

Compute the minimum value of `A` over the singleton dimensions of `r`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> minimum!([1; 1], A)
2-element Vector{Int64}:
 1
 3

julia> minimum!([1 1], A)
1×2 Matrix{Int64}:
 1  2
```
"""
minimum!(r, A)

"""
    extrema(A::AbstractArray; dims) -> Array{Tuple}

Compute the minimum and maximum elements of an array over the given dimensions.

See also: [`minimum`](@ref), [`maximum`](@ref), [`extrema!`](@ref).

# Examples
```jldoctest
julia> A = reshape(Vector(1:2:16), (2,2,2))
2×2×2 Array{Int64, 3}:
[:, :, 1] =
 1  5
 3  7

[:, :, 2] =
  9  13
 11  15

julia> extrema(A, dims = (1,2))
1×1×2 Array{Tuple{Int64, Int64}, 3}:
[:, :, 1] =
 (1, 7)

[:, :, 2] =
 (9, 15)
```
"""
extrema(A::AbstractArray; dims)

"""
    extrema(f, A::AbstractArray; dims) -> Array{Tuple}

Compute the minimum and maximum of `f` applied to each element in the given dimensions
of `A`.

!!! compat "Julia 1.2"
    This method requires Julia 1.2 or later.
"""
extrema(f, A::AbstractArray; dims)

"""
    extrema!(r, A)

Compute the minimum and maximum value of `A` over the singleton dimensions of `r`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

!!! compat "Julia 1.8"
    This method requires Julia 1.8 or later.

# Examples
```jldoctest
julia> A = [1 2; 3 4]
2×2 Matrix{Int64}:
 1  2
 3  4

julia> extrema!([(1, 1); (1, 1)], A)
2-element Vector{Tuple{Int64, Int64}}:
 (1, 2)
 (3, 4)

julia> extrema!([(1, 1);; (1, 1)], A)
1×2 Matrix{Tuple{Int64, Int64}}:
 (1, 3)  (2, 4)
```
"""
extrema!(r, A)

"""
    all(A; dims)

Test whether all values along the given dimensions of an array are `true`.

# Examples
```jldoctest
julia> A = [true false; true true]
2×2 Matrix{Bool}:
 1  0
 1  1

julia> all(A, dims=1)
1×2 Matrix{Bool}:
 1  0

julia> all(A, dims=2)
2×1 Matrix{Bool}:
 0
 1
```
"""
all(A::AbstractArray; dims)

"""
    all(p, A; dims)

Determine whether predicate `p` returns `true` for all elements along the given dimensions of an array.

# Examples
```jldoctest
julia> A = [1 -1; 2 2]
2×2 Matrix{Int64}:
 1  -1
 2   2

julia> all(i -> i > 0, A, dims=1)
1×2 Matrix{Bool}:
 1  0

julia> all(i -> i > 0, A, dims=2)
2×1 Matrix{Bool}:
 0
 1
```
"""
all(::Function, ::AbstractArray; dims)

"""
    all!(r, A)

Test whether all values in `A` along the singleton dimensions of `r` are `true`, and write results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [true false; true false]
2×2 Matrix{Bool}:
 1  0
 1  0

julia> all!(Bool[1; 1], A)
2-element Vector{Bool}:
 0
 0

julia> all!(Bool[1 1], A)
1×2 Matrix{Bool}:
 1  0
```
"""
all!(r, A)

"""
    any(A; dims)

Test whether any values along the given dimensions of an array are `true`.

# Examples
```jldoctest
julia> A = [true false; true false]
2×2 Matrix{Bool}:
 1  0
 1  0

julia> any(A, dims=1)
1×2 Matrix{Bool}:
 1  0

julia> any(A, dims=2)
2×1 Matrix{Bool}:
 1
 1
```
"""
any(::AbstractArray; dims)

"""
    any(p, A; dims)

Determine whether predicate `p` returns `true` for any elements along the given dimensions of an array.

# Examples
```jldoctest
julia> A = [1 -1; 2 -2]
2×2 Matrix{Int64}:
 1  -1
 2  -2

julia> any(i -> i > 0, A, dims=1)
1×2 Matrix{Bool}:
 1  0

julia> any(i -> i > 0, A, dims=2)
2×1 Matrix{Bool}:
 1
 1
```
"""
any(::Function, ::AbstractArray; dims)

"""
    any!(r, A)

Test whether any values in `A` along the singleton dimensions of `r` are `true`, and write
results to `r`.

$(_DOCS_ALIASING_WARNING)

# Examples
```jldoctest
julia> A = [true false; true false]
2×2 Matrix{Bool}:
 1  0
 1  0

julia> any!(Bool[1; 1], A)
2-element Vector{Bool}:
 1
 1

julia> any!(Bool[1 1], A)
1×2 Matrix{Bool}:
 1  0
```
"""
any!(r, A)

for (fname, _fname, op) in [(:sum,     :_sum,     :add_sum), (:prod,    :_prod,    :mul_prod),
                            (:maximum, :_maximum, :max),     (:minimum, :_minimum, :min),
                            (:extrema, :_extrema, :_extrema_rf)]
    mapf = fname === :extrema ? :(ExtremaMap(f)) : :f
    @eval begin
        # User-facing methods with keyword arguments
        @inline ($fname)(a::AbstractArray; dims=:, kw...) = ($_fname)(a, dims; kw...)
        @inline ($fname)(f, a::AbstractArray; dims=:, kw...) = ($_fname)(f, a, dims; kw...)

        # Underlying implementations using dispatch
        ($_fname)(a, ::Colon; kw...) = ($_fname)(identity, a, :; kw...)
        ($_fname)(f, a, ::Colon; kw...) = mapreduce($mapf, $op, a; kw...)
    end
end

any(a::AbstractArray; dims=:)              = _any(a, dims)
any(f::Function, a::AbstractArray; dims=:) = _any(f, a, dims)
_any(a, ::Colon)                           = _any(identity, a, :)
all(a::AbstractArray; dims=:)              = _all(a, dims)
all(f::Function, a::AbstractArray; dims=:) = _all(f, a, dims)
_all(a, ::Colon)                           = _all(identity, a, :)

for (fname, op) in [(:sum, :add_sum), (:prod, :mul_prod),
                    (:maximum, :max), (:minimum, :min),
                    (:all, :and_all), (:any, :or_any),
                    (:extrema, :_extrema_rf)]
    fname! = Symbol(fname, '!')
    _fname = Symbol('_', fname)
    mapf = fname === :extrema ? :(ExtremaMap(f)) : :f
    @eval begin
        $(fname!)(f::Function, r::AbstractArray, A::AbstractArray; init::Bool=true) =
            mapreduce!($mapf, $op, r, A; update=!init)
        $(fname!)(r::AbstractArray, A::AbstractArray; init::Bool=true) = $(fname!)(identity, r, A; init=init)

        $(_fname)(A, dims; kw...)    = $(_fname)(identity, A, dims; kw...)
        $(_fname)(f, A, dims; kw...) = mapreduce($mapf, $(op), A; dims=dims, kw...)
    end
end

##### findmin & findmax #####
# The initial values of Rval are not used if the corresponding indices in Rind are 0.
#
function findminmax!(f, op, Rval, Rind, A::AbstractArray{T,N}) where {T,N}
    (isempty(Rval) || isempty(A)) && return Rval, Rind
    lsiz = check_reducedims(Rval, A)
    for i = 1:N
        axes(Rval, i) == axes(Rind, i) || throw(DimensionMismatch("Find-reduction: outputs must have the same indices"))
    end
    # If we're reducing along dimension 1, for efficiency we can make use of a temporary.
    # Otherwise, keep the result in Rval/Rind so that we traverse A in storage order.
    indsAt, indsRt = safe_tail(axes(A)), safe_tail(axes(Rval))
    keep, Idefault = Broadcast.shapeindexer(indsRt)
    ks = keys(A)
    y = iterate(ks)
    zi = zero(eltype(ks))
    if reducedim1(Rval, A)
        i1 = first(axes1(Rval))
        for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            @inbounds tmpRv = Rval[i1,IR]
            @inbounds tmpRi = Rind[i1,IR]
            for i in axes(A,1)
                k, kss = y::Tuple
                tmpAv_raw = @inbounds A[i,IA]
                tmpAv = f(tmpAv_raw)
                if tmpRi == zi || op(tmpRv, tmpAv)
                    tmpRv = tmpAv
                    tmpRi = k
                end
                y = iterate(ks, kss)
            end
            @inbounds Rval[i1,IR] = tmpRv
            @inbounds Rind[i1,IR] = tmpRi
        end
    else
        for IA in CartesianIndices(indsAt)
            IR = Broadcast.newindex(IA, keep, Idefault)
            for i in axes(A, 1)
                k, kss = y::Tuple
                tmpAv_raw = @inbounds A[i,IA]
                tmpAv = f(tmpAv_raw)
                @inbounds tmpRv = Rval[i,IR]
                @inbounds tmpRi = Rind[i,IR]
                if tmpRi == zi || op(tmpRv, tmpAv)
                    @inbounds Rval[i,IR] = tmpAv
                    @inbounds Rind[i,IR] = k
                end
                y = iterate(ks, kss)
            end
        end
    end
    Rval, Rind
end

struct PairsArray{T,N,A} <: AbstractArray{T,N}
    array::A
end
PairsArray(array::AbstractArray{T,N}) where {T, N} = PairsArray{Tuple{keytype(array),T}, N, typeof(array)}(array)
const PairsVector{T,A} = PairsArray{T, 1, A}
IndexStyle(::PairsVector) = IndexLinear()
IndexStyle(::PairsArray) = IndexCartesian()
size(P::PairsArray) = size(P.array)
axes(P::PairsArray) = axes(P.array)
@inline function getindex(P::PairsVector, i::Int)
    @boundscheck checkbounds(P, i)
    @inbounds (i, P.array[i])
end
@inline function getindex(P::PairsArray{<:Any,N}, I::CartesianIndex{N}) where {N}
    @boundscheck checkbounds(P, I)
    @inbounds (I, P.array[I])
end
@propagate_inbounds getindex(P::PairsVector, i::CartesianIndex{1}) = P[i.I[1]]
@propagate_inbounds getindex(P::PairsArray{<:Any,N}, I::Vararg{Int, N}) where {N} = P[CartesianIndex(I)]
mapreduce_similar(P::PairsArray, ::Type{T}, dims) where {T} = mapreduce_similar(P.array, T, dims)

# Use an ad-hoc specialized StructArray to allow in-place AoS->SoA transform
struct ZippedArray{T,N,Style,A,B} <: AbstractArray{T,N}
    first::A
    second::B
end
function ZippedArray(A::AbstractArray{T,N},B::AbstractArray{S,N}) where {T,S,N}
    axes(A) == axes(B) || throw(DimensionMismatch("both arrays must have the same shape"))
    # TODO: It'd be better if we could transform a Tuple{Int, Union{Int, Missing}} to Union{Tuple{Int,Int}, Tuple{Int, Missing}}
    ZippedArray{Tuple{T,S},N,IndexStyle(A,B),typeof(A),typeof(B)}(A,B)
end
size(Z::ZippedArray) = size(Z.first)
axes(Z::ZippedArray) = axes(Z.first)
IndexStyle(::ZippedArray{<:Any,<:Any,Style}) where {Style} = Style
@inline function getindex(Z::ZippedArray, I::Int...)
    @boundscheck checkbounds(Z, I...)
    @inbounds (Z.first[I...], Z.second[I...])
end
@propagate_inbounds setindex!(Z::ZippedArray{T}, v, I::Int...) where {T} = setindex!(Z, convert(T, v), I...)
@inline function setindex!(Z::ZippedArray{T}, v::T, I::Int...) where {T}
    @boundscheck checkbounds(Z, I...)
    @inbounds Z.first[I...] = v[1]
    @inbounds Z.second[I...] = v[2]
    return Z
end
_unzip(Z::ZippedArray) = (Z.first, Z.second)
_unzip(A::AbstractArray) = ([a[1] for a in A], [a[2] for a in A])

_transform_pair(f) = x-> (x[1], f(x[2]))
_transform_pair(::Type{F}) where {F} = x-> (x[1], F(x[2]))
_transform_pair(f::typeof(identity)) = f

"""
    findmin!(rval, rind, A) -> (minval, index)

Find the minimum of `A` and the corresponding linear index along singleton
dimensions of `rval` and `rind`, and store the results in `rval` and `rind`.
`NaN` is treated as less than all other values except `missing`.

$(_DOCS_ALIASING_WARNING)
"""
findmin!(rval::AbstractArray, rind::AbstractArray, A::AbstractArray; init::Bool=true) = findmin!(identity, rval, rind, A; init)
function findmin!(f, rval::AbstractArray, rind::AbstractArray, A::AbstractArray;
                  init::Bool=true)
    mapreduce!(_transform_pair(f), (x,y)->ifelse(isgreater(x[2], y[2]), y, x), ZippedArray(rind, rval), PairsArray(A); update=!init)
    return (rval, rind)
end

"""
    findmin(A; dims) -> (minval, index)

For an array input, returns the value and index of the minimum over the given dimensions.
`NaN` is treated as less than all other values except `missing`.

# Examples
```jldoctest
julia> A = [1.0 2; 3 4]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> findmin(A, dims=1)
([1.0 2.0], CartesianIndex{2}[CartesianIndex(1, 1) CartesianIndex(1, 2)])

julia> findmin(A, dims=2)
([1.0; 3.0;;], CartesianIndex{2}[CartesianIndex(1, 1); CartesianIndex(2, 1);;])
```
"""
findmin(A::AbstractArray; dims=:) = _findmin(A, dims)
_findmin(A, dims) = _findmin(identity, A, dims)

"""
    findmin(f, A; dims) -> (f(x), index)

For an array input, returns the value in the codomain and index of the corresponding value
which minimize `f` over the given dimensions.

# Examples
```jldoctest
julia> A = [-1.0 1; -0.5 2]
2×2 Matrix{Float64}:
 -1.0  1.0
 -0.5  2.0

julia> findmin(abs2, A, dims=1)
([0.25 1.0], CartesianIndex{2}[CartesianIndex(2, 1) CartesianIndex(1, 2)])

julia> findmin(abs2, A, dims=2)
([1.0; 0.25;;], CartesianIndex{2}[CartesianIndex(1, 1); CartesianIndex(2, 1);;])
```
"""
findmin(f, A::AbstractArray; dims=:) = _findmin(f, A, dims)

function _findmin(f, A, region)
    if f === identity
        # Fast path with pre-allocated arrays
        axs = reduced_indices(A, region)
        return findmin!(identity, mapreduce_similar(A, eltype(A), axs), mapreduce_similar(A, keytype(A), axs), A)
    else
        P = mapreduce(_transform_pair(f), (x,y)->ifelse(isgreater(x[2], y[2]), y, x), PairsArray(A); dims=region)
        (inds, vals) = _unzip(P)
        return (vals, inds)
    end
end
"""
    findmax!(rval, rind, A) -> (maxval, index)

Find the maximum of `A` and the corresponding linear index along singleton
dimensions of `rval` and `rind`, and store the results in `rval` and `rind`.
`NaN` is treated as greater than all other values except `missing`.

$(_DOCS_ALIASING_WARNING)
"""
findmax!(rval::AbstractArray, rind::AbstractArray, A::AbstractArray; init::Bool=true) = findmax!(identity, rval, rind, A; init)
function findmax!(f, rval::AbstractArray, rind::AbstractArray, A::AbstractArray;
                  init::Bool=true)
    mapreduce!(_transform_pair(f), (x,y)->ifelse(isless(x[2], y[2]), y, x), ZippedArray(rind, rval), PairsArray(A); update=!init)
    return (rval, rind)
end

"""
    findmax(A; dims) -> (maxval, index)

For an array input, returns the value and index of the maximum over the given dimensions.
`NaN` is treated as greater than all other values except `missing`.

# Examples
```jldoctest
julia> A = [1.0 2; 3 4]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> findmax(A, dims=1)
([3.0 4.0], CartesianIndex{2}[CartesianIndex(2, 1) CartesianIndex(2, 2)])

julia> findmax(A, dims=2)
([2.0; 4.0;;], CartesianIndex{2}[CartesianIndex(1, 2); CartesianIndex(2, 2);;])
```
"""
findmax(A::AbstractArray; dims=:) = _findmax(A, dims)
_findmax(A, dims) = _findmax(identity, A, dims)

"""
    findmax(f, A; dims) -> (f(x), index)

For an array input, returns the value in the codomain and index of the corresponding value
which maximize `f` over the given dimensions.

# Examples
```jldoctest
julia> A = [-1.0 1; -0.5 2]
2×2 Matrix{Float64}:
 -1.0  1.0
 -0.5  2.0

julia> findmax(abs2, A, dims=1)
([1.0 4.0], CartesianIndex{2}[CartesianIndex(1, 1) CartesianIndex(2, 2)])

julia> findmax(abs2, A, dims=2)
([1.0; 4.0;;], CartesianIndex{2}[CartesianIndex(1, 1); CartesianIndex(2, 2);;])
```
"""
findmax(f, A::AbstractArray; dims=:) = _findmax(f, A, dims)
function _findmax(f, A, region)
    if f === identity
        axs = reduced_indices(A, region)
        return findmax!(identity, mapreduce_similar(A, eltype(A), axs), mapreduce_similar(A, keytype(A), axs), A)
    else
        P = mapreduce(_transform_pair(f), (x,y)->ifelse(isless(x[2], y[2]), y, x), PairsArray(A); dims=region)
        (inds, vals) = _unzip(P)
        return (vals, inds)
    end
end

"""
    argmin(A; dims) -> indices

For an array input, return the indices of the minimum elements over the given dimensions.
`NaN` is treated as less than all other values except `missing`.

# Examples
```jldoctest
julia> A = [1.0 2; 3 4]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> argmin(A, dims=1)
1×2 Matrix{CartesianIndex{2}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)

julia> argmin(A, dims=2)
2×1 Matrix{CartesianIndex{2}}:
 CartesianIndex(1, 1)
 CartesianIndex(2, 1)
```
"""
argmin(A::AbstractArray; dims=:) = findmin(A; dims=dims)[2]

"""
    argmax(A; dims) -> indices

For an array input, return the indices of the maximum elements over the given dimensions.
`NaN` is treated as greater than all other values except `missing`.

# Examples
```jldoctest
julia> A = [1.0 2; 3 4]
2×2 Matrix{Float64}:
 1.0  2.0
 3.0  4.0

julia> argmax(A, dims=1)
1×2 Matrix{CartesianIndex{2}}:
 CartesianIndex(2, 1)  CartesianIndex(2, 2)

julia> argmax(A, dims=2)
2×1 Matrix{CartesianIndex{2}}:
 CartesianIndex(1, 2)
 CartesianIndex(2, 2)
```
"""
argmax(A::AbstractArray; dims=:) = findmax(A; dims=dims)[2]
