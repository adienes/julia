# This file is a part of Julia. License is MIT: https://julialang.org/license

## Functions to compute the reduced shape

# --- Reduction planning (array-generic) ---------------------------------------

struct ReductionPlan{N}
    reduce_mask :: NTuple{N,Bool}   # dims to reduce
    keep_mask   :: NTuple{N,Bool}   # !reduce_mask
    sizes       :: NTuple{N,Int}
    inner_pack  :: Int              # largest k such that dims 1:k are reduced & contiguous
    inner_bs    :: Int              # pairwise blocksize for inner pack sweeps
    tile_B      :: Int              # lanes to batch across discontiguous remainder dims
end

"""
    _use_pairwise_reduction(plan, contig_len, rem_len) :: Bool

Decide if we should use the naive pairwise reduction instead of the blocked
remainder processing. When the remainder dimension count is large, it's more
efficient to do a single pairwise sweep over all inner dimensions.
"""
@inline function _use_pairwise_reduction(plan::ReductionPlan, contig_len::Int, rem_len::Int)
    # The blocked path is good when remainder is small. For large remainders,
    # one call per lane is too expensive versus a single big pairwise.
    # If the remainder exceeds the pairwise blocksize, naive is better
    return rem_len > max(plan.inner_bs, Base.MIN_BLOCK)
end

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

@inline function _mkplan(f, op, A::AbstractArray, is_inner_dim::NTuple{N,Bool}) where {N}
    ReductionPlan{N}(
        is_inner_dim,
        ntuple(d->!is_inner_dim[d], N),
        ntuple(d->size(A,d), N),
        _contiguous_prefix(is_inner_dim),
        Base.pairwise_blocksize(f, op),
        Base.reduce_tile_size(f, op, eltype(A)),
    )
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

# One-lane contiguous-pack sweep using the specialized commutative kernel
@inline function _lane_comm_sweep(f, op, A, init, contig_inds, o::CartesianIndex{N},
                                 r::CartesianIndex{N}, reduce_mask::NTuple{N,Bool},
                                 ::Val{k}) where {N,k}
    _mapreduce_kernel_commutative(f, op, A, init, contig_inds,
                                 (), _trailing(o, r, reduce_mask, Val(k)))
end

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

# _MRInPlaceSink already defined earlier for type dispatch

_MRInPlaceSink(A, op; update::Bool=false) =
    _MRInPlaceSink{typeof(A), typeof(op), Val(update)}(A, op)

mapreduce_allocate(s::_MRInPlaceSink, e, axs) = s.A

@inline @propagate_inbounds function mapreduce_set!(s::_MRInPlaceSink{<:Any,<:Any,Val{false}()}, R, I, v)
    R[I] = v
end
@inline @propagate_inbounds function mapreduce_set!(s::_MRInPlaceSink{<:Any,<:Any,Val{true}()}, R, I, v)
    R[I] = s.op(R[I], v)
end
@inline @propagate_inbounds function mapreduce_accum!(s::_MRInPlaceSink, R, I, v)
    R[I] = s.op(R[I], v)
end
mapreduce_finish(::_MRInPlaceSink, R) = R

# Helper functions for sink operations with proper inbounds propagation
@inline @propagate_inbounds function _sink_set!(sink::_MRSink, R, I, v)
    mapreduce_set!(sink, R, I, v)
end
@inline @propagate_inbounds function _sink_accum!(sink::_MRSink, R, I, v)
    mapreduce_accum!(sink, R, I, v)
end

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

function mapreducedim(f::F, op::OP, A, init, dims, sink=_MRAllocSink(A)) where {F, OP}
    # Check if we should use streaming kernel for "keep dim 1" case
    if !(1 in dims) && ndims(A) > 1
        # Use streaming kernel for better performance when dim 1 is kept
        return _mapreducedim_stream_firstdim(f, op, A, init, dims, sink)
    end

    if sink isa _MRInPlaceSink
        # We can ignore dims and just trust the output array's axes. Note that the
        # other branch here optimizes for the case where the input array _also_ has
        # a singleton dimension, but we cannot do that here because OffsetArrays
        # supports reductions into differently-offset singleton dimensions. This means
        # we cannot index directly into A with an `outer` index. The output may also have
        # fewer dimensions than A, so we may need to add trailing dims here:
        outer = CartesianIndices(ntuple(d->axes(sink.A, d), ndims(A)))
        is_inner_dim = map(==(1), size(outer))
    else
        is_inner_dim = compute_inner_dims(ntuple(d->d in dims, ndims(A)), size(A))
        outer = CartesianIndices(reduced_indices(A, dims))
    end
    inner = CartesianIndices(map((b,ax)->b ? ax : reduced_index(ax), is_inner_dim, axes(A)))
    n = length(inner)
    # Handle the empty and trivial 1-element cases:
    if (n == 0 || isempty(A))
        # This is broken out of the comprehension to ensure it's called, avoiding an empty Vector{Union{}}
        # Handle empty case directly with sink API
        v = _mapreduce_start(f, op, A, init)
        R = mapreduce_allocate(sink, v, axes(outer))
        for I in outer
            mapreduce_set!(sink, R, I, v)
        end
        return mapreduce_finish(sink, R)
    end
    if n == 1
        # Handle trivial 1-element case with sink API
        v0 = _mapreduce_start(f, op, A, init, A[first(inner)])
        R = mapreduce_allocate(sink, v0, axes(outer))
        for i in outer
            v = _mapreduce_start(f, op, A, init, A[i])
            mapreduce_set!(sink, R, i, v)
        end
        return mapreduce_finish(sink, R)
    end
    # Now there are multiple loop ordering strategies depending upon the `dims`:
    if is_inner_dim == keep_first_trues(is_inner_dim) || _mapreduce_might_widen(f, op, A, init, sink)
        # Column major contiguous reduction! This is the easy case
        return mapreducedim_naive(f, op, A, init, is_inner_dim, inner, outer, sink)
    elseif is_inner_dim[1] # `dims` includes the first dimension
        return mapreducedim_colmajor(f, op, A, init, is_inner_dim, inner, outer, sink)
    else
        return mapreducedim_rowmajor(f, op, A, init, is_inner_dim, inner, outer, sink)
    end
end

linear_size(A, is_inner_dim) = linear_size(IndexStyle(A), A, is_inner_dim)
linear_size(::IndexLinear, A, is_inner_dim) = if is_inner_dim == keep_first_trues(is_inner_dim)
    prod(map((b,sz)->ifelse(b, sz, 1), is_inner_dim, size(A)))
else
    0
end
linear_size(::IndexStyle, _, _) = 0

function mapreducedim_naive(f::F, op::OP, A, init, is_inner_dim, inner, outer, sink) where {F, OP}
    lsiz = linear_size(A, is_inner_dim)
    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, axes(outer))

    if lsiz > 0
        i0 = first(LinearIndices(A))
        linear_range = i0:(i0+lsiz-1)
        for (idx, i) in enumerate(outer)
            v = mapreduce_pairwise(f, op, A, init, linear_range .+ (lsiz*(idx-1)))
            mapreduce_set!(sink, R, i, v)
        end
    else
        for i in outer
            v = mapreduce_pairwise(f, op, A, init, mergeindices(is_inner_dim, inner, i))
            mapreduce_set!(sink, R, i, v)
        end
    end
    return mapreduce_finish(sink, R)
end
"""
    mapreducedim_colmajor(f, op, A, init, is_inner_dim, inner, outer, sink) -> R

Column-major reduction kernel for cases where the inner dimension is included
but not fully contiguous. Processes contiguous parts efficiently.
"""
mapreducedim_colmajor(f::F, op::OP, A, init, is_inner_dim, inner, outer, sink, enforce_pairwise=true) where {F,OP} =
    mapreducedim_colmajor(f, op, A, init, is_inner_dim, inner, outer, sink, _mkplan(f, op, A, is_inner_dim), enforce_pairwise)

function mapreducedim_colmajor(f::F, op::OP, A, init, is_inner_dim, inner, outer, sink,
                               plan::ReductionPlan{N}, enforce_pairwise=true) where {F,OP,N}
    # Split the inner (reduced) block into its contiguous prefix and the discontiguous "remainder"
    is_contiguous_inner = keep_first_trues(is_inner_dim)
    contiguous_inner    = mergeindices(is_contiguous_inner, inner, first(inner))  # varying only over the contiguous prefix
    discontiguous_inner = mergeindices(is_contiguous_inner, first(inner), inner)  # varying only over the non-contiguous remainder

    # Decide early: if the remainder fan-out is large, one whole-inner sweep is faster.
    # contig_len is how many indices the contiguous pack spans (e.g. size(A,1) for k=1).
    k = plan.inner_pack
    contig_len = k == 0 ? 0 :
                 (k == 1 ? length(contiguous_inner) :
                           length(mergeindices(is_contiguous_inner, contiguous_inner, first(outer))))
    if enforce_pairwise && _use_pairwise_reduction(plan, contig_len, length(discontiguous_inner))
        return mapreducedim_naive(f, op, A, init, is_inner_dim, inner, outer, sink)
    end

    # Seed the result with the first remainder slice (so we always combine into a real value)
    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, axes(outer))
    for o in outer
        v = mapreduce_pairwise(f, op, A, init, mergeindices(is_inner_dim, contiguous_inner, o))
        mapreduce_set!(sink, R, o, v)
    end

    # No remainder → done
    rem_total = length(discontiguous_inner)
    rem_total <= 1 && return R

    # If op is commutative we can process remainder "lanes" in small tiles, which greatly
    # reduces loop overhead and improves cache behavior on cases like dims=(1,2,4).
    if is_commutative_op(op)
        B = plan.tile_B

        # Advance the remainder iterator past the first element (already folded in R)
        it = iterate(discontiguous_inner)
        @assert it !== nothing
        _, st = it
        it = iterate(discontiguous_inner, st)  # now at the second element (or nothing)

        # Setup for direct kernel calls
        contig_inds = (k <= 1 ? axes(A,1) : CartesianIndices(ntuple(d->axes(A,d), k)))
        tile_buf = Vector{CartesianIndex{N}}(undef, B)

        while true
            # Gather up to B remainder indices into the tile buffer
            tlen = 0
            while tlen < B
                it === nothing && break
                dci, st = it
                tlen += 1
                tile_buf[tlen] = dci
                it = iterate(discontiguous_inner, st)
            end
            tlen == 0 && break  # no more work

            # For each outer position, fold the B-lane tile, then combine once into R[o].
            # This avoids touching R[o] tlen times and lets the compiler keep a single
            # accumulator live in registers per outer element.
            for o in outer
                # Start with first element to avoid neutral element issues
                dci1 = @inbounds tile_buf[1]
                vtile = _lane_comm_sweep(f, op, A, init, contig_inds, o, dci1,
                                        is_inner_dim, Val(k))
                for j in 2:tlen
                    dci = @inbounds tile_buf[j]
                    vtile = op(vtile, _lane_comm_sweep(f, op, A, init, contig_inds, o, dci,
                                                       is_inner_dim, Val(k)))
                end
                @inbounds R[o] = op(R[o], vtile)
            end

            # If the last gather reached the end, we are done.
            it === nothing && break
        end
        return R
    end

    # Non-commutative path: strict left-to-right across remainder lanes.
    # Keep the straightforward order to preserve semantics; we still benefit
    # from avoiding inner branching and from the single seed above.
    for dci in Iterators.drop(discontiguous_inner, 1)
        for o in outer
            i = mergeindices(is_inner_dim, dci, o)
            val = mapreduce_pairwise(f, op, A, init, mergeindices(is_contiguous_inner, contiguous_inner, i))
            @inbounds R[o] = op(R[o], val)
        end
    end
    return R
end

"""
    mapreducedim_rowmajor(f, op, A, init, is_inner_dim, inner, outer, sink) -> R

Row-major reduction kernel for fully discontiguous cases.
Uses @inbounds but avoids @simd for non-vectorizable scalar recurrences.
"""
mapreducedim_rowmajor(f::F, op::OP, A, init, is_inner_dim, inner, outer, sink, enforce_pairwise=true) where {F,OP} =
    mapreducedim_rowmajor(f, op, A, init, is_inner_dim, inner, outer, sink, _mkplan(f, op, A, is_inner_dim), enforce_pairwise)

function mapreducedim_rowmajor(f::F, op::OP, A, init, is_inner_dim, inner, outer, sink,
                               plan::ReductionPlan{N}, enforce_pairwise=true) where {F,OP,N}
    # Row-major is essentially a left-to-right fold over a discontiguous set of dims.
    # Stronger guard: if the inner set is long, prefer the naive whole-inner sweep.
    if enforce_pairwise && _use_pairwise_reduction(plan, 1, length(inner))
        return mapreducedim_naive(f, op, A, init, is_inner_dim, inner, outer, sink)
    end

    # Initialize with the very first inner index so that R[o] is always a real value
    it0 = iterate(inner)
    @assert it0 !== nothing
    i1, st = it0
    v0 = _mapreduce_start(f, op, A, init)
    R = mapreduce_allocate(sink, v0, axes(outer))
    for o in outer
        v = _mapreduce_start(f, op, A, init, A[mergeindices(is_inner_dim, i1, o)])
        mapreduce_set!(sink, R, o, v)
    end

    # If op is commutative we can block the remaining inner indices into small tiles,
    # accumulating per-outer partials in registers before touching R[o] once per tile.
    if is_commutative_op(op)
        B = plan.tile_B
        tile_buf = Vector{CartesianIndex{N}}(undef, B)

        it = iterate(inner, st)  # start from the second inner index
        while true
            # Gather up to B inner indices
            tlen = 0
            while tlen < B
                it === nothing && break
                i, st = it
                tlen += 1
                tile_buf[tlen] = i
                it = iterate(inner, st)
            end
            tlen == 0 && break  # exhausted

            # Fold the tile per outer coordinate, then combine once
            for o in outer
                # Start with first element to avoid neutral element issues
                tb1 = @inbounds tile_buf[1]
                idx = mergeindices(is_inner_dim, tb1, o)
                a_val = @inbounds A[idx]
                vtile = f(a_val)
                # Accumulate the rest
                for j in 2:tlen
                    tbj = @inbounds tile_buf[j]
                    idx = mergeindices(is_inner_dim, tbj, o)
                    a_val = @inbounds A[idx]
                    vtile = op(vtile, f(a_val))
                end
                @inbounds R[o] = op(R[o], vtile)
            end

            it === nothing && break
        end
        return R
    end

    # Non-commutative path: strict left-to-right recurrence (preserve order).
    # Note: @simd on scalar recurrence doesn't vectorize, but can help scheduling.
    for i in Iterators.drop(inner, 1)
        for o in outer
            iA = mergeindices(is_inner_dim, i, o)
            a_val = @inbounds A[iA]
            val = f(a_val)
            @inbounds R[o] = op(R[o], val)
        end
    end
    return R
end

# Streaming kernel for "keep dim 1" reductions using sink API
# Works correctly with OffsetArrays and custom axes
function _mapreducedim_stream_firstdim(f::F, op::OP, A, init, dims, sink) where {F,OP}
    nd = ndims(A)
    @assert !(1 in dims) && nd > 1

    axsA = ntuple(d->axes(A,d), nd)

    if isempty(A)
        v = _mapreduce_start(f, op, A, init)
        out_axes = ntuple(d -> d==1 ? axsA[1] : reduced_index(axsA[d]), nd)
        R = mapreduce_allocate(sink, v, out_axes)
        for I in CartesianIndices(axes(R))
            @inbounds _sink_set!(sink, R, I, v)
        end
        return mapreduce_finish(sink, R)
    end

    # Seed output with the first "column"
    v_seed = _mapreduce_start(f, op, A, init, first(A))
    out_axes = ntuple(d -> d==1 ? axsA[1] : reduced_index(axsA[d]), nd)
    R = mapreduce_allocate(sink, v_seed, out_axes)
    out_axes_R = axes(R)

    # Columns are all reduced dims varying, others fixed to their first index
    cols_axes = ntuple(d -> d==1 ? reduced_index(axsA[1]) :
                           (d in dims ? axsA[d] : reduced_index(axsA[d])), nd)
    cols = CartesianIndices(cols_axes)

    # ---- Seed pass with the first column c0 (no double op!) -------------------
    c0 = first(cols)
    if nd == ndims(R)
        # Build a 1-D view along dim-1 once; other dims fixed at their first in R
        # (Tuple is created once, not per element)
        fixed = ntuple(d -> d==1 ? Colon() : first(out_axes_R[d]), nd)
        Rrow = @view R[fixed...]
        # Also build a 1-D view of A's first column
        colAidx = ntuple(d -> d==1 ? Colon() : (d in dims ? c0[d] : first(axsA[d])), nd)
        Acol = @view A[colAidx...]
        for i in eachindex(Rrow, Acol)
            # Only the array accesses are @inbounds, not f or op
            ai = @inbounds Acol[i]
            val = _mapreduce_start(f, op, A, init, ai)
            @inbounds Rrow[i] = val
        end
    else
        # Fallback: R has fewer dims. Use LinearIndices once.
        LR = LinearIndices(R)
        colAidx = ntuple(d -> d==1 ? Colon() : (d in dims ? c0[d] : first(axsA[d])), nd)
        Acol = @view A[colAidx...]
        # map linear index i -> LR[i]
        for i in eachindex(Acol)
            I = @inbounds LR[i]
            ai = @inbounds Acol[i]
            val = _mapreduce_start(f, op, A, init, ai)
            @inbounds _sink_set!(sink, R, I, val)
        end
    end

    # ---- Accumulate the rest of the columns -----------------------------------
    if nd == ndims(R)
        fixedR = ntuple(d -> d==1 ? Colon() : first(out_axes_R[d]), nd)
        Rrow = @view R[fixedR...]
        for c in Iterators.drop(cols, 1)
            colAidx = ntuple(d -> d==1 ? Colon() : (d in dims ? c[d] : first(axsA[d])), nd)
            Acol = @view A[colAidx...]
            for i in eachindex(Rrow, Acol)
                # Only array accesses are @inbounds, not user functions
                ai = @inbounds Acol[i]
                ri = @inbounds Rrow[i]
                val = op(ri, f(ai))
                @inbounds Rrow[i] = val
            end
        end
    else
        LR = LinearIndices(R)
        for c in Iterators.drop(cols, 1)
            colAidx = ntuple(d -> d==1 ? Colon() : (d in dims ? c[d] : first(axsA[d])), nd)
            Acol = @view A[colAidx...]
            for i in eachindex(Acol)
                I = @inbounds LR[i]
                ai = @inbounds Acol[i]
                val = f(ai)
                @inbounds _sink_accum!(sink, R, I, val)
            end
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
                tmpAv = f(@inbounds(A[i,IA]))
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
                tmpAv = f(@inbounds(A[i,IA]))
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
