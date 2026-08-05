# This file is a part of Julia. License is MIT: https://julialang.org/license

mutable struct IntrusiveLinkedList{T}
    # Invasive list requires that T have a field `.next >: U{T, Nothing}` and `.queue::Any`
    head::Union{T, Nothing}
    tail::Union{T, Nothing}
    IntrusiveLinkedList{T}() where {T} = new{T}(nothing, nothing)
end

struct ILLRef{T}
    list::IntrusiveLinkedList{T}
    waitee::Any # Invariant: waitqueue(waitee).list === list
end

# waitqueue(x) returns the ILLRef for the queue of waiters registered on `x`.
# Methods are added for each waitee type (conditions, tasks, workqueues, ...).
function waitqueue end

eltype(::Type{<:IntrusiveLinkedList{T}}) where {T} = @isdefined(T) ? T : Any

iterate(q::IntrusiveLinkedList) = (h = q.head; h === nothing ? nothing : (h, h))
iterate(q::IntrusiveLinkedList{T}, v::T) where {T} = (h = v.next; h === nothing ? nothing : (h, h))

isempty(q::IntrusiveLinkedList) = (q.head === nothing)

function length(q::IntrusiveLinkedList)
    i = 0
    head = q.head
    while head !== nothing
        i += 1
        head = head.next
    end
    return i
end

isempty(qr::ILLRef) = isempty(qr.list)
length(qr::ILLRef) = length(qr.list)

function push!(qr::ILLRef{T}, val::T) where T
    val.queue === nothing || error("val already in a list")
    val.queue = qr.waitee
    q = qr.list
    tail = q.tail
    if tail === nothing
        q.head = q.tail = val
    else
        tail.next = val
        q.tail = val
    end
    return q
end

function pushfirst!(qr::ILLRef{T}, val::T) where T
    val.queue === nothing || error("val already in a list")
    val.queue = qr.waitee
    q = qr.list
    head = q.head
    if head === nothing
        q.head = q.tail = val
    else
        val.next = head
        q.head = val
    end
    return q
end

function pop!(qr::ILLRef{T}) where {T}
    val = qr.list.tail::T
    _list_deletefirst!(qr.list, val) # expensive!
    return val
end

function popfirst!(qr::ILLRef{T}) where {T}
    val = qr.list.head::T
    _list_deletefirst!(qr.list, val) # cheap
    return val
end

# Delete `val` from the list, but only if it is actually in it, as witnessed by
# `val.queue` holding the ILLRef's waitee. This makes deletion a no-op if `val`
# was concurrently popped, which various cleanup paths rely upon.
function list_deletefirst!(qr::ILLRef{T}, val::T) where T
    val.queue === qr.waitee || return qr.list
    return _list_deletefirst!(qr.list, val)
end

push!(q::IntrusiveLinkedList{T}, val::T) where T = push!(ILLRef(q, q), val)
pushfirst!(q::IntrusiveLinkedList{T}, val::T) where T = pushfirst!(ILLRef(q, q), val)
pop!(q::IntrusiveLinkedList{T}) where T = pop!(ILLRef(q, q))
popfirst!(q::IntrusiveLinkedList{T}) where T = popfirst!(ILLRef(q, q))
list_deletefirst!(q::IntrusiveLinkedList{T}, val::T) where T = list_deletefirst!(ILLRef(q, q), val)

# this function assumes `val` is found in `q`
function _list_deletefirst!(q::IntrusiveLinkedList{T}, val::T) where T
    head = q.head::T
    if head === val
        if q.tail::T === val
            q.head = q.tail = nothing
        else
            q.head = val.next::T
        end
    else
        head_next = head.next::T
        while head_next !== val
            head = head_next
            head_next = head.next::T
        end
        if q.tail::T === val
            head.next = nothing
            q.tail = head
        else
            head.next = val.next::T
        end
    end
    val.next = nothing
    val.queue = nothing
    return q
end
