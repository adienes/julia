# This file is a part of Julia. License is MIT: https://julialang.org/license

"""
    StringView{T <: AbstractVector{UInt8}} <: AbstractString

An `AbstractString` representation of any `vector` of `UInt8` data,
interpreted as UTF-8 encoded Unicode.
Similar to `String`, the underlying data may be invalid UTF-8.

`StringView(v::AbstractVector{UInt8})::StringView` does not make a copy of
or modify the `v`. Use `codeunits` to get `v` from the `StringView`.
After construction, `v` may be mutated, which will be reflected in
the resulting `StringView`.

!!! compat "Julia 1.14"
    The `StringView` type requires at least Julia 1.14.

# Examples
```jldoctest
julia> arr = [0x61, 0xf0, 0x63, 0x64];

julia> s = StringView(arr)
"a\\xf0cd"

julia> codeunits(s) === arr
true

julia> arr[2] = Int('b'); s
"abcd"
```
"""
struct StringView{T <: AbstractVector{UInt8}} <: AbstractString
    data::T

    function StringView{T}(data::T) where {T <: AbstractVector{UInt8}}
        # For now, StringViews code assumes one-based indexing
        require_one_based_indexing(data)

        # Prevent someone constructing e.g. a `StringView{AbstractVector{UInt8}}`,
        # the existence of which will complicate the implementation and provide
        # no usability benefit.
        if !isconcretetype(T)
            throw(ArgumentError("StringView must be parameterized with a concrete type"))
        end

        new{T}(data)
    end
end


"""
    StringIndexError(str, i)

An error occurred when trying to access `str` at index `i` that is not valid.
"""
struct StringIndexError <: Exception
    string::AbstractString
    index::Int
end
@noinline string_index_err((@nospecialize s::AbstractString), i::Integer) =
    throw(StringIndexError(s, Int(i)))
function showerror(io::IO, exc::StringIndexError)
    s = exc.string
    print(io, "StringIndexError: ", "invalid index [$(exc.index)]")
    if firstindex(s) <= exc.index <= ncodeunits(s)
        iprev = thisind(s, exc.index)
        inext = nextind(s, iprev)
        escprev = escape_string(s[iprev:iprev])
        if inext <= ncodeunits(s)
            escnext = escape_string(s[inext:inext])
            print(io, ", valid nearby indices [$iprev]=>'$escprev', [$inext]=>'$escnext'")
        else
            print(io, ", valid nearby index [$iprev]=>'$escprev'")
        end
    end
end

@inline between(b::T, lo::T, hi::T) where {T<:Integer} = (lo ≤ b) & (b ≤ hi)

"""
    String <: AbstractString

The default string type in Julia, used by e.g. string literals.

`String`s are immutable sequences of `Char`s. A `String` is stored internally as
a contiguous byte array, and while they are interpreted as being UTF-8 encoded,
they can be composed of any byte sequence. Use [`isvalid`](@ref) to validate
that the underlying byte sequence is valid as UTF-8.
"""
String

## constructors and conversions ##

# String constructor docstring from boot.jl, workaround for #16730
# and the unavailability of @doc in boot.jl context.
"""
    String(v::AbstractVector{UInt8})

Create a new `String` object using the data buffer from byte vector `v`.
If `v` is a `Vector{UInt8}` it will be truncated to zero length and future
modification of `v` cannot affect the contents of the resulting string.
To avoid truncation of `Vector{UInt8}` data, use `String(copy(v))`; for other
`AbstractVector` types, `String(v)` already makes a copy.

When possible, the memory of `v` will be used without copying when the `String`
object is created. This is guaranteed to be the case for byte vectors returned
by [`take!`](@ref) on a writable [`IOBuffer`](@ref) and by calls to
[`read(io, nb)`](@ref). This allows zero-copy conversion of I/O data to strings.
In other cases, `Vector{UInt8}` data may be copied, but `v` is truncated anyway
to guarantee consistent behavior.
"""
String(v::AbstractVector{UInt8}) = unsafe_takestring(copyto!(StringMemory(length(v)), v))

function String(v::Vector{UInt8})
    len = length(v)
    len == 0 && return ""
    ref = v.ref
    if ref.ptr_or_offset == ref.mem.ptr
        str = ccall(:jl_genericmemory_to_string, Ref{String}, (Any, Int), ref.mem, len)
    else
        str = ccall(:jl_pchar_to_string, Ref{String}, (Ptr{UInt8}, Int), ref, len)
    end
    # optimized empty!(v); sizehint!(v, 0) calls
    setfield!(v, :size, (0,))
    setfield!(v, :ref, memoryref(Memory{UInt8}()))
    return str
end

"""
    unsafe_takestring(m::Memory{UInt8})::String

Create a `String` from `m`, changing the interpretation of the contents of `m`.
This is done without copying, if possible. Thus, any access to `m` after
calling this function, either to read or to write, is undefined behavior.
"""
function unsafe_takestring(m::Memory{UInt8})
    isempty(m) ? "" : ccall(:jl_genericmemory_to_string, Ref{String}, (Any, Int), m, length(m))
end

"""
    takestring!(x)::AbstractString

Create a string from the content of `x`, emptying `x`.

# Examples
```jldoctest
julia> v = [0x61, 0x62, 0x63];

julia> s = takestring!(v)
"abc"

julia> isempty(v)
true
```

!!! compat "Julia 1.13"
    This function requires at least Julia 1.13.
"""
takestring!(v::Vector{UInt8}) = String(v)

"""
    unsafe_string(p::Ptr{UInt8}, [length::Integer])
    unsafe_string(p::Cstring)

Copy a string from the address of a C-style (NUL-terminated) string encoded as UTF-8.
(The pointer can be safely freed afterwards.) If `length` is specified
(the length of the data in bytes), the string does not have to be NUL-terminated.

This function is labeled "unsafe" because it will crash if `p` is not
a valid memory address to data of the requested length.
"""
function unsafe_string(p::Union{Ptr{UInt8},Ptr{Int8}}, len::Integer)
    p == C_NULL && throw(ArgumentError("cannot convert NULL to string"))
    ccall(:jl_pchar_to_string, Ref{String}, (Ptr{UInt8}, Int), p, len)
end
function unsafe_string(p::Union{Ptr{UInt8},Ptr{Int8}})
    p == C_NULL && throw(ArgumentError("cannot convert NULL to string"))
    ccall(:jl_cstr_to_string, Ref{String}, (Ptr{UInt8},), p)
end

# This is `@assume_effects :total !:consistent @ccall jl_alloc_string(n::Csize_t)::Ref{String}`,
# but the macro is not available at this time in bootstrap, so we write it manually.
const _string_n_override = 0x04ee
@eval _string_n(n::Integer) = $(Expr(:foreigncall, QuoteNode(:jl_alloc_string), Ref{String},
    :(Core.svec(Csize_t)), 1, QuoteNode((:ccall, _string_n_override, false)), :(convert(Csize_t, n))))

"""
    String(s::AbstractString)

Create a new `String` from an existing `AbstractString`.
"""
String(s::AbstractString) = print_to_string(s)
@assume_effects :total String(s::Symbol) = unsafe_string(unsafe_convert(Ptr{UInt8}, s))

unsafe_wrap(::Type{Memory{UInt8}}, s::String) = ccall(:jl_string_to_genericmemory, Ref{Memory{UInt8}}, (Any,), s)
unsafe_wrap(::Type{Vector{UInt8}}, s::String) = wrap(Array, unsafe_wrap(Memory{UInt8}, s))

Vector{UInt8}(s::CodeUnits{UInt8,String}) = copyto!(Vector{UInt8}(undef, length(s)), s)
Vector{UInt8}(s::String) = Vector{UInt8}(codeunits(s))
Array{UInt8}(s::String)  = Vector{UInt8}(codeunits(s))

String(s::CodeUnits{UInt8,String}) = s.s

## low-level functions ##

pointer(s::String) = unsafe_convert(Ptr{UInt8}, s)
pointer(s::String, i::Integer) = pointer(s) + Int(i)::Int - 1

ncodeunits(s::String) = Core.sizeof(s)
codeunit(s::String) = UInt8

codeunit(s::String, i::Integer) = codeunit(s, Int(i)::Int)
@assume_effects :foldable @inline function codeunit(s::String, i::Int)
    @boundscheck checkbounds(s, i)
    b = GC.@preserve s unsafe_load(pointer(s, i))
    return b
end

## comparison ##

@assume_effects :total _memcmp(a::String, b::String) = @invoke _memcmp(a::Union{Ptr{UInt8},AbstractString},b::Union{Ptr{UInt8},AbstractString})

_memcmp(a::Union{Ptr{UInt8},AbstractString}, b::Union{Ptr{UInt8},AbstractString}) = _memcmp(a, b, min(sizeof(a), sizeof(b)))
function _memcmp(a::Union{Ptr{UInt8},AbstractString}, b::Union{Ptr{UInt8},AbstractString}, len::Int)
    GC.@preserve a b begin
        pa = unsafe_convert(Ptr{UInt8}, a)
        pb = unsafe_convert(Ptr{UInt8}, b)
        memcmp(pa, pb, len % Csize_t) % Int
    end
end

function cmp(a::String, b::String)
    al, bl = sizeof(a), sizeof(b)
    c = _memcmp(a, b)
    return c < 0 ? -1 : c > 0 ? +1 : cmp(al,bl)
end

==(a::String, b::String) = a===b

typemin(::Type{String}) = ""
typemin(::String) = typemin(String)

## thisind, nextind ##

@propagate_inbounds thisind(s::String, i::Int) = _thisind_str(s, i)

# nothrow: i == ncodeunits(s) always satisfies the bounds check inside _thisind_str
# (it short-circuits when i == 0, otherwise 1 ≤ i ≤ n).
@assume_effects :nothrow lastindex(s::String) = thisind(s, ncodeunits(s)::Int)

# s should be String, StringView, or SubString{String}
@inline function _thisind_str(s, i::Int)
    i == 0 && return 0
    n = ncodeunits(s)
    i == n + 1 && return i
    @boundscheck between(i, 1, n) || throw(BoundsError(s, i))
    @inbounds b = codeunit(s, i)
    (b & 0xc0 == 0x80) & (i-1 > 0) || return i
    (@noinline function _thisind_continued(s, i, n) # mark the rest of the function as a slow-path
        local b
        @inbounds b = codeunit(s, i-1)
        between(b, 0b11000000, 0b11110111) && return i-1
        (b & 0xc0 == 0x80) & (i-2 > 0) || return i
        @inbounds b = codeunit(s, i-2)
        between(b, 0b11100000, 0b11110111) && return i-2
        (b & 0xc0 == 0x80) & (i-3 > 0) || return i
        @inbounds b = codeunit(s, i-3)
        between(b, 0b11110000, 0b11110111) && return i-3
        return i
    end)(s, i, n)
end

@propagate_inbounds nextind(s::String, i::Int) = _nextind_str(s, i)

# s should be String or SubString{String}
@inline function _nextind_str(s, i::Int)
    i == 0 && return 1
    n = ncodeunits(s)
    @boundscheck between(i, 1, n) || throw(BoundsError(s, i))
    @inbounds l = codeunit(s, i)
    between(l, 0x80, 0xf7) || return i+1
    (@noinline function _nextind_continued(s, i, n, l) # mark the rest of the function as a slow-path
        if l < 0xc0
            # handle invalid codeunit index by scanning back to the start of this index
            # (which may be the same as this index)
            i′ = @inbounds thisind(s, i)
            i′ >= i && return i+1
            i = i′
            @inbounds l = codeunit(s, i)
            (l < 0x80) | (0xf8 ≤ l) && return i+1
            @assert l >= 0xc0 "invalid codeunit"
        end
        # first continuation byte
        (i += 1) > n && return i
        @inbounds b = codeunit(s, i)
        b & 0xc0 ≠ 0x80 && return i
        ((i += 1) > n) | (l < 0xe0) && return i
        # second continuation byte
        @inbounds b = codeunit(s, i)
        b & 0xc0 ≠ 0x80 && return i
        ((i += 1) > n) | (l < 0xf0) && return i
        # third continuation byte
        @inbounds b = codeunit(s, i)
        return ifelse(b & 0xc0 ≠ 0x80, i, i+1)
    end)(s, i, n, l)
end

## checking UTF-8 & ASCII validity ##
#=
    The UTF-8 Validation is performed by a shift based DFA.
    ┌───────────────────────────────────────────────────────────────────┐
    │    UTF-8 DFA State Diagram    ┌──────────────2──────────────┐     │
    │                               ├────────3────────┐           │     │
    │                 ┌──────────┐  │     ┌─┐        ┌▼┐          │     │
    │      ASCII      │  UTF-8   │  ├─5──►│9├───1────► │          │     │
    │                 │          │  │     ├─┤        │ │         ┌▼┐    │
    │                 │  ┌─0─┐   │  ├─6──►│8├─1,7,9──►4├──1,7,9──► │    │
    │      ┌─0─┐      │  │   │   │  │     ├─┤        │ │         │ │    │
    │      │   │      │ ┌▼───┴┐  │  ├─11─►│7├──7,9───► │ ┌───────►3├─┐  │
    │     ┌▼───┴┐     │ │     │  ▼  │     └─┘        └─┘ │       │ │ │  │
    │     │  0  ├─────┘ │  1  ├─► ──┤                    │  ┌────► │ │  │
    │     └─────┘       │     │     │     ┌─┐            │  │    └─┘ │  │
    │                   └──▲──┘     ├─10─►│5├─────7──────┘  │        │  │
    │                      │        │     ├─┤               │        │  │
    │                      │        └─4──►│6├─────1,9───────┘        │  │
    │          INVALID     │              └─┘                        │  │
    │           ┌─*─┐      └──────────────────1,7,9──────────────────┘  │
    │          ┌▼───┴┐                                                  │
    │          │  2  ◄─── All undefined transitions result in state 2   │
    │          └─────┘                                                  │
    └───────────────────────────────────────────────────────────────────┘

        Validation States
            0 -> _UTF8_DFA_ASCII is the start state and will only stay in this state if the string is only ASCII characters
                        If the DFA ends in this state the string is ASCII only
            1 -> _UTF8_DFA_ACCEPT is the valid complete character state of the DFA once it has encountered a UTF-8 Unicode character
            2 -> _UTF8_DFA_INVALID is only reached by invalid bytes and once in this state it will not change
                    as seen by all 1s in that column of table below
            3 -> One valid continuation byte needed to return to state 0
        4,5,6 -> Two valid continuation bytes needed to return to state 0
        7,8,9 -> Three valid continuation bytes needed to return to state 0

                        Current State
                    0̲  1̲  2̲  3̲  4̲  5̲  6̲  7̲  8̲  9̲
                0 | 0  1  2  2  2  2  2  2  2  2
                1 | 2  2  2  1  3  2  3  2  4  4
                2 | 3  3  2  2  2  2  2  2  2  2
                3 | 4  4  2  2  2  2  2  2  2  2
                4 | 6  6  2  2  2  2  2  2  2  2
    Character   5 | 9  9  2  2  2  2  2  2  2  2     <- Next State
    Class       6 | 8  8  2  2  2  2  2  2  2  2
                7 | 2  2  2  1  3  3  2  4  4  2
                8 | 2  2  2  2  2  2  2  2  2  2
                9 | 2  2  2  1  3  2  3  4  4  2
               10 | 5  5  2  2  2  2  2  2  2  2
               11 | 7  7  2  2  2  2  2  2  2  2

           Shifts | 0  4 10 14 18 24  8 20 12 26

    The shifts that represent each state were derived using the SMT solver Z3, to ensure when encoded into
    the rows the correct shift was a result.

    Each character class row is encoding 10 states with shifts as defined above. By shifting the bits of a row by
    the current state then masking the result with 0x11110 give the shift for the new state


=#

#State type used by UTF-8 DFA
const _UTF8DFAState = UInt32
# Fill the table with 256 UInt64 representing the DFA transitions for all bytes
const _UTF8_DFA_TABLE = let # let block rather than function doesn't pollute base
    num_classes=12
    num_states=10
    bit_per_state = 6

    # These shifts were derived using a SMT solver
    state_shifts = [0, 4, 10, 14, 18, 24, 8, 20, 12, 26]

    character_classes = [   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                            8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                            10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3,
                            11, 6, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8 ]

    # These are the rows discussed in comments above
    state_arrays = [ 0  1  2  2  2  2  2  2  2  2;
                     2  2  2  1  3  2  3  2  4  4;
                     3  3  2  2  2  2  2  2  2  2;
                     4  4  2  2  2  2  2  2  2  2;
                     6  6  2  2  2  2  2  2  2  2;
                     9  9  2  2  2  2  2  2  2  2;
                     8  8  2  2  2  2  2  2  2  2;
                     2  2  2  1  3  3  2  4  4  2;
                     2  2  2  2  2  2  2  2  2  2;
                     2  2  2  1  3  2  3  4  4  2;
                     5  5  2  2  2  2  2  2  2  2;
                     7  7  2  2  2  2  2  2  2  2]

    #This converts the state_arrays into the shift encoded _UTF8DFAState
    class_row = zeros(_UTF8DFAState, num_classes)

    for i = 1:num_classes
        row = _UTF8DFAState(0)
        for j in 1:num_states
            #Calculate the shift required for the next state
            to_shift = UInt8((state_shifts[state_arrays[i,j]+1]) )
            #Shift the next state into the position of the current state
            row = row | (_UTF8DFAState(to_shift) << state_shifts[j])
        end
        class_row[i]=row
    end

    tuple(map(c->class_row[c+1],character_classes)...)
end


const _UTF8_DFA_ASCII = _UTF8DFAState(0) #This state represents the start and end of any valid string
const _UTF8_DFA_ACCEPT = _UTF8DFAState(4) #This state represents the start and end of any valid string
const _UTF8_DFA_INVALID = _UTF8DFAState(10) # If the state machine is ever in this state just stop

# The dfa step is broken out so that it may be used in other functions. The mask was calculated to work with state shifts above
# noub: `byte + 1` is always within the 256-entry table for a UInt8 argument
@assume_effects :noub @inline _utf_dfa_step(state::_UTF8DFAState, byte::UInt8) = @inbounds (_UTF8_DFA_TABLE[byte+1] >> state) & _UTF8DFAState(0x0000001E)

@inline function _isvalid_utf8_dfa(state::_UTF8DFAState, bytes::AbstractVector{UInt8}, first::Int = firstindex(bytes), last::Int = lastindex(bytes))
    @assume_effects :terminates_locally for i = first:last
       @inbounds state = _utf_dfa_step(state, bytes[i])
    end
    return (state)
end

# `_find_first_nonascii` for long inputs scanned from the start:
# `_ASCII_CHUNK_SIZE`-wide probes, with the all-clean case finished by a single
# overlapping wide scan; the narrow cascade only runs inside a known-dirty block.
@inline function _find_first_nonascii_wide(cu::AbstractVector{UInt8}, i::Int, n::Int)
    chunk = _ASCII_CHUNK_SIZE
    # terminates_locally: each iteration advances `i` by `chunk`; the guards
    # leave room for the advance so the arithmetic cannot wrap
    @inbounds if n - i >= chunk
        @assume_effects :terminates_locally while true
            _isascii(cu, i, i + chunk - 1) || break
            i += chunk
            if n - i < chunk
                _isascii(cu, n - chunk + 1, n) && return 0
                break
            end
        end
    end
    return _find_first_nonascii(cu, i, n)
end

##

# Classifications of string
    # 0: neither valid ASCII nor UTF-8
    # 1: valid ASCII
    # 2: valid UTF-8
 byte_string_classify(s::AbstractString) = byte_string_classify(codeunits(s))


function byte_string_classify(bytes::AbstractVector{UInt8})
    n = length(bytes)
    if n < _ASCII_CHUNK_SIZE + _ASCII_CHUNK_SIZE ÷ 2
        _isascii(bytes, 1, n) && return 1
        start = 1
    else
        start = _find_first_nonascii_wide(bytes, 1, n)
        start == 0 && return 1
    end
    return _byte_string_classify_nonascii(bytes, start, n)
end

function _byte_string_classify_nonascii(bytes::AbstractVector{UInt8}, first::Int, last::Int)
    chunk_size = 256

    start = first
    stop = start + min(chunk_size - 1, last - start)
    state = _UTF8_DFA_ACCEPT
    # terminates_locally: both loops advance `start` by the constant
    # `chunk_size`; subtraction guards keep the arithmetic from wrapping
    @assume_effects :terminates_locally while true
        # try to process ascii chunks
        while state == _UTF8_DFA_ACCEPT
            _isascii(bytes,start,stop) || break
            last - start < chunk_size && return ifelse(state == _UTF8_DFA_ACCEPT,2,0)
            start += chunk_size
            stop = start + min(chunk_size - 1, last - start)
        end
        # Process non ascii chunk
        state = _isvalid_utf8_dfa(state,bytes,start,stop)
        state == _UTF8_DFA_INVALID && return 0

        last - start < chunk_size && break
        start += chunk_size
        stop = start + min(chunk_size - 1, last - start)
    end
    return ifelse(state == _UTF8_DFA_ACCEPT,2,0)
end

isvalid(::Type{String}, bytes::AbstractVector{UInt8}) = (@inline byte_string_classify(bytes)) ≠ 0
isvalid(::Type{String}, s::AbstractString) =  (@inline byte_string_classify(s)) ≠ 0

@inline isvalid(s::AbstractString) = @inline isvalid(String, codeunits(s))

# nothrow: every index the classification chain reads is derived from the
# codeunits' length, hence in bounds for a String (not so for arbitrary vectors)
isvalid(s::String) = @assume_effects :nothrow isvalid(String, codeunits(s))

is_valid_continuation(c) = c & 0xc0 == 0x80

## required core functionality ##

@inline function iterate(s::Union{String, StringView}, i::Int=firstindex(s))
    (i % UInt) - 1 < ncodeunits(s) || return nothing
    b = @inbounds codeunit(s, i)
    u = UInt32(b) << 24
    between(b, 0x80, 0xf7) || return reinterpret(Char, u), i+1
    return @noinline iterate_continued(s, i, u)
end

# duck-type s so that external UTF-8 string packages like StringViews can hook in
function iterate_continued(s, i::Int, u::UInt32)
    @label begin
        u < 0xc0000000 && (i += 1; break)
        n = ncodeunits(s)
        # first continuation byte
        (i += 1) > n && break
        @inbounds b = codeunit(s, i)
        b & 0xc0 == 0x80 || break
        u |= UInt32(b) << 16
        # second continuation byte
        ((i += 1) > n) | (u < 0xe0000000) && break
        @inbounds b = codeunit(s, i)
        b & 0xc0 == 0x80 || break
        u |= UInt32(b) << 8
        # third continuation byte
        ((i += 1) > n) | (u < 0xf0000000) && break
        @inbounds b = codeunit(s, i)
        b & 0xc0 == 0x80 || break
        u |= UInt32(b); i += 1
    end
    return reinterpret(Char, u), i
end

@propagate_inbounds function getindex(s::Union{String, StringView}, i::Int)
    b = codeunit(s, i)
    u = UInt32(b) << 24
    between(b, 0x80, 0xf7) || return reinterpret(Char, u)
    return getindex_continued(s, i, u)
end

# duck-type s so that external UTF-8 string packages like StringViews can hook in
function getindex_continued(s, i::Int, u::UInt32)
    @label begin
        if u < 0xc0000000
            # called from `getindex` which checks bounds
            @inbounds isvalid(s, i) && break
            string_index_err(s, i)
        end
        n = ncodeunits(s)

        (i += 1) > n && break
        @inbounds b = codeunit(s, i) # cont byte 1
        b & 0xc0 == 0x80 || break
        u |= UInt32(b) << 16

        ((i += 1) > n) | (u < 0xe0000000) && break
        @inbounds b = codeunit(s, i) # cont byte 2
        b & 0xc0 == 0x80 || break
        u |= UInt32(b) << 8

        ((i += 1) > n) | (u < 0xf0000000) && break
        @inbounds b = codeunit(s, i) # cont byte 3
        b & 0xc0 == 0x80 || break
        u |= UInt32(b)
    end
    return reinterpret(Char, u)
end

function getindex(s::Union{String, StringView}, r::AbstractUnitRange{<:Integer})
    span = (Int(first(r))::Int):(Int(last(r)))::Int
    return s[span]
end

@inline function getindex(s::String, r::UnitRange{Int})
    isempty(r) && return ""
    i, j = first(r), last(r)
    @boundscheck begin
        checkbounds(s, r)
        @inbounds isvalid(s, i) || string_index_err(s, i)
        @inbounds isvalid(s, j) || string_index_err(s, j)
    end
    # Safety: The boundscheck checked r is inbounds in s,
    # and since we also checked r is not empty, j must be inbounds in s
    j = @inbounds nextind(s, j) - 1
    n = (j - i + 1) % UInt
    ss = _string_n(n)
    GC.@preserve s ss unsafe_copyto!(pointer(ss), pointer(s, i), n)
    return ss
end

# nothrow because we know the start and end indices are valid
@assume_effects :nothrow function length(s::String)
    return length_continued(s, 1, ncodeunits(s), ncodeunits(s))
end

function length(s::StringView)
    return length_continued(s, 1, ncodeunits(s), ncodeunits(s))
end

# effects needed because @inbounds
@assume_effects :consistent :effect_free @inline function length(s::String, i::Int, j::Int)
    _length(s, i, j)
end

@inline function length(s::StringView, i::Int, j::Int)
    _length(s, i, j)
end

@inline function _length(s::Union{String, StringView}, i::Int, j::Int)
    @boundscheck begin
        0 < i ≤ ncodeunits(s)+1 || throw(BoundsError(s, i))
        0 ≤ j < ncodeunits(s)+1 || throw(BoundsError(s, j))
    end
    j < i && return 0
    @inbounds i, k = thisind(s, i), i
    c = j - i + (i == k)
    @inbounds length_continued(s, i, j, c)
end

@assume_effects :terminates_globally @propagate_inbounds function length_continued(s::String, i::Int, n::Int, c::Int)
    _length_continued(s, i, n, c)
end

@propagate_inbounds function length_continued(s::StringView, i::Int, n::Int, c::Int)
    _length_continued(s, i, n, c)
end

# bytes counted bytewise around non-ASCII text before retrying the ASCII scan;
# larger amortizes the hand-off cost, smaller resumes ASCII skipping sooner
const _STRING_LENGTH_SCAN_WINDOW = 128

# Index of the first byte >= 0x80 in cu[i:n], or 0 if all are ASCII.
# Guards must use subtraction and leave room for the advance so that index
# arithmetic cannot wrap even for a StringView over huge virtual data.
@inline function _find_first_nonascii(cu::AbstractVector{UInt8}, i::Int, n::Int)
    # terminates_locally: each loop advances `i` by a positive constant
    @inbounds begin
        @assume_effects :terminates_locally while n - i >= 256
            _isascii(cu, i, i + 255) || break
            i += 256
        end
        @assume_effects :terminates_locally while n - i >= 32
            _isascii(cu, i, i + 31) || break
            i += 32
        end
        @assume_effects :terminates_locally for k in i:n
            cu[k] >= 0x80 && return k
        end
    end
    return 0
end

@propagate_inbounds function _length_continued(s::Union{String, StringView}, i::Int, n::Int, c::Int)
    i < n || return c
    if s isa StringView && n == typemax(Int)
        # only virtual StringView data can be this large; count the final byte
        # separately so the hot loops below may assume `n + 1` cannot wrap
        # (the `isa` keeps the `thisind` call out of the String specialization,
        # preserving its effects)
        c = _length_continued(s, i, n - 1, c)
        return c - (thisind(s, n) < n)
    end
    cu = codeunits(s)
    n - i < 32 && return first(_length_bytewise(cu, i, n, n, c, false))
    @inbounds while true
        if cu[i] < 0x80 # avoid paying for a failed scan on non-ASCII-dense text
            j = _find_first_nonascii(cu, i, n)
            # a trailing non-ASCII byte at n cannot extend a lead byte
            (j == 0) | (j >= n) && return c
            i = j
        end
        c, i = _length_bytewise(cu, i, i + min(_STRING_LENGTH_SCAN_WINDOW - 1, n - i), n, c)
        i < n || return c
    end
end

# Count the characters starting in `cu[i:stop]`: decrement `c` once per
# continuation byte extending a lead byte. The lead-byte scan ends at `stop`,
# but a character being consumed is always finished (never reading past `n`).
# Returns `(c, i′)` with `i′ > stop` the first unconsumed index. Requires
# `n < typemax(Int)` so the increments cannot wrap.
# `windowed = false` drops the `stop` bound and its per-byte check.
@inline function _length_bytewise(cu::AbstractVector{UInt8}, i::Int, stop::Int, n::Int, c::Int,
                                  windowed::Bool = true)
    @inbounds b = cu[i]
    @inbounds while true
        while true
            (i += 1) ≤ n || return c, i
            0xc0 ≤ b ≤ 0xf7 && break
            windowed && (i ≤ stop || return c, i)
            b = cu[i]
        end
        l = b
        b = cu[i] # cont byte 1
        c -= (x = b & 0xc0 == 0x80)
        x & (l ≥ 0xe0) || continue

        (i += 1) ≤ n || return c, i
        b = cu[i] # cont byte 2
        c -= (x = b & 0xc0 == 0x80)
        x & (l ≥ 0xf0) || continue

        (i += 1) ≤ n || return c, i
        b = cu[i] # cont byte 3
        c -= (b & 0xc0 == 0x80)
    end
end

## overload methods for efficiency ##

isvalid(s::String, i::Int) = checkbounds(Bool, s, i) && thisind(s, i) == i

# `isascii(::AbstractVector)` reduces to `@inbounds codeunit(::String, ::Int)`, total.
isascii(s::String) = @assume_effects :nothrow :foldable isascii(codeunits(s))

# don't assume effects for general integers since we cannot know their implementation
@assume_effects :foldable repeat(c::Char, r::BitInteger) = @invoke repeat(c::Char, r::Integer)

"""
    repeat(c::AbstractChar, r::Integer)::String

Repeat a character `r` times. This can equivalently be accomplished by calling
[`c^r`](@ref :^(::Union{AbstractString, AbstractChar}, ::Integer)).

# Examples
```jldoctest
julia> repeat('A', 3)
"AAA"
```
"""
function repeat(c::AbstractChar, r::Integer)
    r < 0 && throw(ArgumentError("can't repeat a character $r times"))
    r = UInt(r)::UInt
    c = Char(c)::Char
    r == 0 && return ""
    u = bswap(reinterpret(UInt32, c))
    n = 4 - (leading_zeros(u | 0xff) >> 3)
    s = _string_n(n*r)
    p = pointer(s)
    GC.@preserve s if n == 1
        memset(p, u % UInt8, r)
    elseif n == 2
        p16 = reinterpret(Ptr{UInt16}, p)
        for i = 1:r
            unsafe_store!(p16, u % UInt16, i)
        end
    elseif n == 3
        b1 = (u >> 0) % UInt8
        b2 = (u >> 8) % UInt8
        b3 = (u >> 16) % UInt8
        for i = 0:r-1
            unsafe_store!(p, b1, 3i + 1)
            unsafe_store!(p, b2, 3i + 2)
            unsafe_store!(p, b3, 3i + 3)
        end
    elseif n == 4
        p32 = reinterpret(Ptr{UInt32}, p)
        for i = 1:r
            unsafe_store!(p32, u, i)
        end
    end
    return s
end
