# This file is a part of Julia. License is MIT: https://julialang.org/license

module GMP

export BigInt

import .Base: *, +, -, /, <, <<, >>, >>>, <=, ==, >, >=, ^, (~), (&), (|), xor, nand, nor,
             binomial, cmp, convert, div, divrem, factorial, cld, fld, gcd, gcdx, lcm, mod,
             ndigits, promote_rule, rem, show, isqrt, string, powermod, sum, prod,
             trailing_zeros, trailing_ones, count_ones, count_zeros, tryparse_internal,
             bin, oct, dec, hex, isequal, invmod, _prevpow2, _nextpow2, ndigits0zpb,
             widen, signed, unsafe_trunc, trunc, iszero, isone, big, flipsign, signbit,
             sign, isodd, iseven, digits!, hash, hash_integer, top_set_bit,
             ispositive, isnegative, clamp, unsafe_takestring

import Core: Signed, Float16, Float32, Float64

if Clong == Int32
    const ClongMax = Union{Int8, Int16, Int32}
    const CulongMax = Union{UInt8, UInt16, UInt32}
else
    const ClongMax = Union{Int8, Int16, Int32, Int64}
    const CulongMax = Union{UInt8, UInt16, UInt32, UInt64}
end
const CdoubleMax = Union{Float16, Float32, Float64}

if Sys.iswindows()
    const libgmp = "libgmp-10.dll"
elseif Sys.isapple()
    const libgmp = "@rpath/libgmp.10.dylib"
else
    const libgmp = "libgmp.so.10"
end

_version() = unsafe_string(unsafe_load(cglobal((:__gmp_version, libgmp), Ptr{Cchar})))
version() = VersionNumber(_version())
major_version() = _version()[1]
bits_per_limb() = Int(unsafe_load(cglobal((:__gmp_bits_per_limb, libgmp), Cint)))

const VERSION = version()
const MAJOR_VERSION = major_version()
const BITS_PER_LIMB = bits_per_limb()

# GMP's mp_limb_t is by default a typedef of `unsigned long`, but can also be configured to be either
# `unsigned int` or `unsigned long long int`. The correct unsigned type is here named Limb, and must
# be used whenever mp_limb_t is in the signature of ccall'ed GMP functions.
if BITS_PER_LIMB == 32
    const Limb = UInt32
    const SLimbMax = Union{Int8, Int16, Int32}
    const ULimbMax = Union{UInt8, UInt16, UInt32}
elseif BITS_PER_LIMB == 64
    const Limb = UInt64
    const SLimbMax = Union{Int8, Int16, Int32, Int64}
    const ULimbMax = Union{UInt8, UInt16, UInt32, UInt64}
else
    error("GMP: cannot determine the type mp_limb_t (__gmp_bits_per_limb == $BITS_PER_LIMB)")
end

"""
    BigInt <: Signed

Arbitrary precision integer type.
"""
mutable struct BigInt <: Signed
    alloc::Cint
    size::Cint
    d::Ptr{Limb}

    function BigInt(; nbits::Integer=0)
        b = MPZ.init2!(new(), nbits)
        finalizer(cglobal((:__gmpz_clear, libgmp)), b)
        return b
    end
end

"""
    BigInt(x)

Create an arbitrary precision integer. `x` may be an `Int` (or anything that can be
converted to an `Int`). The usual mathematical operators are defined for this type, and
results are promoted to a [`BigInt`](@ref).

Instances can be constructed from strings via [`parse`](@ref), or using the `big`
string literal.

# Examples
```jldoctest
julia> parse(BigInt, "42")
42

julia> big"313"
313

julia> BigInt(10)^19
10000000000000000000
```
"""
BigInt(x)

"""
    ALLOC_OVERFLOW_FUNCTION

A reference that holds a boolean, if true, indicating julia is linked with a patched GMP that
does not abort on huge allocation and throws OutOfMemoryError instead.
"""
const ALLOC_OVERFLOW_FUNCTION = Ref(false)

function __init__()
    try
        if major_version() != MAJOR_VERSION || bits_per_limb() != BITS_PER_LIMB
            msg = """The dynamically loaded GMP library (v\"$(version())\" with __gmp_bits_per_limb == $(bits_per_limb()))
                     does not correspond to the compile time version (v\"$VERSION\" with __gmp_bits_per_limb == $BITS_PER_LIMB).
                     Please rebuild Julia."""
            bits_per_limb() != BITS_PER_LIMB ? @error(msg) : @warn(msg)
        end

        ccall((:__gmp_set_memory_functions, libgmp), Cvoid,
              (Ptr{Cvoid},Ptr{Cvoid},Ptr{Cvoid}),
              cglobal(:jl_gc_counted_malloc),
              cglobal(:jl_gc_counted_realloc_with_old_size),
              cglobal(:jl_gc_counted_free_with_size))
        ZERO.alloc, ZERO.size, ZERO.d = 0, 0, C_NULL
        ONE.alloc, ONE.size, ONE.d = 1, 1, pointer(_ONE)
    catch ex
        Base.showerror_nostdio(ex, "WARNING: Error during initialization of module GMP")
    end
    # This only works with a patched version of GMP, ignore otherwise
    try
        ccall((:__gmp_set_alloc_overflow_function, libgmp), Cvoid,
              (Ptr{Cvoid},),
              cglobal(:jl_throw_out_of_memory_error))
        ALLOC_OVERFLOW_FUNCTION[] = true
    catch ex
        # ErrorException("ccall: could not find function...")
        if typeof(ex) != ErrorException
            rethrow()
        end
    end
end


module MPZ
# wrapping of libgmp functions
# - "output parameters" are labeled x, y, z, and are returned when appropriate
# - constant input parameters are labeled a, b, c
# - a method modifying its input has a "!" appended to its name, according to Julia's conventions
# - some convenient methods are added (in addition to the pure MPZ ones), e.g. `add(a, b) = add!(BigInt(), a, b)`
#   and `add!(x, a) = add!(x, x, a)`.
using ..GMP: BigInt, Limb, BITS_PER_LIMB, libgmp

const mpz_t = Ref{BigInt}
const bitcnt_t = Culong

gmpz(op::Symbol) = (Symbol(:__gmpz_, op), libgmp)

init!(x::BigInt) = (ccall((:__gmpz_init, libgmp), Cvoid, (mpz_t,), x); x)
init2!(x::BigInt, a) = (ccall((:__gmpz_init2, libgmp), Cvoid, (mpz_t, bitcnt_t), x, a); x)

realloc2!(x, a) = (ccall((:__gmpz_realloc2, libgmp), Cvoid, (mpz_t, bitcnt_t), x, a); x)
realloc2(a) = realloc2!(BigInt(), a)

sizeinbase(a::BigInt, b) = Int(ccall((:__gmpz_sizeinbase, libgmp), Csize_t, (mpz_t, Cint), a, b))

for (op, nbits) in (:add => :(BITS_PER_LIMB*(1 + max(abs(a.size), abs(b.size)))),
                    :sub => :(BITS_PER_LIMB*(1 + max(abs(a.size), abs(b.size)))),
                    :mul => 0, :fdiv_q => 0, :tdiv_q => 0, :cdiv_q => 0,
                    :fdiv_r => 0, :tdiv_r => 0, :cdiv_r => 0,
                    :gcd => 0, :lcm => 0, :and => 0, :ior => 0, :xor => 0)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt, b::BigInt) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t, mpz_t), x, a, b); x)
        $op(a::BigInt, b::BigInt) = $op!(BigInt(nbits=$nbits), a, b)
        $op!(x::BigInt, b::BigInt) = $op!(x, x, b)
    end
end

invert!(x::BigInt, a::BigInt, b::BigInt) =
    ccall((:__gmpz_invert, libgmp), Cint, (mpz_t, mpz_t, mpz_t), x, a, b)
invert!(x::BigInt, b::BigInt) = invert!(x, x, b)
invert(a::BigInt, b::BigInt) = (ret=BigInt(); invert!(ret, a, b); ret)

for op in (:add_ui, :sub_ui, :mul_ui, :mul_2exp, :fdiv_q_2exp, :pow_ui, :bin_ui)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt, b) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t, Culong), x, a, b); x)
        $op(a::BigInt, b) = $op!(BigInt(), a, b)
        $op!(x::BigInt, b) = $op!(x, x, b)
    end
end

ui_sub!(x::BigInt, a, b::BigInt) = (ccall((:__gmpz_ui_sub, libgmp), Cvoid, (mpz_t, Culong, mpz_t), x, a, b); x)
ui_sub(a, b::BigInt) = ui_sub!(BigInt(), a, b)

for op in (:scan1, :scan0)
    # when there is no meaningful answer, ccall returns typemax(Culong), where Culong can
    # be UInt32 (Windows) or UInt64; we return -1 in this case for all architectures
    @eval $op(a::BigInt, b) = Int(signed(ccall($(gmpz(op)), Culong, (mpz_t, Culong), a, b)))
end

mul_si!(x::BigInt, a::BigInt, b) = (ccall((:__gmpz_mul_si, libgmp), Cvoid, (mpz_t, mpz_t, Clong), x, a, b); x)
mul_si(a::BigInt, b) = mul_si!(BigInt(), a, b)
mul_si!(x::BigInt, b) = mul_si!(x, x, b)

for op in (:neg, :com, :sqrt, :set)
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a::BigInt) = (ccall($(gmpz(op)), Cvoid, (mpz_t, mpz_t), x, a); x)
        $op(a::BigInt) = $op!(BigInt(), a)
    end
    op === :set && continue # MPZ.set!(x) would make no sense
    @eval $op!(x::BigInt) = $op!(x, x)
end

for (op, T) in ((:fac_ui, Culong), (:set_ui, Culong), (:set_si, Clong), (:set_d, Cdouble))
    op! = Symbol(op, :!)
    @eval begin
        $op!(x::BigInt, a) = (ccall($(gmpz(op)), Cvoid, (mpz_t, $T), x, a); x)
        $op(a) = $op!(BigInt(), a)
    end
end

popcount(a::BigInt) = Int(signed(ccall((:__gmpz_popcount, libgmp), Culong, (mpz_t,), a)))

mpn_popcount(d::Ptr{Limb}, s::Integer) = Int(ccall((:__gmpn_popcount, libgmp), Culong, (Ptr{Limb}, Csize_t), d, s))
mpn_popcount(a::BigInt) = mpn_popcount(a.d, abs(a.size))

function tdiv_qr!(x::BigInt, y::BigInt, a::BigInt, b::BigInt)
    ccall((:__gmpz_tdiv_qr, libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t), x, y, a, b)
    x, y
end
tdiv_qr(a::BigInt, b::BigInt) = tdiv_qr!(BigInt(), BigInt(), a, b)

powm!(x::BigInt, a::BigInt, b::BigInt, c::BigInt) =
    (ccall((:__gmpz_powm, libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t), x, a, b, c); x)
powm(a::BigInt, b::BigInt, c::BigInt) = powm!(BigInt(), a, b, c)
powm!(x::BigInt, b::BigInt, c::BigInt) = powm!(x, x, b, c)

function gcdext!(x::BigInt, y::BigInt, z::BigInt, a::BigInt, b::BigInt)
    ccall((:__gmpz_gcdext, libgmp), Cvoid, (mpz_t, mpz_t, mpz_t, mpz_t, mpz_t), x, y, z, a, b)
    x, y, z
end
gcdext(a::BigInt, b::BigInt) = gcdext!(BigInt(), BigInt(), BigInt(), a, b)

cmp(a::BigInt, b::BigInt) = Int(ccall((:__gmpz_cmp, libgmp), Cint, (mpz_t, mpz_t), a, b))
cmp_si(a::BigInt, b) = Int(ccall((:__gmpz_cmp_si, libgmp), Cint, (mpz_t, Clong), a, b))
cmp_ui(a::BigInt, b) = Int(ccall((:__gmpz_cmp_ui, libgmp), Cint, (mpz_t, Culong), a, b))
cmp_d(a::BigInt, b) = Int(ccall((:__gmpz_cmp_d, libgmp), Cint, (mpz_t, Cdouble), a, b))

mpn_cmp(a::Ptr{Limb}, b::Ptr{Limb}, c) = ccall((:__gmpn_cmp, libgmp), Cint, (Ptr{Limb}, Ptr{Limb}, Clong), a, b, c)
mpn_cmp(a::BigInt, b::BigInt, c) = mpn_cmp(a.d, b.d, c)

get_str!(x, a, b::BigInt) = (ccall((:__gmpz_get_str,libgmp), Ptr{Cchar}, (Ptr{Cchar}, Cint, mpz_t), x, a, b); x)
set_str!(x::BigInt, a, b) = Int(ccall((:__gmpz_set_str, libgmp), Cint, (mpz_t, Ptr{UInt8}, Cint), x, a, b))
get_d(a::BigInt) = ccall((:__gmpz_get_d, libgmp), Cdouble, (mpz_t,), a)

function export!(a::AbstractVector{T}, n::BigInt; order::Integer=-1, nails::Integer=0, endian::Integer=0) where {T<:Base.BitInteger}
    stride(a, 1) == 1 || throw(ArgumentError("a must have stride 1"))
    ndigits = cld(sizeinbase(n, 2), 8*sizeof(T) - nails)
    length(a) < ndigits && resize!(a, ndigits)
    fill!(a, zero(T))
    count = Ref{Csize_t}()
    ccall((:__gmpz_export, libgmp), Ptr{T}, (Ptr{T}, Ref{Csize_t}, Cint, Csize_t, Cint, Csize_t, mpz_t),
        a, count, order, sizeof(T), endian, nails, n)
    @assert count[] ≤ length(a)
    return a, Int(count[])
end

limbs_write!(x::BigInt, a) = ccall((:__gmpz_limbs_write, libgmp), Ptr{Limb}, (mpz_t, Clong), x, a)
limbs_finish!(x::BigInt, a) = ccall((:__gmpz_limbs_finish, libgmp), Cvoid, (mpz_t, Clong), x, a)

setbit!(x, a) = (ccall((:__gmpz_setbit, libgmp), Cvoid, (mpz_t, bitcnt_t), x, a); x)
tstbit(a::BigInt, b) = ccall((:__gmpz_tstbit, libgmp), Cint, (mpz_t, bitcnt_t), a, b) % Bool

end # module MPZ

const ZERO = BigInt()
const ONE  = BigInt()
const _ONE = Limb[1]

widen(::Type{Int128})  = BigInt
widen(::Type{UInt128}) = BigInt
widen(::Type{BigInt})  = BigInt

signed(x::BigInt) = x

BigInt(x::BigInt) = x
Signed(x::BigInt) = x

function tryparse_internal(::Type{BigInt}, s::AbstractString, startpos::Int, endpos::Int, base_::Integer, raise::Bool)
    # don't make a copy in the common case where we are parsing a whole String
    bstr = startpos == firstindex(s) && endpos == lastindex(s) ? String(s) : String(SubString(s,startpos,endpos))

    sgn, base, i = Base.parseint_preamble(true,Int(base_),bstr,firstindex(bstr),lastindex(bstr))
    if !(2 <= base <= 62)
        raise && throw(ArgumentError("invalid base: base must be 2 ≤ base ≤ 62, got $base"))
        return nothing
    end
    if i == 0
        raise && throw(ArgumentError("premature end of integer: $(repr(bstr))"))
        return nothing
    end
    z = BigInt()
    if Base.containsnul(bstr)
        err = -1 # embedded NUL char (not handled correctly by GMP)
    else
        err = GC.@preserve bstr MPZ.set_str!(z, pointer(bstr)+(i-firstindex(bstr)), base)
    end
    if err != 0
        raise && throw(ArgumentError("invalid BigInt: $(repr(bstr))"))
        return nothing
    end
    flipsign!(z, sgn)
end

BigInt(x::Union{Clong,Int32}) = MPZ.set_si(x)
BigInt(x::Union{Culong,UInt32}) = MPZ.set_ui(x)
BigInt(x::Bool) = BigInt(UInt(x))

unsafe_trunc(::Type{BigInt}, x::Union{Float16,Float32,Float64}) = MPZ.set_d(x)

function BigInt(x::Float64)
    isinteger(x) || throw(InexactError(:BigInt, BigInt, x))
    unsafe_trunc(BigInt,x)
end

BigInt(x::Float16) = BigInt(Float64(x))
BigInt(x::Float32) = BigInt(Float64(x))

function BigInt(x::Integer)
    # On 64-bit Windows, `Clong` is `Int32`, not `Int64`, so construction of
    # `Int64` constants, e.g. `BigInt(3)`, uses this method.
    isbits(x) && typemin(Clong) <= x <= typemax(Clong) && return BigInt((x % Clong)::Clong)
    nd = ndigits(x, base=2)
    z = MPZ.realloc2(nd)
    ux = unsigned(x < 0 ? -x : x)
    size = 0
    limbnbits = sizeof(Limb) << 3
    while nd > 0
        size += 1
        unsafe_store!(z.d, ux % Limb, size)
        ux >>= limbnbits
        nd -= limbnbits
    end
    z.size = x < 0 ? -size : size
    z
end


rem(x::BigInt, ::Type{Bool}) = !iszero(x) & unsafe_load(x.d) % Bool # never unsafe here

rem(x::BigInt, ::Type{T}) where T<:Union{SLimbMax,ULimbMax} =
    iszero(x) ? zero(T) : flipsign(unsafe_load(x.d) % T, x.size)

function rem(x::BigInt, ::Type{T}) where T<:Union{Base.BitUnsigned,Base.BitSigned}
    u = zero(T)
    for l = 1:min(abs(x.size), cld(sizeof(T), sizeof(Limb)))
        u += (unsafe_load(x.d, l) % T) << ((sizeof(Limb)<<3)*(l-1))
    end
    flipsign(u, x.size)
end

rem(x::Integer, ::Type{BigInt}) = BigInt(x)

clamp(x, ::Type{BigInt}) = convert(BigInt, x)

isodd(x::BigInt) = MPZ.tstbit(x, 0)
iseven(x::BigInt) = !isodd(x)

function (::Type{T})(x::BigInt) where T<:Base.BitUnsigned
    if sizeof(T) < sizeof(Limb)
        convert(T, convert(Limb,x))
    else
        0 <= x.size <= cld(sizeof(T),sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        x % T
    end
end

function (::Type{T})(x::BigInt) where T<:Base.BitSigned
    n = abs(x.size)
    if sizeof(T) < sizeof(Limb)
        SLimb = typeof(Signed(one(Limb)))
        convert(T, convert(SLimb, x))
    else
        0 <= n <= cld(sizeof(T),sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        y = x % T
        ispositive(x) ⊻ (y > 0) && throw(InexactError(nameof(T), T, x)) # catch overflow
        y
    end
end


Float64(n::BigInt, ::RoundingMode{:ToZero}) = MPZ.get_d(n)

function (::Type{T})(n::BigInt, ::RoundingMode{:ToZero}) where T<:Union{Float16,Float32}
    T(Float64(n,RoundToZero),RoundToZero)
end

function (::Type{T})(n::BigInt, ::RoundingMode{:Down}) where T<:CdoubleMax
    x = T(n,RoundToZero)
    x > n ? prevfloat(x) : x
end
function (::Type{T})(n::BigInt, ::RoundingMode{:Up}) where T<:CdoubleMax
    x = T(n,RoundToZero)
    x < n ? nextfloat(x) : x
end

function Float64(x::BigInt, ::RoundingMode{:Nearest})
    x == 0 && return 0.0
    xsize = abs(x.size)
    if xsize*BITS_PER_LIMB > 1024
        z = Inf64
    elseif xsize == 1
        z = Float64(unsafe_load(x.d))
    elseif Limb == UInt32 && xsize == 2
        z = Float64((unsafe_load(x.d, 2) % UInt64) << BITS_PER_LIMB + unsafe_load(x.d))
    else
        y1 = unsafe_load(x.d, xsize) % UInt64
        n = top_set_bit(y1)
        # load first 54(1 + 52 bits of fraction + 1 for rounding)
        y = y1 >> (n - (precision(Float64)+1))
        if Limb == UInt64
            y += n > precision(Float64) ? 0 : (unsafe_load(x.d, xsize-1) >> (10+n))
        else
            y += (unsafe_load(x.d, xsize-1) % UInt64) >> (n-22)
            y += n > (precision(Float64) - 32) ? 0 : (unsafe_load(x.d, xsize-2) >> (10+n))
        end
        y = (y + 1) >> 1 # round, ties up
        y &= ~UInt64(trailing_zeros(x) == (n-54 + (xsize-1)*BITS_PER_LIMB)) # fix last bit to round to even
        d = ((n+1021) % UInt64) << 52
        z = reinterpret(Float64, d+y)
        z = ldexp(z, (xsize-1)*BITS_PER_LIMB)
    end
    return flipsign(z, x.size)
end

function Float32(x::BigInt, ::RoundingMode{:Nearest})
    x == 0 && return 0f0
    xsize = abs(x.size)
    if xsize*BITS_PER_LIMB > 128
        z = Inf32
    elseif xsize == 1
        z = Float32(unsafe_load(x.d))
    else
        y1 = unsafe_load(x.d, xsize)
        n = BITS_PER_LIMB - leading_zeros(y1)
        # load first 25(1 + 23 bits of fraction + 1 for rounding)
        y = (y1 >> (n - (precision(Float32)+1))) % UInt32
        y += (n > precision(Float32) ? 0 : unsafe_load(x.d, xsize-1) >> (BITS_PER_LIMB - (25-n))) % UInt32
        y = (y + one(UInt32)) >> 1 # round, ties up
        y &= ~UInt32(trailing_zeros(x) == (n-25 + (xsize-1)*BITS_PER_LIMB)) # fix last bit to round to even
        d = ((n+125) % UInt32) << 23
        z = reinterpret(Float32, d+y)
        z = ldexp(z, (xsize-1)*BITS_PER_LIMB)
    end
    return flipsign(z, x.size)
end

function Float16(x::BigInt, ::RoundingMode{:Nearest})
    x == 0 && return Float16(0.0)
    y1 = unsafe_load(x.d)
    n = BITS_PER_LIMB - leading_zeros(y1)
    if n > 16 || abs(x.size) > 1
        z = Inf16
    else
        # load first 12(1 + 10 bits for fraction + 1 for rounding)
        y = (y1 >> (n - (precision(Float16)+1))) % UInt16
        y = (y + one(UInt16)) >> 1 # round, ties up
        y &= ~UInt16(trailing_zeros(x) == (n-12)) # fix last bit to round to even
        d = ((n+13) % UInt16) << 10
        z = reinterpret(Float16, d+y)
    end
    return flipsign(z, x.size)
end

Float64(n::BigInt) = Float64(n, RoundNearest)
Float32(n::BigInt) = Float32(n, RoundNearest)
Float16(n::BigInt) = Float16(n, RoundNearest)

promote_rule(::Type{BigInt}, ::Type{<:Integer}) = BigInt

"""
    big(x)

Convert a number to a maximum precision representation (typically [`BigInt`](@ref) or
`BigFloat`). See [`BigFloat`](@ref BigFloat(::Any, rounding::RoundingMode)) for
information about some pitfalls with floating-point numbers.
"""
function big end

big(::Type{<:Integer})  = BigInt
big(::Type{<:Rational}) = Rational{BigInt}

big(n::Integer) = convert(BigInt, n)

# Binary ops
for (fJ, fC) in ((:+, :add), (:-,:sub), (:*, :mul),
                 (:mod, :fdiv_r), (:rem, :tdiv_r),
                 (:gcd, :gcd), (:lcm, :lcm),
                 (:&, :and), (:|, :ior), (:xor, :xor))
    @eval begin
        ($fJ)(x::BigInt, y::BigInt) = MPZ.$fC(x, y)
    end
end

for (r, f) in ((RoundToZero, :tdiv_q),
               (RoundDown, :fdiv_q),
               (RoundUp, :cdiv_q))
    @eval div(x::BigInt, y::BigInt, ::typeof($r)) = MPZ.$f(x, y)
end

# For compat only. Remove in 2.0.
div(x::BigInt, y::BigInt) = div(x, y, RoundToZero)
fld(x::BigInt, y::BigInt) = div(x, y, RoundDown)
cld(x::BigInt, y::BigInt) = div(x, y, RoundUp)

/(x::BigInt, y::BigInt) = float(x)/float(y)

function invmod(x::BigInt, y::BigInt)
    z = zero(BigInt)
    ya = abs(y)
    if ya == 1
        return z
    end
    if (y==0 || MPZ.invert!(z, x, ya) == 0)
        throw(DomainError(y))
    end
    # GMP always returns a positive inverse; we instead want to
    # normalize such that div(z, y) == 0, i.e. we want a negative z
    # when y is negative.
    if y < 0
        MPZ.add!(z, y)
    end
    # The postcondition is: mod(z * x, y) == mod(big(1), m) && div(z, y) == 0
    return z
end

# More efficient commutative operations
for (fJ, fC) in ((:+, :add), (:*, :mul), (:&, :and), (:|, :ior), (:xor, :xor))
    fC! = Symbol(fC, :!)
    @eval begin
        ($fJ)(a::BigInt, b::BigInt, c::BigInt) = MPZ.$fC!(MPZ.$fC(a, b), c)
        ($fJ)(a::BigInt, b::BigInt, c::BigInt, d::BigInt) = MPZ.$fC!(MPZ.$fC!(MPZ.$fC(a, b), c), d)
        ($fJ)(a::BigInt, b::BigInt, c::BigInt, d::BigInt, e::BigInt) =
            MPZ.$fC!(MPZ.$fC!(MPZ.$fC!(MPZ.$fC(a, b), c), d), e)
    end
end

# Basic arithmetic without promotion
+(x::BigInt, c::CulongMax) = MPZ.add_ui(x, c)
+(c::CulongMax, x::BigInt) = x + c

-(x::BigInt, c::CulongMax) = MPZ.sub_ui(x, c)
-(c::CulongMax, x::BigInt) = MPZ.ui_sub(c, x)

+(x::BigInt, c::ClongMax) = c < 0 ? -(x, -(c % Culong)) : x + convert(Culong, c)
+(c::ClongMax, x::BigInt) = c < 0 ? -(x, -(c % Culong)) : x + convert(Culong, c)
-(x::BigInt, c::ClongMax) = c < 0 ? +(x, -(c % Culong)) : -(x, convert(Culong, c))
-(c::ClongMax, x::BigInt) = c < 0 ? -(x + -(c % Culong)) : -(convert(Culong, c), x)

*(x::BigInt, c::CulongMax) = MPZ.mul_ui(x, c)
*(c::CulongMax, x::BigInt) = x * c
*(x::BigInt, c::ClongMax) = MPZ.mul_si(x, c)
*(c::ClongMax, x::BigInt) = x * c

/(x::BigInt, y::Union{ClongMax,CulongMax}) = float(x)/y
/(x::Union{ClongMax,CulongMax}, y::BigInt) = x/float(y)

# unary ops
(-)(x::BigInt) = MPZ.neg(x)
(~)(x::BigInt) = MPZ.com(x)

<<(x::BigInt, c::UInt) = c == 0 ? x : MPZ.mul_2exp(x, c)
>>(x::BigInt, c::UInt) = c == 0 ? x : MPZ.fdiv_q_2exp(x, c)
>>>(x::BigInt, c::UInt) = x >> c

function trailing_zeros(x::BigInt)
    c = MPZ.scan1(x, 0)
    c == -1 && throw(DomainError(x, "`x` must be non-zero"))
    c
end

function trailing_ones(x::BigInt)
    c = MPZ.scan0(x, 0)
    c == -1 && throw(DomainError(x, "`x` must not be equal to -1"))
    c
end

function count_ones(x::BigInt)
    c = MPZ.popcount(x)
    c == -1 && throw(DomainError(x, "`x` cannot be negative"))
    c
end

# generic definition is not used to provide a better error message
function count_zeros(x::BigInt)
    c = MPZ.popcount(~x)
    c == -1 && throw(DomainError(x, "`x` must be negative"))
    c
end

"""
    count_ones_abs(x::BigInt)

Number of ones in the binary representation of abs(x).
"""
count_ones_abs(x::BigInt) = iszero(x) ? 0 : MPZ.mpn_popcount(x)

function top_set_bit(x::BigInt)
    isnegative(x) && throw(DomainError(x, "top_set_bit only supports negative arguments when they have type BitSigned."))
    iszero(x) && return 0
    x.size * sizeof(Limb) << 3 - leading_zeros(GC.@preserve x unsafe_load(x.d, x.size))
end

divrem(x::BigInt, y::BigInt,  ::typeof(RoundToZero) = RoundToZero) = MPZ.tdiv_qr(x, y)
divrem(x::BigInt, y::Integer, ::typeof(RoundToZero) = RoundToZero) = MPZ.tdiv_qr(x, BigInt(y))

cmp(x::BigInt, y::BigInt) = sign(MPZ.cmp(x, y))
cmp(x::BigInt, y::ClongMax) = sign(MPZ.cmp_si(x, y))
cmp(x::BigInt, y::CulongMax) = sign(MPZ.cmp_ui(x, y))
cmp(x::BigInt, y::Integer) = cmp(x, big(y))
cmp(x::Integer, y::BigInt) = -cmp(y, x)

cmp(x::BigInt, y::CdoubleMax) = isnan(y) ? -1 : sign(MPZ.cmp_d(x, y))
cmp(x::CdoubleMax, y::BigInt) = -cmp(y, x)

isqrt(x::BigInt) = MPZ.sqrt(x)

^(x::BigInt, y::Culong) = MPZ.pow_ui(x, y)

function bigint_pow(x::BigInt, y::Integer)
    x == 1 && return x
    x == -1 && return isodd(y) ? x : -x
    if y<0; throw(DomainError(y, "`y` cannot be negative.")); end
    @noinline throw1(y) =
        throw(OverflowError("exponent $y is too large and computation will overflow"))
    if y>typemax(Culong)
       x==0 && return x

       #At this point, x is not 1, 0 or -1 and it is not possible to use
       #gmpz_pow_ui to compute the answer. Note that the magnitude of the
       #answer is:
       #- at least 2^(2^32-1) ≈ 10^(1.3e9) (if Culong === UInt32).
       #- at least 2^(2^64-1) ≈ 10^(5.5e18) (if Culong === UInt64).
       #
       #Assume that the answer will definitely overflow.

       throw1(y)
    end
    return x^convert(Culong, y)
end

^(x::BigInt , y::BigInt ) = bigint_pow(x, y)
^(x::BigInt , y::Bool   ) = y ? x : one(x)
^(x::BigInt , y::Integer) = bigint_pow(x, y)
^(x::Integer, y::BigInt ) = bigint_pow(BigInt(x), y)
^(x::Bool   , y::BigInt ) = Base.power_by_squaring(x, y)

function powermod(x::BigInt, p::BigInt, m::BigInt)
    r = MPZ.powm(x, p, m)
    return m < 0 && r > 0 ? MPZ.add!(r, m) : r # choose sign consistent with mod(x^p, m)
end

powermod(x::Integer, p::Integer, m::BigInt) = powermod(big(x), big(p), m)

function gcdx(a::BigInt, b::BigInt)
    g, s, t = MPZ.gcdext(a, b)
    if t == 0
        # work around a difference in some versions of GMP
        if a == b
            return g, t, s
        elseif abs(a)==abs(b)
            return g, t, -s
        end
    end
    g, s, t
end

+(x::BigInt, y::BigInt, rest::BigInt...) = sum(tuple(x, y, rest...))
sum(arr::Union{AbstractArray{BigInt}, Tuple{BigInt, Vararg{BigInt}}}) =
    foldl(MPZ.add!, arr; init=BigInt(0))

function prod(arr::AbstractArray{BigInt})
    # compute first the needed number of bits for the result,
    # to avoid re-allocations;
    # GMP will always request n+m limbs for the result in MPZ.mul!,
    # if the arguments have n and m limbs; so we add all the bits
    # taken by the array elements, and add BITS_PER_LIMB to that,
    # to account for the rounding to limbs in MPZ.mul!
    # (BITS_PER_LIMB-1 would typically be enough, to which we add
    # 1 for the initial multiplication by init=1 in foldl)
    nbits = BITS_PER_LIMB
    for x in arr
        iszero(x) && return zero(BigInt)
        xsize = abs(x.size)
        lz = GC.@preserve x leading_zeros(unsafe_load(x.d, xsize))
        nbits += xsize * BITS_PER_LIMB - lz
    end
    init = BigInt(; nbits)
    MPZ.set_si!(init, 1)
    foldl(MPZ.mul!, arr; init)
end

factorial(n::BigInt) = !isnegative(n) ? MPZ.fac_ui(n) : throw(DomainError(n, "`n` must not be negative."))

function binomial(n::BigInt, k::Integer)
    k < 0 && return BigInt(0)
    k <= typemax(Culong) && return binomial(n, Culong(k))
    n < 0 && return isodd(k) ? -binomial(k - n - 1, k) : binomial(k - n - 1, k)
    κ = n - k
    κ < 0 && return BigInt(0)
    κ <= typemax(Culong) && return binomial(n, Culong(κ))
    throw(OverflowError("Computation would exceed memory"))
end
binomial(n::BigInt, k::Culong) = MPZ.bin_ui(n, k)

==(x::BigInt, y::BigInt) = cmp(x,y) == 0
==(x::BigInt, i::Integer) = cmp(x,i) == 0
==(i::Integer, x::BigInt) = cmp(x,i) == 0
==(x::BigInt, f::CdoubleMax) = isnan(f) ? false : cmp(x,f) == 0
==(f::CdoubleMax, x::BigInt) = isnan(f) ? false : cmp(x,f) == 0
iszero(x::BigInt) = x.size == 0
isone(x::BigInt) = x == Culong(1)

<=(x::BigInt, y::BigInt) = cmp(x,y) <= 0
<=(x::BigInt, i::Integer) = cmp(x,i) <= 0
<=(i::Integer, x::BigInt) = cmp(x,i) >= 0
<=(x::BigInt, f::CdoubleMax) = isnan(f) ? false : cmp(x,f) <= 0
<=(f::CdoubleMax, x::BigInt) = isnan(f) ? false : cmp(x,f) >= 0

<(x::BigInt, y::BigInt) = cmp(x,y) < 0
<(x::BigInt, i::Integer) = cmp(x,i) < 0
<(i::Integer, x::BigInt) = cmp(x,i) > 0
<(x::BigInt, f::CdoubleMax) = isnan(f) ? false : cmp(x,f) < 0
<(f::CdoubleMax, x::BigInt) = isnan(f) ? false : cmp(x,f) > 0
isnegative(x::BigInt) = x.size < 0
ispositive(x::BigInt) = x.size > 0

signbit(x::BigInt) = isnegative(x)
flipsign!(x::BigInt, y::Integer) = (signbit(y) && (x.size = -x.size); x)
flipsign( x::BigInt, y::Integer) = signbit(y) ? -x : x
flipsign( x::BigInt, y::BigInt)  = signbit(y) ? -x : x
# above method to resolving ambiguities with flipsign(::T, ::T) where T<:Signed
function sign(x::BigInt)
    isnegative(x) && return -one(x)
    ispositive(x) && return one(x)
    return x
end

show(io::IO, x::BigInt) = print(io, string(x))

function string(n::BigInt; base::Integer = 10, pad::Integer = 1)
    base < 0 && return Base._base(Int(base), n, pad, (base>0) & (n.size<0))
    2 <= base <= 62 || throw(ArgumentError("base must be 2 ≤ base ≤ 62, got $base"))
    iszero(n) && pad < 1 && return ""
    nd1 = ndigits(n, base=base)
    nd  = max(nd1, pad)
    sv  = Base.StringMemory(nd + isnegative(n))
    GC.@preserve sv MPZ.get_str!(pointer(sv) + nd - nd1, base, n)
    @inbounds for i = (1:nd-nd1) .+ isnegative(n)
        sv[i] = '0' % UInt8
    end
    isnegative(n) && (sv[1] = '-' % UInt8)
    unsafe_takestring(sv)
end

function digits!(a::AbstractVector{T}, n::BigInt; base::Integer = 10) where {T<:Integer}
    if base ≥ 2
        if base ≤ 62
            # fast path using mpz_get_str via string(n; base)
            s = codeunits(string(n; base))
            i, j = firstindex(a)-1, length(s)+1
            lasti = min(lastindex(a), firstindex(a) + length(s)-1 - isnegative(n))
            while i < lasti
                # base ≤ 36: 0-9, plus a-z for 10-35
                # base > 36: 0-9, plus A-Z for 10-35 and a-z for 36..61
                x = s[j -= 1]
                a[i += 1] = base ≤ 36 ? (x>0x39 ? x-0x57 : x-0x30) : (x>0x39 ? (x>0x60 ? x-0x3d : x-0x37) : x-0x30)
            end
            lasti = lastindex(a)
            while i < lasti; a[i+=1] = zero(T); end
            return isnegative(n) ? map!(-,a,a) : a
        elseif a isa StridedVector{<:Base.BitInteger} && stride(a,1) == 1 && ispow2(base) && base-1 ≤ typemax(T)
            # fast path using mpz_export
            origlen = length(a)
            _, writelen = MPZ.export!(a, n; nails = 8sizeof(T) - trailing_zeros(base))
            length(a) != origlen && resize!(a, origlen) # truncate to least-significant digits
            a[begin+writelen:end] .= zero(T)
            return isnegative(n) ? map!(-,a,a) : a
        end
    end
    return invoke(digits!, Tuple{typeof(a), Integer}, a, n; base) # slow generic fallback
end

function ndigits0zpb(x::BigInt, b::Integer)
    b < 2 && throw(DomainError(b, "`b` cannot be less than 2."))
    x.size == 0 && return 0 # for consistency with other ndigits0z methods
    if ispow2(b) && 2 <= b <= 62 # GMP assumes b is in this range
        MPZ.sizeinbase(x, b)
    else
        # non-base 2 mpz_sizeinbase might return an answer 1 too big
        # use property that log(b, x) < ndigits(x, base=b) <= log(b, x) + 1
        n = MPZ.sizeinbase(x, 2)
        lb = log2(b) # assumed accurate to <1ulp (true for openlibm)
        q,r = divrem(n,lb)
        iq = Int(q)
        maxerr = q*eps(lb) # maximum error in remainder
        if r-1.0 < maxerr
            abs(x) >= big(b)^iq ? iq+1 : iq
        elseif lb-r < maxerr
            abs(x) >= big(b)^(iq+1) ? iq+2 : iq+1
        else
            iq+1
        end
    end
end

# Fast paths for nextpow(2, x::BigInt)
# below, ONE is always left-shifted by at least one digit, so a new BigInt is
# allocated, which can be safely mutated
_prevpow2(x::BigInt) = -2 <= x <= 2 ? x : flipsign!(ONE << (ndigits(x, base=2) - 1), x)
_nextpow2(x::BigInt) = count_ones_abs(x) <= 1 ? x : flipsign!(ONE << ndigits(x, base=2), x)

Base.checked_abs(x::BigInt) = abs(x)
Base.checked_neg(x::BigInt) = -x
Base.checked_add(a::BigInt, b::BigInt) = a + b
Base.checked_sub(a::BigInt, b::BigInt) = a - b
Base.checked_mul(a::BigInt, b::BigInt) = a * b
Base.checked_div(a::BigInt, b::BigInt) = div(a, b)
Base.checked_rem(a::BigInt, b::BigInt) = rem(a, b)
Base.checked_fld(a::BigInt, b::BigInt) = fld(a, b)
Base.checked_mod(a::BigInt, b::BigInt) = mod(a, b)
Base.checked_cld(a::BigInt, b::BigInt) = cld(a, b)
Base.add_with_overflow(a::BigInt, b::BigInt) = a + b, false
Base.sub_with_overflow(a::BigInt, b::BigInt) = a - b, false
Base.mul_with_overflow(a::BigInt, b::BigInt) = a * b, false

# checked_pow doesn't follow the same promotion rules as the others, above.
Base.checked_pow(x::BigInt, p::Integer) = x^p
Base.checked_pow(x::Integer, p::BigInt) = x^p
Base.checked_pow(x::BigInt, p::BigInt) = x^p

Base.deepcopy_internal(x::BigInt, stackdict::IdDict) = get!(() -> MPZ.set(x), stackdict, x)::BigInt

## streamlined hashing for BigInt, by avoiding allocation from shifts ##

Base._hash_shl!(x::BigInt, n) = MPZ.mul_2exp!(x, n)

if Limb === UInt64 === UInt
    # On 64 bit systems we can define
    # an optimized version for BigInt of hash_integer (used e.g. for Rational{BigInt}),
    # and of hash

    using .Base: HASH_SECRET, hash_bytes, hash_finalizer

    function hash_integer(n::BigInt, h::UInt)
        iszero(n) && return hash_integer(0, h)
        GC.@preserve n begin
            s = n.size
            h ⊻= (s < 0)

            us = abs(s)
            leading_zero_bytes = div(leading_zeros(unsafe_load(n.d, us)), 8)
            hash_bytes(
                Ptr{UInt8}(n.d),
                8 * us - leading_zero_bytes,
                h,
                HASH_SECRET
            )
        end
    end

    function hash(x::BigInt, h::UInt)
        GC.@preserve x begin
            sz = x.size
            sz == 0 && return hash(0, h)
            ptr = Ptr{UInt64}(x.d)
            if sz == 1
                return hash(unsafe_load(ptr), h)
            elseif sz == -1
                limb = unsafe_load(ptr)
                limb <= typemin(Int) % UInt && return hash(-(limb % Int), h)
            end
            pow = trailing_zeros(x)
            nd = Base.ndigits0z(x, 2)
            idx = (pow >>> 6) + 1
            shift = (pow & 63) % UInt
            upshift = BITS_PER_LIMB - shift
            asz = abs(sz)
            if shift == 0
                limb = unsafe_load(ptr, idx)
            else
                limb1 = unsafe_load(ptr, idx)
                limb2 = idx < asz ? unsafe_load(ptr, idx+1) : UInt(0)
                limb = limb2 << upshift | limb1 >> shift
            end
            if nd <= 1024 && nd - pow <= 53
                return hash(ldexp(flipsign(Float64(limb), sz), pow), h)
            end
            h = hash_integer(pow, h)

            h ⊻= (sz < 0)
            leading_zero_bytes = div(leading_zeros(unsafe_load(x.d, asz)), 8)
            trailing_zero_bytes = div(pow, 8)
            return hash_bytes(
                Ptr{UInt8}(x.d) + trailing_zero_bytes,
                8 * asz - (leading_zero_bytes + trailing_zero_bytes),
                h,
                HASH_SECRET
            )
        end
    end
end

module MPQ

# Rational{BigInt}
import .Base: unsafe_rational, __throw_rational_argerror_zero
import ..GMP: BigInt, MPZ, Limb, libgmp

gmpq(op::Symbol) = (Symbol(:__gmpq_, op), libgmp)

mutable struct _MPQ
    num_alloc::Cint
    num_size::Cint
    num_d::Ptr{Limb}
    den_alloc::Cint
    den_size::Cint
    den_d::Ptr{Limb}
    # to prevent GC
    rat::Rational{BigInt}
end

const mpq_t = Ref{_MPQ}

_MPQ(x::BigInt,y::BigInt) = _MPQ(x.alloc, x.size, x.d,
                                 y.alloc, y.size, y.d,
                                 unsafe_rational(BigInt, x, y))
_MPQ() = _MPQ(BigInt(), BigInt())
_MPQ(x::Rational{BigInt}) = _MPQ(x.num, x.den)

function sync_rational!(xq::_MPQ)
    xq.rat.num.alloc = xq.num_alloc
    xq.rat.num.size  = xq.num_size
    xq.rat.num.d     = xq.num_d
    xq.rat.den.alloc = xq.den_alloc
    xq.rat.den.size  = xq.den_size
    xq.rat.den.d     = xq.den_d
    return xq.rat
end

function Rational{BigInt}(num::BigInt, den::BigInt)
    if iszero(den)
        iszero(num) && __throw_rational_argerror_zero(BigInt)
        return set_si(flipsign(1, num), 0)
    end
    xq = _MPQ(MPZ.set(num), MPZ.set(den))
    ccall((:__gmpq_canonicalize, libgmp), Cvoid, (mpq_t,), xq)
    return sync_rational!(xq)
end

# define set, set_ui, set_si, set_z, and their inplace versions
function set!(z::Rational{BigInt}, x::Rational{BigInt})
    zq = _MPQ(z)
    ccall((:__gmpq_set, libgmp), Cvoid, (mpq_t, mpq_t), zq, _MPQ(x))
    return sync_rational!(zq)
end

function set_z!(z::Rational{BigInt}, x::BigInt)
    zq = _MPQ(z)
    ccall((:__gmpq_set_z, libgmp), Cvoid, (mpq_t, MPZ.mpz_t), zq, x)
    return sync_rational!(zq)
end

for (op, T) in ((:set, Rational{BigInt}), (:set_z, BigInt))
    op! = Symbol(op, :!)
    @eval $op(a::$T) = $op!(unsafe_rational(BigInt(), BigInt()), a)
end

# note that rationals returned from set_ui and set_si are not checked,
# set_ui(0, 0) will return 0//0 without errors, just like unsafe_rational
for (op, T1, T2) in ((:set_ui, Culong, Culong), (:set_si, Clong, Culong))
    op! = Symbol(op, :!)
    @eval begin
        function $op!(z::Rational{BigInt}, a, b)
            zq = _MPQ(z)
            ccall($(gmpq(op)), Cvoid, (mpq_t, $T1, $T2), zq, a, b)
            return sync_rational!(zq)
        end
        $op(a, b) = $op!(unsafe_rational(BigInt(), BigInt()), a, b)
    end
end

# define add, sub, mul, div, and their inplace versions
function add!(z::Rational{BigInt}, x::Rational{BigInt}, y::Rational{BigInt})
    if iszero(x.den) || iszero(y.den)
        if iszero(x.den) && iszero(y.den) && isnegative(x.num) != isnegative(y.num)
            throw(DivideError())
        end
        return set!(z, iszero(x.den) ? x : y)
    end
    zq = _MPQ(z)
    ccall((:__gmpq_add, libgmp), Cvoid,
          (mpq_t,mpq_t,mpq_t), zq, _MPQ(x), _MPQ(y))
    return sync_rational!(zq)
end

function sub!(z::Rational{BigInt}, x::Rational{BigInt}, y::Rational{BigInt})
    if iszero(x.den) || iszero(y.den)
        if iszero(x.den) && iszero(y.den) && isnegative(x.num) == isnegative(y.num)
            throw(DivideError())
        end
        iszero(x.den) && return set!(z, x)
        return set_si!(z, flipsign(-1, y.num), 0)
    end
    zq = _MPQ(z)
    ccall((:__gmpq_sub, libgmp), Cvoid,
          (mpq_t,mpq_t,mpq_t), zq, _MPQ(x), _MPQ(y))
    return sync_rational!(zq)
end

function mul!(z::Rational{BigInt}, x::Rational{BigInt}, y::Rational{BigInt})
    if iszero(x.den) || iszero(y.den)
        if iszero(x.num) || iszero(y.num)
            throw(DivideError())
        end
        return set_si!(z, ifelse(xor(isnegative(x.num), isnegative(y.num)), -1, 1), 0)
    end
    zq = _MPQ(z)
    ccall((:__gmpq_mul, libgmp), Cvoid,
          (mpq_t,mpq_t,mpq_t), zq, _MPQ(x), _MPQ(y))
    return sync_rational!(zq)
end

function div!(z::Rational{BigInt}, x::Rational{BigInt}, y::Rational{BigInt})
    if iszero(x.den)
        if iszero(y.den)
            throw(DivideError())
        end
        isnegative(y.num) || return set!(z, x)
        return set_si!(z, flipsign(-1, x.num), 0)
    elseif iszero(y.den)
        return set_si!(z, 0, 1)
    elseif iszero(y.num)
        if iszero(x.num)
            throw(DivideError())
        end
        return set_si!(z, flipsign(1, x.num), 0)
    end
    zq = _MPQ(z)
    ccall((:__gmpq_div, libgmp), Cvoid,
          (mpq_t,mpq_t,mpq_t), zq, _MPQ(x), _MPQ(y))
    return sync_rational!(zq)
end

for (fJ, fC) in ((:+, :add), (:-, :sub), (:*, :mul), (://, :div))
    fC! = Symbol(fC, :!)
    @eval begin
        ($fC!)(x::Rational{BigInt}, y::Rational{BigInt}) = $fC!(x, x, y)
        (Base.$fJ)(x::Rational{BigInt}, y::Rational{BigInt}) = $fC!(unsafe_rational(BigInt(), BigInt()), x, y)
    end
end

function Base.cmp(x::Rational{BigInt}, y::Rational{BigInt})
    Int(ccall((:__gmpq_cmp, libgmp), Cint, (mpq_t, mpq_t), _MPQ(x), _MPQ(y)))
end

end # MPQ module

end # module
