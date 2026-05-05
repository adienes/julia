# Serialization stdlib benchmark suite.
#
# Usage (pick one):
#
#   $ <julia> --project=<env-with-BenchmarkTools> /tmp/serialization_bench/suite.jl
#
#   julia> using Pkg; Pkg.add("BenchmarkTools")
#   julia> include("/tmp/serialization_bench/suite.jl")
#
# To measure changes to stdlib/Serialization without rebuilding:
#
#   $ <julia> -e 'using Revise; Revise.track(Serialization); include("/tmp/serialization_bench/suite.jl")'

using Serialization, BenchmarkTools, LinearAlgebra

const N = 1_000_000
const x_int = rand(Int, N)
const x_um  = convert(Vector{Union{Int,Missing}}, x_int)
x_um[1:1000:end] .= missing

mutable struct MinimalNode
    branches1::Union{NTuple{1,MinimalNode}, NTuple{2,MinimalNode}, NTuple{3,MinimalNode}, NTuple{4,MinimalNode}, Tuple{}}
    branches2::Union{NTuple{1,MinimalNode}, NTuple{2,MinimalNode}, NTuple{3,MinimalNode}, NTuple{4,MinimalNode}, Tuple{}}
    MinimalNode() = (n = new(); n.branches1 = (); n.branches2 = (); n)
end
function _addbranches!(node, depth)
    depth == 0 && return
    if isodd(depth)
        node.branches1 = (MinimalNode(),)
        for c in node.branches1; _addbranches!(c, depth-1); end
        node.branches2 = (MinimalNode(), MinimalNode())
        for c in node.branches2; _addbranches!(c, depth-1); end
    else
        node.branches1 = (MinimalNode(), MinimalNode(), MinimalNode())
        for c in node.branches1; _addbranches!(c, depth-1); end
        node.branches2 = (MinimalNode(), MinimalNode(), MinimalNode(), MinimalNode())
        for c in node.branches2; _addbranches!(c, depth-1); end
    end
end
const trees = [(n = MinimalNode(); _addbranches!(n, 4); n) for _ in 1:50]

const inner_union = Base.unwrap_unionall(
    methods(LinearAlgebra.herk_wrapper!).ms[1].sig).parameters[2]

serbytes(x) = (io = IOBuffer(); serialize(io, x); take!(io))
const x_int_bytes       = serbytes(x_int)
const x_um_bytes        = serbytes(x_um)
const trees_bytes       = serbytes(trees)
const inner_union_bytes = serbytes(inner_union)

println("=== #30148 Vector{Int} (baseline) ===")
print("serialize:   "); @btime serialize(IOBuffer(), $x_int)
print("deserialize: "); @btime deserialize(IOBuffer($x_int_bytes))

println("\n=== #30148 Vector{Union{Int,Missing}} ===")
print("serialize:   "); @btime serialize(IOBuffer(), $x_um)
print("deserialize: "); @btime deserialize(IOBuffer($x_um_bytes))

println("\n=== #44175 50 trees of depth 4 ===")
print("serialize:   "); @btime serialize(IOBuffer(), $trees)
print("deserialize: "); @btime deserialize(IOBuffer($trees_bytes))

println("\n=== #58007 inner Union of herk_wrapper!.sig ===")
print("serialize:   "); @btime serialize(IOBuffer(), $inner_union)
print("deserialize: "); @btime deserialize(IOBuffer($inner_union_bytes))
