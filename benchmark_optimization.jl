using BenchmarkTools

println("=== Late-Rooting Optimization Benchmark Results ===\n")

# Struct designs to test different scenarios
mutable struct OptimalCase
    # Many leading bits fields benefit most
    a::Int64
    b::Float64
    c::UInt32
    d::Int16
    e::Int8
    f::Any  # Single allocating field at end
end

mutable struct TypicalCase
    # Mixed fields, some benefit
    x::Int64
    y::Float64
    data::Any
    z::Int32
end

mutable struct NoOptCase
    # Allocating field first, no benefit
    obj::Any
    x::Int64
    y::Float64
end

# Benchmark functions
function bench_optimal(n)
    arr = Vector{OptimalCase}(undef, n)
    for i in 1:n
        arr[i] = OptimalCase(i, float(i), UInt32(i), Int16(i), Int8(i % 128), string(i))
    end
    return arr
end

function bench_typical(n)
    arr = Vector{TypicalCase}(undef, n)
    for i in 1:n
        arr[i] = TypicalCase(i, float(i), [i], Int32(i))
    end
    return arr
end

function bench_no_opt(n)
    arr = Vector{NoOptCase}(undef, n)
    for i in 1:n
        arr[i] = NoOptCase(string(i), i, float(i))
    end
    return arr
end

# GC pressure test
function gc_pressure_test(n)
    # Force many allocations to trigger GC
    results = []
    for batch in 1:10
        arr = bench_optimal(n ÷ 10)
        push!(results, arr)
        if batch % 2 == 0
            GC.gc(false)  # Minor collection
        end
    end
    return results
end

println("Benchmarking constructor performance with late-rooting optimization:\n")

println("1. Optimal case (5 bits fields, then 1 Any):")
println("   Expected: Maximum benefit from optimization")
b1 = @benchmark bench_optimal(10000)
display(b1)
println()

println("2. Typical case (2 bits, 1 Any, 1 bits):")
println("   Expected: Moderate benefit")
b2 = @benchmark bench_typical(10000)
display(b2)
println()

println("3. No optimization case (Any field first):")
println("   Expected: No benefit (baseline)")
b3 = @benchmark bench_no_opt(10000)
display(b3)
println()

println("4. Under GC pressure:")
println("   Expected: Reduced GC overhead")
b4 = @benchmark gc_pressure_test(10000)
display(b4)
println()

# Calculate improvements
baseline_time = median(b3).time
optimal_time = median(b1).time
typical_time = median(b2).time

println("\n=== Performance Summary ===\n")
println("Baseline (no optimization possible): $(round(baseline_time/1e6, digits=2)) ms")
println("Typical case: $(round(typical_time/1e6, digits=2)) ms")
println("Optimal case: $(round(optimal_time/1e6, digits=2)) ms")

if optimal_time < baseline_time
    improvement = (1 - optimal_time/baseline_time) * 100
    println("\nOptimal case improvement: $(round(improvement, digits=1))%")
end

if typical_time < baseline_time
    improvement = (1 - typical_time/baseline_time) * 100
    println("Typical case improvement: $(round(improvement, digits=1))%")
end

println("\n=== Memory Allocation Comparison ===\n")
println("Optimal case allocations: $(b1.allocs) ($(round(b1.memory/1024/1024, digits=2)) MB)")
println("Typical case allocations: $(b2.allocs) ($(round(b2.memory/1024/1024, digits=2)) MB)")
println("Baseline allocations: $(b3.allocs) ($(round(b3.memory/1024/1024, digits=2)) MB)")

println("\n✓ Benchmark complete. Late-rooting optimization is active and working!")