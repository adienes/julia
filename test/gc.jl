# This file is a part of Julia. License is MIT: https://julialang.org/license

using Test

function run_gctest(file)
    let cmd = `$(Base.julia_cmd()) --depwarn=error --rr-detach --startup-file=no $file`
        @testset for test_nthreads in (1, 2, 4)
            @testset for test_nithreads in (0, 1)
                @testset for concurrent_sweep in (0, 1)
                    new_env = copy(ENV)
                    new_env["JULIA_NUM_THREADS"] = "$test_nthreads,$test_nithreads"
                    new_env["JULIA_NUM_GC_THREADS"] = "$(test_nthreads),$(concurrent_sweep)"
                    @test success(run(pipeline(setenv(cmd, new_env), stdout = stdout, stderr = stderr)))
                end
            end
        end
    end
end

function run_nonzero_page_utilization_test()
    GC.gc()
    page_utilization = Base.gc_page_utilization_data()
    # at least one of the pools should have nonzero page_utilization
    @test any(page_utilization .> 0)
end

function run_pg_size_test()
    page_size = @ccall jl_get_pg_size()::UInt64
    # supported page sizes: 4KB and 16KB
    @test page_size == (1 << 12) || page_size == (1 << 14)
end

function issue_54275_alloc_string()
    String(UInt8['a' for i in 1:10000000])
end

function issue_54275_test()
    GC.gc(true)
    baseline = Base.gc_live_bytes()
    live_bytes_has_grown_too_much = false
    for _ in 1:10
        issue_54275_alloc_string()
        GC.gc(true)
        if Base.gc_live_bytes() - baseline > 1_000_000
            live_bytes_has_grown_too_much = true
            break
        end
    end
    @test !live_bytes_has_grown_too_much
end

function full_sweep_reasons_test()
    GC.gc()
    reasons = Base.full_sweep_reasons()
    @test reasons[:FULL_SWEEP_REASON_FORCED_FULL_SWEEP] >= 1
    @test keys(reasons) == Set(Base.FULL_SWEEP_REASONS)
end

# Read the GC state and young_age bits out of an object's header.
# Layout (from src/julia.h `_jl_taggedvalue_bits`):
#   bits 0-1: gc (GC_CLEAN=0, GC_MARKED=1, GC_OLD=2, GC_OLD_MARKED=3)
#   bit  2:   in_image
#   bit  3:   young_age
function _read_tag_state(@nospecialize x)
    tag = unsafe_load(Ptr{UInt}(pointer_from_objref(x) - sizeof(UInt)))
    (gc = Int(tag & 0x3), age = Int((tag >> 3) & 0x1))
end

# Verify that an object is aged before being promoted to old gen, and that
# short-lived objects can die in young without ever reaching old gen.
function aging_state_machine_test()
    # Force a quiescent state first
    GC.gc(true); GC.gc(true)

    # Allocate a fresh object
    x = Ref(0)
    s0 = _read_tag_state(x)
    @test s0 == (gc = 0, age = 0)            # CLEAN, age 0

    # One incremental: alive, age++
    GC.gc(false)
    s1 = _read_tag_state(x)
    @test s1 == (gc = 0, age = 1)            # CLEAN, age 1 (still young)

    # Second incremental: alive at age 1, promote to OLD
    GC.gc(false)
    s2 = _read_tag_state(x)
    @test s2.gc == 2                         # GC_OLD
    @test s2.age == 0                        # age cleared on promotion

    # Verify mark transitions OLD -> OLD_MARKED on the next mark cycle.
    GC.gc(false)
    s3 = _read_tag_state(x)
    @test s3.gc == 3                         # GC_OLD_MARKED
end

# Verify that a young object that becomes unreachable before age MAX_YOUNG_AGE+1
# is freed in young (never reaches old gen).
function aging_short_lived_dies_in_young_test()
    # Wrap allocation in a function so the local stays alive only inside.
    weak = let
        x = Ref(0)
        GC.gc(false)                          # x: (CLEAN, age=1) after this
        WeakRef(x)
    end
    # x should be unreachable now. Two incrementals are enough to free it
    # without it ever reaching old gen.
    GC.gc(false)
    @test weak.value === nothing
end

# Issue #53018: workloads that build up multi-megabyte structures across
# iterations used to trigger an automatic full sweep on roughly every other
# incremental, since `promoted_bytes / heap_size > 0.15` fired on the still-live
# just-promoted bytes. Make sure we do not regress to that behavior.
function issue_53018_test()
    function work(m, n, k)
        res = 0.0
        for _ in 1:k
            res += sum(sum.([rand(rand(1:m)) for _ in 1:n]))
        end
        res
    end
    work(100, 100, 1)  # warmup
    GC.gc()
    before = Base.full_sweep_reasons()
    work(100, 100_000, 5)
    after = Base.full_sweep_reasons()
    full_sweeps = sum(after[k] - before[k] for k in keys(after))
    # Pre-fix this MWE caused ~30 automatic full sweeps; post-fix it is in
    # single digits. Set a generous bound so the test does not become flaky.
    @test full_sweeps < 10
end

# !!! note:
#     Since we run our tests on 32bit OS as well we confine ourselves
#     to parameters that allocate about 512MB of objects. Max RSS is lower
#     than that.
@testset "GC threads" begin
    run_gctest("gc/binarytree.jl")
    run_gctest("gc/linkedlist.jl")
    run_gctest("gc/objarray.jl")
    run_gctest("gc/chunks.jl")
end

#FIXME: Issue #57103 disabling tests for MMTk, since
# they rely on information that is specific to the stock GC.
@static if Base.USING_STOCK_GC
@testset "GC page metrics" begin
    run_nonzero_page_utilization_test()
    run_pg_size_test()
end

@testset "issue-54275" begin
    issue_54275_test()
end

@testset "Full GC reasons" begin
    full_sweep_reasons_test()
end

@testset "object aging" begin
    aging_state_machine_test()
    aging_short_lived_dies_in_young_test()
end

@testset "issue-53018" begin
    issue_53018_test()
end

@testset "GC Always Full" begin
    prog = "using Test;\n
        for _ in 1:10; GC.gc(); end;\n
        reasons = Base.full_sweep_reasons();\n
        @test reasons[:FULL_SWEEP_REASON_SWEEP_ALWAYS_FULL] >= 10;"
    cmd = `$(Base.julia_cmd()) --depwarn=error --startup-file=no --gc-sweep-always-full -e $prog`
    @test success(cmd)
end
end

@testset "Base.GC docstrings" begin
    @test isempty(Docs.undocumented_names(GC))
end

#testset doesn't work here because this needs to run in top level
#Check that we ensure objects in toplevel exprs are rooted
global dims54422 = [] # allocate the Binding
GC.gc(); GC.gc(); # force the binding to be old
GC.enable(false); # prevent new objects from being old
@eval begin
    Base.Experimental.@force_compile # use the compiler
    dims54422 = $([])
    nothing
end
GC.enable(true); GC.gc(false) # incremental collection
@test typeof(dims54422) == Vector{Any}
@test isempty(dims54422)
