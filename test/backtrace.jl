# This file is a part of Julia. License is MIT: https://julialang.org/license

import Base.StackTraces: lookup

# Test location information for inlined code (ref issues #1334 #12544)
module test_inline_bt
using Test
import ..lookup

function get_bt_frames(functionname, bt)
    for i = 1:length(bt)
        lkup = lookup(bt[i])
        lkup[end].func == functionname && return lkup
    end
    return StackTraces.StackFrame[]
end

# same-file inline
eval(Expr(:function, Expr(:call, :test_inline_1),
          Expr(:block, Expr(:line, 99, Symbol("backtrace.jl")),
                       Expr(:block, Expr(:line, 42),
                                    Expr(:meta, :push_loc, Symbol("backtrace.jl"), :inlfunc),
                                    Expr(:line, 37),
                                    Expr(:call, :throw, "foo"),
                                    Expr(:meta, :pop_loc),
                                    Expr(:line, 99)))))

@test functionloc(test_inline_1) == ("backtrace.jl", 99)
try
    test_inline_1()
    error("unexpected")
catch err
    lkup = get_bt_frames(:test_inline_1, catch_backtrace())
    @test length(lkup) == 2
    @test endswith(string(lkup[2].file), "backtrace.jl")
    @test lkup[2].line == 42
    # TODO: we don't support surface AST locations with inlined function names
    @test_broken lkup[1].func === :inlfunc
    @test endswith(string(lkup[1].file), "backtrace.jl")
    @test lkup[1].line == 37
end

# different-file inline
const absfilepath = Sys.iswindows() ? "C:\\foo\\bar\\baz.jl" : "/foo/bar/baz.jl"
eval(Expr(:function, Expr(:call, :test_inline_2),
          Expr(:block, Expr(:line, 81, Symbol("backtrace.jl")),
                       Expr(:block, Expr(:meta, :push_loc, Symbol(absfilepath)),
                                    Expr(:line, 111),
                                    Expr(:call, :throw, "foo"),
                                    Expr(:meta, :pop_loc),
                                    Expr(:line, 99)))))

@test functionloc(test_inline_2) == ("backtrace.jl", 81)
try
    test_inline_2()
    error("unexpected")
catch err
    lkup = get_bt_frames(:test_inline_2, catch_backtrace())
    @test length(lkup) == 2
    @test endswith(string(lkup[2].file), "backtrace.jl")
    @test lkup[2].line == 81
    @test string(lkup[1].file) == absfilepath
    @test lkup[1].line == 111
end

end # module

#issue 12977: line numbers for kwarg methods.
linenum = @__LINE__; f12977(; args...) = ()
loc = functionloc(f12977)
@test endswith(loc[1], "backtrace.jl")
@test loc[2] == linenum

@noinline function test_throw_commoning(x)
    if x==1; throw(AssertionError()); end
    if x==2; throw(AssertionError()); end
end

let
    local b1, b2
    try
        test_throw_commoning(1)
    catch
        b1 = stacktrace(catch_backtrace())
    end
    try
        test_throw_commoning(2)
    catch
        b2 = stacktrace(catch_backtrace())
    end
    i1 = findfirst(frame -> frame.func === :test_throw_commoning, b1)
    i2 = findfirst(frame -> frame.func === :test_throw_commoning, b2)
    @test i1 > 0 && i2 > 0
    @test b1[i1].line != b2[i2].line
end

module BackTraceTesting
using Test
import ..lookup

@inline bt2() = backtrace()
@inline bt1() = bt2()
bt() = bt1()

lkup = map(lookup, bt())
hasbt = hasbt2 = false
for sfs in lkup
    for sf in sfs
        if sf.func === :bt
            global hasbt = true
        end
        if sf.func === :bt2
            global hasbt2 = true
        end
    end
end
@test hasbt
@test hasbt2

function btmacro()
    ret = @timed backtrace()
    ret[1]
end
lkup = map(lookup, btmacro())
hasme = hasbtmacro = false
for sfs in lkup
    for sf in sfs
        if sf.func === Symbol("macro expansion")
            global hasme = true
        end
        if sf.func === :btmacro
            global hasbtmacro = true
        end
    end
end
@test hasme
@test hasbtmacro

end

# Interpreter backtraces
bt = eval(quote
    try
        error()
    catch
        catch_backtrace()
    end
end)
lkup = map(lookup, bt)
hastoplevel = false
for sfs in lkup
    for sf in sfs
        if sf.linfo isa Core.CodeInfo
            global hastoplevel = true
        end
    end
end
@test hastoplevel

# issue #23971
let
    for i = 1:1
        global bt23971 = backtrace()
    end
end
let st = stacktrace(bt23971)
    @test StackTraces.is_top_level_frame(st[1])
    @test string(st[1].file) == @__FILE__
    @test !occursin("missing", string(st[2].file))
end

# issue #27959
let bt, found = false
    @testset begin
        bt = backtrace()
    end
    for frame in map(lookup, bt)
        if frame[1].line == @__LINE__() - 3 && frame[1].file === Symbol(@__FILE__)
            found = true; break
        end
    end
    @test found
end

# issue 28618
let bt, found = false
    @debug ""
    bt = backtrace()
    for frame in map(lookup, bt)
        if frame[1].line == @__LINE__() - 2 && frame[1].file === Symbol(@__FILE__)
            found = true; break
        end
    end
    @test found
end

# Syntax error locations appear in backtraces
let trace = try
        eval(Expr(:error, 1))
    catch
        stacktrace(catch_backtrace())
    end
    @test trace[1].func === Symbol("top-level scope")
end
let trace = try
        eval(Expr(:toplevel, LineNumberNode(3, :a_filename), Expr(:error, 1)))
    catch
        stacktrace(catch_backtrace())
    end
    @test trace[1].func === Symbol("top-level scope")
    @test trace[1].file === :a_filename
    @test trace[1].line == 3
end
let trace = try
        include_string(@__MODULE__,
            """

            )

            """, "a_filename")
    catch
        stacktrace(catch_backtrace())
    end
    @test trace[1].func === Symbol("top-level scope")
    @test trace[1].file === :a_filename
    @test trace[1].line == 2
end
let trace = try
        include_string(@__MODULE__,
            """

            incomplete_syntax(

            """, "a_filename")
    catch
        stacktrace(catch_backtrace())
    end
    @test trace[1].func === Symbol("top-level scope")
    @test trace[1].file === :a_filename
    @test trace[1].line in (2, 3)
end

# issue #45171
linenum = @__LINE__; function f45171(;kwarg = true)
    1
    error()
end
let trace = try
        f45171()
    catch
        stacktrace(catch_backtrace())
    end
    @test trace[3].line == linenum
end

# issue #29695 (see also test for #28442)
let code = """
    f29695(c) = g29695(c)
    g29695(c) = c >= 1000 ? (return backtrace()) : f29695(c + 1)
    bt = f29695(1)
    meth_names = [ip.code.def.name for ip in bt
                  if ip isa Base.InterpreterIP && ip.code isa Core.MethodInstance]
    num_fs = sum(meth_names .== :f29695)
    num_gs = sum(meth_names .== :g29695)
    if num_fs != 1000 || num_gs != 1000
        Base.show_backtrace(stderr, bt)
        error("Expected 1000 frames each, got \$num_fs, \$num_fs")
    end
    exit()
    """

    @test success(pipeline(`$(Base.julia_cmd()) --startup-file=no --compile=min -e $code`; stderr))
end

# Test that modules make it into InterpreterIP for top-level code
let code = """
    module A
    foo() = error("Expected")
    try
        foo()
    catch
        global bt = catch_backtrace()
    end
    end

    foreach(println, A.bt)
    """

    bt_str = read(`$(Base.julia_cmd()) --startup-file=no --compile=min -e $code`, String)
    @test occursin(r"InterpreterIP in MethodInstance for .*A\.foo", bt_str)
    @test occursin("InterpreterIP in top-level CodeInfo for Main.A", bt_str)
end

"""
    _reformat_sp(bt_data...) -> sp::Vector{Ptr{Cvoid}}

Convert the output `bt_data` of `jl_backtrace_from_here` with `returnsp` flag set to a
vector of valid stack pointers `sp`; i.e., `sp` is a subset of `bt_data[3]`.

See also `Base._reformat_bt`.
"""
function _reformat_sp(
    bt_raw::Array{Ptr{Cvoid},1},
    bt2::Array{Any,1},
    sp_raw::Array{Ptr{Cvoid},1},
)
    bt = Base._reformat_bt(bt_raw, bt2)
    sp = empty!(similar(sp_raw))
    i = j = 0
    while true
        # Advance `i` such that `bt[i] isa Ptr{Cvoid}` (native pointer).
        local ip
        while true
            if i == lastindex(bt)
                return sp
            end
            i += 1
            x = bt[i]
            if x isa Ptr{Cvoid}
                ip = x
                break
            end
        end
        # Advance `j` such that `bt_raw[j] == bt[i]` to find a valid stack pointer.
        while true
            if j == lastindex(bt_raw)
                return sp
            end
            j += 1
            if bt_raw[j] == ip
                push!(sp, sp_raw[j])
                break
            end
        end
    end
end

"""
    withframeaddress(f)

Call function `f` with an address `ptr::Ptr{Cvoid}` of an independent frame
immediately outer to `f`.
"""
withframeaddress
@eval @noinline function withframeaddress(f)
    sp = Core.Intrinsics.llvmcall(
        ($"""
        declare i8* @llvm.frameaddress(i32)
        define private i$(Sys.WORD_SIZE) @frameaddr() {
            %1 = call i8* @llvm.frameaddress(i32 0)
            %2 = ptrtoint i8* %1 to i$(Sys.WORD_SIZE)
            ret i$(Sys.WORD_SIZE) %2
        }""", "frameaddr"),
        UInt,
        Tuple{},
    )
    @noinline f(Ptr{Cvoid}(sp))
end

function sandwiched_backtrace()
    local ptr1, ptr2, bt
    withframeaddress() do p1
        ptr1 = p1
        bt = ccall(:jl_backtrace_from_here, Ref{Base.SimpleVector}, (Cint, Cint), true, 0)
        withframeaddress() do p2
            ptr2 = p2
        end
    end
    return ptr1, ptr2, bt
end

@testset "stack pointers" begin
    ptr1, ptr2, bt_data = sandwiched_backtrace()
    sp = _reformat_sp(bt_data...)
    @test ptr2 < sp[2]
    @test sp[1] < ptr1
    @test all(diff(Int128.(UInt.(sp))) .> 0)
end
