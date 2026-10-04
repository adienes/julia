using Random, Serialization
include(joinpath(@__DIR__, "oracle.jl"))
include(joinpath(@__DIR__, ENV["TPFILE"]))
include(joinpath(@__DIR__, "..", "test", "testhelpers", "IntersectionWitnesses.jl"))
const IW = IntersectionWitnesses
function main(out)
    T0 = time()
    qs = Any[]
    for seed in 1:3
        rng = Xoshiro(seed)
        for _ in 1:2000
            A = IW.gentype(rng, (:T, :S)[1:rand(rng, 1:2)])
            seeds = filter(w -> w <: A, IW.instances(rng, A, 4))
            isempty(seeds) && continue
            B = IW.generalization(rng, seeds[1])
            IW.instances(rng, A, 10); IW.instances(rng, B, 10)
            for (x, y) in ((A, B), (B, A))
                t = try TwoPhase.intersect(x, y) catch; nothing end
                push!(qs, (x, y, t, typeintersect(x, y)))
            end
        end
    end
    println(out, length(qs), " rows computed in ", round(time() - T0, digits = 1), "s"); flush(out)
    # rows where the prototype and C differ, and every 10th row where they agree
    sel = [i for (i, (x, y, t, cr)) in enumerate(qs) if t !== nothing && (t != cr || get(ENV, "SAMPLE", "") != "" && i % 10 == 0)]
    println(out, length(sel), " rows selected (", count(i -> qs[i][3] != qs[i][4], sel), " differing)"); flush(out)
    judged = Vector{Any}(nothing, length(qs))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for i in sel
        x, y, t, cr = qs[i]
        jt = Oracle.judge(x, y, t)
        judged[i] = (jt, t == cr ? jt : Oracle.judge(x, y, cr))
        (Threads.atomic_add!(done, 1) + 1) % 50 == 0 && (println(out, "judged ", done[], " ", round(time() - T0, digits = 1), "s"); flush(out))
    end
    c = Dict{Symbol,Int}()
    bump(k) = (c[k] = get(c, k, 0) + 1)
    rows = Any[]
    for (i, (x, y, t, cr)) in enumerate(qs)
        j = judged[i]
        j === nothing && continue
        jt, jc = j
        (jt === nothing || jc === nothing) && (bump(:too_big); continue)
        bump(:rows)
        mt, et, s = jt; mc, ec, _ = jc
        bump(isempty(mt) ? (isempty(et) ? :tp_exact : :tp_sound_excess) : :tp_MISSING)
        bump(isempty(mc) ? (isempty(ec) ? :c_exact : :c_sound_excess) : :c_missing)
        isempty(et) && !isempty(ec) && bump(:tp_exact_c_not)
        !isempty(et) && isempty(ec) && isempty(mc) && bump(:c_exact_tp_not)
        !isempty(et) && !isempty(ec) && bump(:both_excess)
        push!(rows, (x, y, t, cr, length(mt), length(et), length(mc), length(ec), length(s), first(et, 3), first(mt, 3)))
    end
    serialize(joinpath(@__DIR__, ENV["OUT"] * ".jls"), rows)
    for k in sort!(collect(keys(c))); println(out, rpad(k, 18), c[k]); end
    println(out, "--- prototype misses (unsound on the universe)")
    for r in filter(r -> r[5] > 0, rows)[1:min(end, 5)]
        println(out, "  A=", r[1], "\n  B=", r[2], "\n  TP=", r[3], "\n  missing e.g. ", r[11])
    end
    println(out, "--- C exact, prototype not")
    for r in filter(r -> r[6] > 0 && r[8] == 0 && r[7] == 0, rows)[1:min(end, 8)]
        println(out, "  A=", r[1], "\n  B=", r[2], "\n  TP=", r[3], "\n  C =", r[4], "\n  excess ", r[6], " e.g. ", r[10])
    end
    println(out, "total ", round(time() - T0, digits = 1), "s")
end
open(main, joinpath(@__DIR__, ENV["OUT"]), "w")
