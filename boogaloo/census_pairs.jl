using Serialization

function collect_sigs()
    pairs = Tuple{String,Any,Any}[]
    seen = Set{Any}()
    for mod in (Core, Base), name in sort!(names(mod; all=true))
        isdefined(mod, name) || continue
        f = getglobal(mod, name)
        f isa Function || continue
        f in seen && continue
        push!(seen, f)
        ms = sort!(collect(methods(f)), by = m -> (string(m.file), m.line, string(m.sig)))
        length(ms) > 150 && continue
        for i in 1:length(ms), j in i+1:length(ms)
            key = string(name, " ", ms[i].file, ":", ms[i].line, " × ", ms[j].file, ":", ms[j].line)
            push!(pairs, (key, ms[i].sig, ms[j].sig))
        end
    end
    return pairs
end

