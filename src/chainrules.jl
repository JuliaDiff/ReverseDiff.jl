using ChainRulesCore, ChainRules
using SpecialFunctions, NaNMath, DiffRules

const noop = (x) -> x

"""
In ChainRules, we define `rrule` for many functions, hence the function
`rrule` has many methods.

This function, `ruleof`, takes a method of the `rrule` function, and returns
the function for which the `rrule` is defined in the input method.
"""
function ruleof(method)
    parameters = if isa(method.sig, DataType)
        method.sig.parameters
    elseif method.sig.body |> typeof == DataType
        method.sig.body.parameters
    else
        []
    end
    if length(parameters) >= 2
        type2 = parameters[2]
        typeof(type2) == DataType && type2.super == Function && return type2.instance
    end

    # type2 is RuleConfig, use the next parameter
    if length(parameters) >= 3
        type3 = parameters[3]
        typeof(type3) == DataType && type3.super == Function && return type3.instance
    end

    return noop # return a function to keep type-stable
end


"""
    qname(Base.Filesystem.isdirpath) -> :(Base.Filesystem.isdirpath)

`qname` takes a function, returns its qualified name.
"""
function qname(f)
    mth = first(methods(f))
    names = [mth.name]
    mod = mth.module
    while true
        modname = nameof(mod)
        if names[1] == modname || :Main == modname
            break
        end
        pushfirst!(names, modname)
        mod = parentmodule(mod)
    end
    names
    expr = names[1]
    for s in names[2:end]
        expr = Expr(:., expr, QuoteNode(s))
    end
    return expr, names[1]
end



function import_rrules()
    # all the function who has `rrule` in ChainRules.
    FUNCS_WITH_RRULE = Set([ruleof(m) for m in methods(rrule)])

    # all the function who has special derivative rules in ReverseDiff.
    FUNCS_IN_RD = Set([eval(Expr(:., m, QuoteNode(f))) for (m, f, a) in DiffRules.diffrules()])


    # all the function who has derivative rules provided with `ReverseDiff.@grad`.
    FUNCS_WITH_GRAD = [Base.cat, Base.hcat, Base.vcat, fill,
                       _copytranspose, _copyadjoint]

    for func in FUNCS_WITH_RRULE
        in(func, FUNCS_IN_RD) && continue
        in(func, FUNCS_WITH_GRAD) && continue
        fname, mod = qname(func)
        isletter(String(mod)[1]) || continue
        occursin("#", repr(fname)) && continue
        @eval using $mod
        @eval @grad_from_cr $fname
    end
end
