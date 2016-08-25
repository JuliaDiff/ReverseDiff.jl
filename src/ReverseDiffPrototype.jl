module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue

import ForwardDiff: Dual, Partials, value, valtype, partials

############################
# initial type definitions #
############################

include("Trace.jl")
include("TraceReal.jl")

##############
# wrap/wrap! #
##############

wrap(x, tr::Trace = Trace()) = wrap(eltype(x), x, tr)
wrap{S}(::Type{S}, x, tr::Trace = Trace()) = wrap(S, x, Nullable(tr))
wrap{S}(::Type{S}, x::Real, tr::Nullable{Trace}) = TraceReal{S}(x, tr)

function wrap{S}(::Type{S}, x, tr::Nullable{Trace})
    return wrap!(similar(x, TraceReal{S,eltype(x)}), x, tr)
end

function wrap!(out, x, tr::Nullable{Trace})
    S = adjtype(eltype(out))
    for i in eachindex(out)
        out[i] = TraceReal{S}(x[i], tr)
    end
    return out
end

###############################
# ...and the rest of the code #
###############################

include("array.jl")
include("backprop.jl")
include("api.jl")

export @fastdiff

end # module
