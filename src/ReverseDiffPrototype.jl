module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue
import ForwardDiff: Dual, value, partials

include("TraceReal.jl")
include("api.jl")

end # module
