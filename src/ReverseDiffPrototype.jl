module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue
import ForwardDiff: Dual, Partials, value, partials

include("TraceReal.jl")
include("array.jl")
include("trace.jl")
include("api.jl")

end # module
