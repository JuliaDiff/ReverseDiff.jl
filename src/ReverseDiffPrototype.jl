module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue
import ForwardDiff: Dual, Partials, value, partials, numtype

include("TraceReal.jl")
include("trace.jl")
include("array_ops.jl")
include("api.jl")

end # module
