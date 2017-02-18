#########
# Types #
#########

const NULL_INDEX = typemin(Int)
const NULL_TAPE = RawTape()

# TrackedReal #
#-------------#

#=
A `TrackedReal` stores a value and a reference back to the original `TrackedArray` which
provided the value.

When performing a forward pass through a previously-recorded `RawTape`, any encountered
`TrackedReal` instances which are direct descendents of their origin array must re-validate
themselves by re-retrieving their value from their origin via the given index. A similar
strategy is taken during the reverse pass, when derivs are updated; they are always
re-validated with the origin before and after "local" updates to the TrackedReal. The big
benefit of this strategy is that scalar `getindex` operations don't need to be explicitly
recorded to the tape.

Note that we don't have to worry about the origin's values being invalidated during `RawTape`
execution, since the `TrackedArray` type is immutable.

Also note that it's possible to instantiate an origin-less `TrackedReal`. This will most
often happen as a result of conversion, or via a numerical constructor (e.g. `zero`).
In that case, we leave the `origin` field uninitialized, and set the `index` field to
`NULL_INDEX`.

There are two alternative approaches we could've taken for dealing with the potential
missing-ness of the origin, and it's worth explaining why we don't take these approaches.

The first alternative is to encode the lack of an origin as a parameter in the
`TrackedReal` type. Indeed, this is exactly what we do if the origin of a `TrackedReal` is a
constructor like `zero`, setting `O` to `Void`. However, this strategy is not a viable
solution for origin-less `TrackedReal`s that result from conversions. For example, take the
expression `y = convert(::Type{T<:TrackedReal}, x::Real)`. `y` will not have an origin, but
`isa(y, T)` MUST hold true regardless. This could break in a world where `T` forcibly
specifies an origin of type `O != Void`.

The second alternative is to use `Nullable` + `isnull`. This solution would work, but it
overcomplicates the API and would incur unneccesary pointer loads during re-validation,
making the below implementation preferable.
=#

type TrackedReal{V<:Real,D<:Real,O} <: Real
    value::V
    deriv::D
    tape::RawTape
    index::Int
    origin::O
    TrackedReal(value, deriv, tape, index, origin) = new(value, deriv, tape, index, origin)
    TrackedReal(value, deriv, tape) = new(value, deriv, tape, NULL_INDEX)
    TrackedReal(value, deriv) = new(value, deriv, NULL_TAPE, NULL_INDEX)
    TrackedReal(value) = new(value, zero(D), NULL_TAPE, NULL_INDEX)
end

TrackedReal{V,D,O}(v::V, a::D, tp::RawTape, i::Int, o::O) = TrackedReal{V,D,O}(v, a, tp, i, o)

TrackedReal{V,D}(v::V, a::D, tp::RawTape = NULL_TAPE) = TrackedReal{V,D,Void}(v, a, tp)

# TrackedArray #
#--------------#

immutable TrackedArray{V,D,N,VA,DA} <: AbstractArray{TrackedReal{V,D,TrackedArray{V,D,N,VA,DA}},N}
    value::VA
    deriv::DA
    tape::RawTape
    function TrackedArray(value::AbstractArray{V,N}, deriv::AbstractArray{D,N}, tape::RawTape)
        @assert @compat Base.IndexStyle(value) === Base.IndexLinear()
        @assert size(value) === size(deriv)
        return new(value, deriv, tape)
    end
end

function TrackedArray{V,D,N}(value::AbstractArray{V,N},
                             deriv::AbstractArray{D,N},
                             tape::RawTape)
    return TrackedArray{V,D,N,typeof(value),typeof(deriv)}(value, deriv, tape)
end

###########
# getters #
###########

istracked(x) = false
istracked(::TrackedReal) = true
istracked(::TrackedArray) = true
istracked{T}(::AbstractArray{T}) = T <: TrackedReal || !(isleaftype(T))

@inline value(x) = x
@inline value(x::AbstractArray) = istracked(x) ? map(value, x) : x
@inline value(t::TrackedReal) = t.value
@inline value(t::TrackedArray) = t.value

@inline deriv(t::TrackedArray) = t.deriv
@inline deriv(t::TrackedReal) =  t.deriv

@inline valtype{V}(::TrackedReal{V}) = V
@inline valtype{V,D,O}(::Type{TrackedReal{V,D,O}}) = V
@inline valtype{V}(::TrackedArray{V}) = V
@inline valtype{V,D,VA,DA,N}(::Type{TrackedArray{V,D,N,VA,DA}}) = V

@inline derivtype{V,D}(::TrackedReal{V,D}) = D
@inline derivtype{V,D,O}(::Type{TrackedReal{V,D,O}}) = D
@inline derivtype{V,D}(t::TrackedArray{V,D}) = D
@inline derivtype{V,D,VA,DA,N}(::Type{TrackedArray{V,D,N,VA,DA}}) = D

@inline origintype{V,D,O}(::TrackedReal{V,D,O}) = O
@inline origintype{V,D,O}(::Type{TrackedReal{V,D,O}}) = O

@inline hasorigin(x::Real) = false
@inline hasorigin(t::TrackedReal) = t.index !== NULL_INDEX

@inline hastape(x) = false
@inline hastape(t::TrackedArray) = tape(t) !== NULL_TAPE
@inline hastape(t::TrackedReal) = tape(t) !== NULL_TAPE
@inline hastape(x::AbstractArray) = tape(x) !== NULL_TAPE

@inline tape(x) = NULL_TAPE
@inline tape(t::TrackedArray) = t.tape
@inline tape(t::TrackedReal) = t.tape

function tape(x::AbstractArray)
    if istracked(x)
        for i in x
            hastape(i) && return tape(i)
        end
    end
    return NULL_TAPE
end

function tape(ts...)
    for t in ts
        hastape(t) && return tape(t)
    end
    return NULL_TAPE
end

###########
# setters #
###########

@inline value!(t::TrackedReal, v::Real) = (t.value = v; nothing)
@inline value!(t::TrackedArray, v::AbstractArray) = (copy!(value(t), v); nothing)

function value!{N}(t::NTuple{N,Any}, v::NTuple{N,Any})
    for i in eachindex(t)
        value!(t[i], v[i])
    end
    return nothing
end

@inline deriv!(t::TrackedReal, v::Real) = (t.deriv = v; nothing)
@inline deriv!(t::TrackedArray, v::AbstractArray) = (copy!(deriv(t), v); nothing)

function deriv!{N}(t::NTuple{N,Any}, v::NTuple{N,Any})
    for i in eachindex(t)
        deriv!(t[i], v[i])
    end
    return nothing
end

# pulling values from origin #
#----------------------------#

pull_value!(x) = nothing
pull_value!(t::TrackedArray) = nothing
pull_value!(t::TrackedReal) = (hasorigin(t) && value!(t, value(t.origin)[t.index]); nothing)
pull_value!(x::AbstractArray) = (istracked(x) && foreach(pull_value!, x); nothing)

# pulling derivs from origin #
#----------------------------#

pull_deriv!(x) = nothing
pull_deriv!(t::TrackedArray) = nothing
pull_deriv!(t::TrackedReal) = (hasorigin(t) && deriv!(t, deriv(t.origin)[t.index]); nothing)
pull_deriv!(x::AbstractArray) = (istracked(x) && foreach(pull_deriv!, x); nothing)

# pushing derivs back to origin #
#-------------------------------#

push_deriv!(x) = nothing
push_deriv!(t::TrackedArray) = nothing
push_deriv!(t::TrackedReal) = (hasorigin(t) && (t.origin.deriv[t.index] = deriv(t)); nothing)
push_deriv!(x::AbstractArray) = (istracked(x) && foreach(push_deriv!, x); nothing)

# seed/unseed #
#-------------#

seed!(x) = nothing
seed!(t::TrackedReal) = (t.deriv = one(derivtype(t)); push_deriv!(t); nothing)
seed!(t::TrackedArray, i) = (t.deriv[i] = one(derivtype(t)); nothing)
seed!(x::AbstractArray, i) = seed!(x[i])

unseed!(x) = nothing
unseed!(t::TrackedReal) = (t.deriv = zero(derivtype(t)); push_deriv!(t); nothing)
unseed!(t::TrackedArray) = (fill!(deriv(t), zero(derivtype(t))); nothing)
unseed!(x::AbstractArray) = (istracked(x) && foreach(unseed!, x); nothing)
unseed!(t::Tuple) = foreach(unseed!, t)
unseed!(t::TrackedArray, i) = (t.deriv[i] = zero(derivtype(t)); nothing)
unseed!(x::AbstractArray, i) = unseed!(x[i])

#########################
# capture (see tape.jl) #
#########################

# This is type unstable, but that shouldn't be too much of a problem as it's only used below
# the function barrier of `record!`, which just pushes `t` to the tape and always returns
# `nothing`. It enables us to not waste time updating constants during
# `forward_pass!`/`reverse_pass!`.
capture(t::TrackedReal) = ifelse(hastape(t), t, value(t))
capture(t::TrackedArray) = t
capture(t::AbstractArray) = istracked(t) ?  map(capture, t) : copy(t)

########################
# Conversion/Promotion #
########################

# recording a instruction for this preserves the line of references back to the origin's deriv
function Base.convert{T1<:TrackedReal,T2<:TrackedReal}(::Type{T1}, t::T2)
    V1, D1, O1 = valtype(T1), derivtype(T1), origintype(T1)
    tp = tape(t)
    out = TrackedReal{V1,D1,O1}(V1(value(t)), D1(deriv(t)), tp)
    record!(tp, SpecialInstruction, convert, t, out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(convert)})
    output = instruction.output
    increment_deriv!(instruction.input, deriv(output))
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(convert)})
    input = instruction.input
    pull_value!(input)
    value!(instruction.output, value(input))
    return nothing
end

Base.convert{R<:Real,T<:TrackedReal}(::Type{R}, t::T) = R(value(t))
Base.convert{T<:TrackedReal,R<:Real}(::Type{T}, x::R) = TrackedReal{valtype(T),derivtype(T),origintype(T)}(valtype(T)(value(x)))

Base.convert{T<:TrackedReal}(::Type{T}, t::T) = t
Base.convert{T<:TrackedArray}(::Type{T}, t::T) = t

for R in REAL_TYPES
    @eval Base.promote_rule{V,D,O}(::Type{$R}, ::Type{TrackedReal{V,D,O}}) = TrackedReal{promote_type($R,V),D,O}
end

Base.promote_rule{R<:Real,V,D,O}(::Type{R}, ::Type{TrackedReal{V,D,O}}) = TrackedReal{promote_type(R,V),D,O}
Base.promote_rule{V1,V2,D1,D2,O1,O2}(::Type{TrackedReal{V1,D1,O1}}, ::Type{TrackedReal{V2,D2,O2}}) = TrackedReal{promote_type(V1,V2),promote_type(D1,D2),Void}

Base.promote_array_type{T<:TrackedReal, F<:AbstractFloat}(_, ::Type{T}, ::Type{F}) = promote_type(T, F)
Base.promote_array_type{T<:TrackedReal, F<:AbstractFloat, S}(_, ::Type{T}, ::Type{F}, ::Type{S}) = S
Base.promote_array_type{F<:AbstractFloat, T<:TrackedReal}(_, ::Type{F}, ::Type{T}) = promote_type(T, F)
Base.promote_array_type{F<:AbstractFloat, T<:TrackedReal, S}(_, ::Type{F}, ::Type{T}, ::Type{S}) = S

Base.r_promote{T<:TrackedReal}(::typeof(+), t::T) = t
Base.r_promote{T<:TrackedReal}(::typeof(*), t::T) = t

###########################
# AbstractArray Interface #
###########################

Base.getindex(t::TrackedArray, i::Int) = TrackedReal(value(t)[i], deriv(t)[i], tape(t), i, t)

colon2range(s, i) = i
colon2range(s, ::Colon) = s

function index_iterable{N,M}(shape::NTuple{N,Any}, i::NTuple{M,Any})
    if N < M
        return index_iterable(shape, ntuple(n -> i[n], Val{N}))
    elseif M < N && isa(last(i), Colon)
        return index_iterable(shape, ntuple(n -> (n > M ? Colon() : i[n]), Val{N}))
    else
        return compat_product(map(colon2range, shape[1:M], i)...)
    end
end

for T in (:Range, :Colon, :(Union{Colon,Range}))
    @eval function Base.getindex(t::TrackedArray, i::$(T)...)
        tp = tape(t)
        out = TrackedArray(value(t)[i...], deriv(t)[i...], tp)
        idx = index_iterable(indices(t), i)
        record!(tp, SpecialInstruction, getindex, (t, idx), out)
        return out
    end
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(getindex)})
    input, indices = instruction.input
    output = instruction.output
    input_deriv = deriv(input)
    output_deriv = deriv(output)
    i = 0
    for idx in indices
        input_deriv[CartesianIndex(idx)] += output_deriv[i += 1]
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(getindex)})
    input, indices = instruction.input
    input_value = value(input)
    output_value = value(instruction.output)
    i = 0
    for idx in indices
        output_value[i += 1] = input_value[CartesianIndex(idx)]
    end
    return nothing
end

Base.setindex!(t::TrackedArray, args...) = error("TrackedArrays do not support setindex!")

@compat Base.IndexStyle(::TrackedArray) = Base.IndexLinear()

Base.size(t::TrackedArray) = size(value(t))

Base.copy{T<:TrackedArray}(t::T) = t

Base.ones{V,D}(t::TrackedArray{V,D}) = ones(TrackedReal{V,D,Void}, size(t))

Base.zeros{V,D}(t::TrackedArray{V,D}) = zeros(TrackedReal{V,D,Void}, size(t))

reshape_body = :(TrackedArray(reshape(value(t), dims), reshape(deriv(t), dims), tape(t)))
@eval Base.reshape{N}(t::TrackedArray, dims::Type{Val{N}}) = $reshape_body
@eval Base.reshape{N}(t::TrackedArray, dims::Tuple{Vararg{Int,N}}) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Int64...) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::AbstractUnitRange...) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Union{AbstractUnitRange,Int64}...) = $reshape_body

####################
# `Real` Interface #
####################

Base.hash(t::TrackedReal) = hash(value(t))
Base.hash(t::TrackedReal, hsh::UInt64) = hash(value(t), hsh)

Base.deepcopy{T<:TrackedReal}(t::T) = t
Base.copy{T<:TrackedReal}(t::T) = t

function Base.float{V,D,O}(t::TrackedReal{V,D,O})
    v = float(value(t))
    return TrackedReal{typeof(v),D,O}(v)
end

Base.float{V<:AbstractFloat}(t::TrackedReal{V}) = t

Base.one{V,D,O}(::Type{TrackedReal{V,D,O}}) = TrackedReal{V,D,O}(one(V))
Base.zero{V,D,O}(::Type{TrackedReal{V,D,O}}) = TrackedReal{V,D,O}(zero(V))

Base.rand{V,D,O}(::Type{TrackedReal{V,D,O}}) = TrackedReal{V,D,O}(rand(V))
Base.rand{V,D,O}(rng::AbstractRNG, ::Type{TrackedReal{V,D,O}}) = TrackedReal{V,D,O}(rand(rng, V))

Base.eps(t::TrackedReal) = eps(value(t))
Base.eps{T<:TrackedReal}(::Type{T}) = eps(valtype(T))

Base.floor(t::TrackedReal) = floor(value(t))
Base.floor{R<:Real}(::Type{R}, t::TrackedReal) = floor(R, value(t))

Base.ceil(t::TrackedReal) = ceil(value(t))
Base.ceil{R<:Real}(::Type{R}, t::TrackedReal) = ceil(R, value(t))

Base.trunc(t::TrackedReal) = trunc(value(t))
Base.trunc{R<:Real}(::Type{R}, t::TrackedReal) = trunc(R, value(t))

Base.round(t::TrackedReal) = round(value(t))
Base.round{R<:Real}(::Type{R}, t::TrackedReal) = round(R, value(t))

################
# track/track! #
################

track(x::Real, tp::RawTape = RawTape()) = track(x, typeof(x), tp)

track(x::AbstractArray, tp::RawTape = RawTape()) = track(x, eltype(x), tp)

track{D}(x::Real, ::Type{D}, tp::RawTape = RawTape()) = TrackedReal(x, zero(D), tp)

track{D}(x::AbstractArray, ::Type{D}, tp::RawTape = RawTape()) = TrackedArray(x, fill!(similar(x, D), zero(D)), tp)

track!(t::TrackedArray, x::AbstractArray) = (value!(t, x); unseed!(t); t)

track!(t::TrackedReal, x::Real) = (value!(t, x); unseed!(t); t)

function track!{D}(t::AbstractArray{TrackedReal{D,D,Void}}, x::AbstractArray, tp::RawTape)
    for i in eachindex(t)
        t[i] = track(x[i], D, tp)
    end
    return t
end

###################
# Pretty Printing #
###################

idstr(x) = string(base(62, object_id(x)))[1:3]

function Base.show(io::IO, t::TrackedReal)
    tape_id = hastape(t) ? idstr(t.tape) : "---"
    origin_id = hasorigin(t) ? "$(t.index), $(idstr(t.origin))" : "---"
    id = idstr(t)
    print(io, "TrackedReal<$(id)>($(value(t)), $(deriv(t)), $(tape_id), $(origin_id))")
end
