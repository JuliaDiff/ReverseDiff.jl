##########
## fill ##
##########

function Base.fill(
    value::TrackedReal,
    dims::Vararg{Union{Integer, AbstractUnitRange}},
)
    return track(fill, value, dims...)
end
@grad function fill(v::Real, dims...)
    return fill(value(v), dims...), function(Δ)
        size(Δ) ≢  dims && error("Dimension mismatch")
        return (sum(Δ), map(_->nothing, dims)...)
    end
end

###############
## any & all ##
###############

Base.any(f::Function, x::TrackedArray; dims=:) = any(f, value(x), dims = dims)
Base.all(f::Function, x::TrackedArray; dims=:) = all(f, value(x), dims = dims)

#########
## cat ##
#########

function combinations(xs, n)
    n < 1 && return [[]]
    cs = combinations(xs, n-1)
    [[x, c...] for x in xs, c in cs]
end

for f in [:hcat, :vcat]
    for i = 0:2, c = combinations([:AbstractVector, :TrackedVector, :AbstractMatrix, :TrackedMatrix, :Number, :TrackedReal], i)
        cnames = map(_ -> gensym(), c)
        @eval begin
            Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedVector) = track($f, $(cnames...), x)
            Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedMatrix) = track($f, $(cnames...), x)
            Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedReal) = track($f, $(cnames...), x)
        end
        for T in [
            :AbstractVector,
            :AbstractMatrix,
            :Number,
            :AbstractVecOrMat,
            :(Union{AbstractVector, Number}),
        ]
            @eval begin
                Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedVector, xs::$T...) = track($f, $(cnames...), x, xs...)
                Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedMatrix, xs::$T...) = track($f, $(cnames...), x, xs...)
                Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedReal, xs::$T...) = track($f, $(cnames...), x, xs...)
            end
        end
    end
end

@grad function vcat(xs::Union{Number, AbstractVecOrMat}...)
    xs_value = value.(xs)
    out_value = reduce(vcat,xs_value)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          if xsi isa Number
            d = Δ[start+1]
          else
            d = Δ[start+1:start+size(xsi,1), :]
          end
          start += size(xsi, 1)
          d
        end
        return (Δs...,)
    end
    return out_value, back
end

@grad function hcat(xs::Union{Number, AbstractVecOrMat}...)
    xs_value = value.(xs)
    out_value = reduce(hcat, xs_value)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          d = if ndims(xsi) == 0
            Δ[start+1]
          elseif ndims(xsi) == 1
            Δ[:, start+1]
          else
            i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
            Δ[:, start+1:start+size(xsi,2), i...]
          end
          start += size(xsi, 2)
          d
        end
        return (Δs...,)
    end        
    return out_value, back
end

for i = 0:2, c = combinations([:AbstractArray, :TrackedArray, :Number, :TrackedReal], i)
    cnames = map(_ -> gensym(), c)
    @eval Base.cat($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::Union{TrackedArray,TrackedReal}, xs::Union{AbstractArray,Number}...; dims) = track(cat, $(cnames...), x, xs...; dims=dims)
end
@grad function cat(Xs::Union{Number, AbstractArray}...; dims)
    Xs_value = value.(Xs)
    return cat(Xs_value...; dims = dims), Δ -> begin
        start = ntuple(i -> 0, Val(ndims(Δ)))
        Δs = map(Xs) do xs
          if xs isa Number
            d = Δ[start+1]
            start = start .+ 1
          else
            dim_xs = 1:ndims(xs)
            till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
            xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
            d = reshape(Δ[xs_in_Δ...],size(xs))
            start = start .+ till_xs
          end
          d
        end
        return (Δs...,)
    end
end
