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
    for i = 0:2, c = combinations([:AbstractArray, :TrackedArray, :Number, :TrackedReal], i)
        cnames = map(_ -> gensym(), c)
        @eval Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::Union{TrackedArray,TrackedReal}, xs::Union{AbstractArray,Number}...) = track($f, $(cnames...), x, xs...)
    end
    @eval begin
        Base.$f(x::TrackedVecOrMat{T}, xs::AbstractVecOrMat{T}...) where T = track($f, x, xs...)
        Base.$f(x1::TrackedVecOrMat{T}, x2::TrackedVecOrMat{T}, xs::AbstractVecOrMat{T}...) where T = track($f, x1, x2, xs...)
        Base.$f(x::TrackedVector{T}, xs::AbstractVector{T}...) where T = track($f, x, xs...)
        Base.$f(x1::TrackedVector{T}, x2::TrackedVector{T}, xs::AbstractVector{T}...) where T = track($f, x1, x2, xs...)

        @grad function $f(x::Real)
            $f(value(x)), (Δ) -> (Δ[1],)
        end
        @grad function $f(x1::Real, x2::Real)
            $f(value(x1), value(x2)), (Δ) -> (Δ[1], Δ[2])
        end
        @grad function $f(x1::AbstractVector, x2::Real)
            $f(value(x1), value(x2)), (Δ) -> (Δ[1:length(x1)], Δ[length(x1)+1])
        end
    end
end

@grad function vcat(xs::Union{TrackedVector, TrackedMatrix}...)
    xs_value = value.(xs)
    out_value = vcat(xs_value...)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          x = map(_ -> :, size(xsi))
          i = isempty(x) ? x : Base.tail(x)
          d = Δ[start+1:start+size(xsi,1), i...]
          start += size(xsi, 1)
          d
        end
        return (Δs...,)
    end
    return out_value, back
end

@grad function hcat(xs::Union{TrackedVector, TrackedMatrix}...)
    xs_value = value.(xs)
    out_value = reduce(hcat,xs_value)
    function back(Δ)
        start = 0
        Δs = map(xs) do xsi
          d = if ndims(xsi) == 1
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

Base.cat(Xs::TrackedArray...; dims) = track(cat, Xs...; dims = dims)
@grad function cat(Xs::TrackedArray{<:Any, D}...; dims) where {D}
    Xs_value = value.(Xs)
    return cat(Xs_value...; dims = dims), Δ -> begin
        start = ntuple(i -> 0, Val(ndims(Δ)))
        Δs = map(Xs) do xs
          dim_xs = 1:ndims(xs)
          till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
          xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
          d = reshape(Δ[xs_in_Δ...],size(xs))
          start = start .+ till_xs
          d
        end
        return (Δs...,)
    end
end
