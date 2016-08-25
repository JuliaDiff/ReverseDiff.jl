#############
# accessors #
#############

ForwardDiff.value{T}(arr::AbstractArray{T}) = value!(similar(arr, valtype(T)), arr)

function value!(out, arr)
    for i in eachindex(out)
        out[i] = value(arr[i])
    end
    return out
end


adjoint{T}(arr::AbstractArray{T}) = adjoint!(similar(arr, adjtype(T)), arr)

function adjoint!(out, arr)
    for i in eachindex(out)
        out[i] = adjoint(arr[i])
    end
    return out
end

###################
# trace selection #
###################

function trace(arr::AbstractArray)
    for t in arr
        !(isnull(trace(t))) && return trace(t)
    end
    return Nullable{Trace}()
end

trace(a::AbstractArray, b::AbstractArray) = isnull(trace(a)) ? trace(b) : trace(a)
trace(a::AbstractArray, b::AbstractArray, c::AbstractArray) = isnull(trace(a)) ? trace(b, c) : trace(a)

##############
# transforms #
##############

function dual_array{R<:TraceReal,N}(arr::AbstractArray{R}, ::Type{Val{N}}, i)
    T = valtype(R)
    out_partials = Partials{N,T}((z = zeros(T,N); z[i] = one(T); (z...)))
    out = similar(arr, Dual{N,T})
    for j in eachindex(out)
        out[j] = Dual{N,T}(value(arr[j]), out_partials)
    end
    return out
end

function dual_wrap{S,N,T}(::Type{S}, duals::AbstractArray{Dual{N,T}}, tr::Nullable{Trace})
    ts = similar(duals, TraceReal{S,T})
    ps = similar(duals, Partials{N,T})
    for i in eachindex(duals)
        dual = duals[i]
        ts[i] = TraceReal{S}(value(dual), tr)
        ps[i] = partials(dual)
    end
    return ts, ps
end

###########################
# optimized array methods #
###########################

# higher-order functions #
#------------------------#

const FASTDIFF_FUNCS = (:broadcast, :map)

fastdiffname(name::Symbol) = Symbol(string("fastdiff_", name))

macro fastdiff(x)
    if x.head == :call
        f = first(x.args)
        if in(f, FASTDIFF_FUNCS)
            x.args[1] = :(ReverseDiffPrototype.$(fastdiffname(first(x.args))))
            return esc(x)
        end
    end
    error("@fastdiff only works on calls to: $(FASTDIFF_FUNCS)")
end

for g in FASTDIFF_FUNCS
    fastg = fastdiffname(g)
    @eval begin
        # fallback
        @inline $(fastg)(args...) = Base.$(g)(args...)

        # 1 arg
        function $(fastg){S,T,N}(f, x::AbstractArray{TraceReal{S,T},N})
            duals = $(g)(f, dual_array(x, Val{1}, 1))
            tr = trace(x)
            out, partials = dual_wrap(S, duals, tr)
            record!(tr, nothing, x, out, partials)
            return out
        end

        # 2 args
        function $(fastg){S,T1,T2,N}(f,
                                     x1::AbstractArray{TraceReal{S,T1},N},
                                     x2::AbstractArray{TraceReal{S,T2},N})
            dual1 = dual_array(x1, Val{2}, 1)
            dual2 = dual_array(x2, Val{2}, 2)
            duals = $(g)(f, dual1, dual2)
            tr = trace(x1, x2)
            out, partials = dual_wrap(S, duals, tr)
            record!(tr, nothing, (x1, x2), out, partials)
            return out
        end

        # 3 args
        function $(fastg){S,T1,T2,T3,N}(f,
                                        x1::AbstractArray{TraceReal{S,T1},N},
                                        x2::AbstractArray{TraceReal{S,T2},N},
                                        x3::AbstractArray{TraceReal{S,T3},N})
            dual1 = dual_array(x1, Val{3}, 1)
            dual2 = dual_array(x2, Val{3}, 2)
            dual3 = dual_array(x3, Val{3}, 3)
            duals = $(g)(f, dual1, dual2, dual3)
            tr = trace(x1, x2, x3)
            out, partials = dual_wrap(S, duals, tr)
            record!(tr, nothing, (x1, x2, x3), out, partials)
            return out
        end
    end
end

# unary functions #
#-----------------#

for f in (:-, :inv, :det)
    for A in (:AbstractArray, :AbstractMatrix, :Array, :Matrix)
        @eval function Base.$(f){S,T}(x::$(A){TraceReal{S,T}})
            tr = trace(x)
            out = wrap(S, $(f)(value(x)), tr)
            record!(tr, $(f), x, out)
            return out
        end
    end
end

for A in (:AbstractArray, :Array)
    @eval function Base.sum{S,T}(x::$(A){TraceReal{S,T}})
        result = zero(T)
        for t in x
            result += value(t)
        end
        tr = trace(x)
        out = TraceReal{S}(result, tr)
        record!(tr, sum, x, out)
        return out
    end
end

# binary functions #
#------------------#

for f in (:-, :+, :*,
          :A_mul_Bt, :At_mul_B, :At_mul_Bt,
          :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
    @eval function Base.$(f){S,A,B}(a::AbstractMatrix{TraceReal{S,A}},
                                    b::AbstractMatrix{TraceReal{S,B}})
        tr = trace(a, b)
        out = wrap(S, $(f)(value(a), value(b)), tr)
        record!(tr, $(f), (a, b), out)
        return out
    end
end

# in-place A_mul_B family #
#-------------------------#

for (f!, f) in ((:A_mul_B!, :*),
                (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
    @eval function Base.$(f!){S,Y,A,B}(out::AbstractMatrix{TraceReal{S,Y}},
                                       a::AbstractMatrix{TraceReal{S,A}},
                                       b::AbstractMatrix{TraceReal{S,B}})
        tr = trace(a, b)
        wrap!(out, $(f)(value(a), value(b)), tr)
        record!(tr, $(f), (a, b), out)
        return out
    end
end
