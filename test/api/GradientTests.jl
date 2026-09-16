module GradientTests

using DiffTests, ForwardDiff, ReverseDiff, Test, LinearAlgebra

include(joinpath(dirname(@__FILE__), "../utils.jl"))

function test_unary_gradient(f, x)
    test = ForwardDiff.gradient!(DiffResults.GradientResult(x), f, x)

    # without GradientConfig

    test_approx(ReverseDiff.gradient(f, x), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, f, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig(x)

    test_approx(ReverseDiff.gradient(f, x, cfg), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x, cfg)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with GradientTape

    seedx = rand(eltype(x), size(x))
    tp = ReverseDiff.GradientTape(f, seedx)

    test_approx(ReverseDiff.gradient!(tp, x), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, tp, x)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, tp, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        test_approx(ReverseDiff.gradient!(ctp, x), DiffResults.gradient(test))

        out = similar(x)
        ReverseDiff.gradient!(out, ctp, x)
        test_approx(out, DiffResults.gradient(test))

        result = DiffResults.GradientResult(x)
        ReverseDiff.gradient!(result, ctp, x)
        test_approx(DiffResults.value(result), DiffResults.value(test))
        test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    end
end

function test_ternary_gradient(f, a, b, c)
    test_val = f(a, b, c)
    test_a = ForwardDiff.gradient(x -> f(x, b, c), a)
    test_b = ForwardDiff.gradient(x -> f(a, x, c), b)
    test_c = ForwardDiff.gradient(x -> f(a, b, x), c)

    # without GradientConfig

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig((a, b, c))

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c), cfg)
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with GradientTape

    tp = ReverseDiff.GradientTape(f, (rand(eltype(a), size(a)), rand(eltype(b), size(b)), rand(eltype(c), size(c))))

    ∇a, ∇b, ∇c = ReverseDiff.gradient!(tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        ∇a, ∇b, ∇c = ReverseDiff.gradient!(ctp, (a, b, c))
        test_approx(∇a, test_a)
        test_approx(∇b, test_b)
        test_approx(∇c, test_c)

        ∇a, ∇b, ∇c = map(similar, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        test_approx(∇a, test_a)
        test_approx(∇b, test_b)
        test_approx(∇c, test_c)

        ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        test_approx(DiffResults.value(∇a), test_val)
        test_approx(DiffResults.value(∇b), test_val)
        test_approx(DiffResults.value(∇c), test_val)
        test_approx(DiffResults.gradient(∇a), test_a)
        test_approx(DiffResults.gradient(∇b), test_b)
        test_approx(DiffResults.gradient(∇c), test_c)
    end
end

# issue https://github.com/JuliaDiff/ReverseDiff.jl/issues/140
nested_array_mul_140(x) = sum(sum(x[1] * [[x[2], x[3]]]))
test_println("Issue #140", nested_array_mul_140)
test_unary_gradient(nested_array_mul_140, [1.0, 2.0, 1.0, -2.4, 4.0])

for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

# PR #227
norm_hermitian1(v) = (A = I - 2 * v * v'; norm(A' * A))
norm_hermitian2(v) = (A = I - 2 * v * transpose(v); norm(transpose(A) * A))
norm_hermitian3(v) = (A = I - 2 * v * collect(v'); norm(collect(A') * A))
norm_hermitian4(v) = (A = I - 2 * v * v'; norm(transpose(A) * A))
norm_hermitian5(v) = (A = I - 2 * v * transpose(v); norm(A' * A))
norm_hermitian6(v) = (A = (v'v)*I - 2 * v * v'; norm(A' * A))

for f in (norm_hermitian1, norm_hermitian2, norm_hermitian3,
            norm_hermitian4, norm_hermitian5, norm_hermitian6)
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    test_println("TERNARY_MATRIX_TO_NUMBER_FUNCS", f)
    test_ternary_gradient(f, rand(5, 5), rand(5, 5), rand(5, 5))
end

# logical indices, which are only normalized when the instruction is executed
getindex_logical(x) = sum(abs2, x[[true, false, true, false, true]])
getindex_logical_bitvector(x) = sum(abs2, x[BitVector([false, true, true, false, true])])
getindex_logical_rows(m) = sum(abs2, m[[true, false, true, false, true], :])
getindex_logical_mask(m) = sum(abs2, m[isodd.(LinearIndices(m))])

# issue #281
view_intermediate(x) = sum(abs2, view(2 .* x, 2:4))
view_and_parent(x) = sum(view(x, 2:4)) + 3 * sum(x)
view_overlapping(x) = sum(view(x, 1:4)) * sum(view(x, 3:5))
view_nested(x) = sum(abs2, view(view(x, 1:4), 2:3))
view_logical(x) = sum(abs2, view(x, [true, false, true, false, true]))
view_dot(x) = dot(view(x, 1:3), view(x, 3:5))
view_cartesian(m) = sum(abs2, view(m, 1:2, :))
view_strided(m) = sum(abs2, view(m, :, 2:3)' * view(m, :, 1:2))

for f in (getindex_logical, getindex_logical_bitvector, view_intermediate, view_and_parent,
          view_overlapping, view_nested, view_logical, view_dot)
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in (getindex_logical_rows, getindex_logical_mask, view_cartesian, view_strided)
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

end # module
