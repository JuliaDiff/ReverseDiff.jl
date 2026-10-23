include("elementwise_utils.jl")

for (M, fsym, arity) in DiffRules.diffrules(; filter_modules = nothing)
    # ensure that all rules can be tested
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), fsym))
        error("$M.$fsym is not available")
    end
    (M, fsym) in ReverseDiff.SKIPPED_DIFFRULES && continue
    if arity == 2
        in(fsym, SKIPPED_BINARY_SCALAR_TESTS) && continue
        f = eval(:($M.$fsym))
        test_println("forward-mode binary scalar functions", f)
        test_map(f, f, x, y, tp)
        test_map(f, f, a, b, tp)
        test_broadcast(f, f, x, y, tp)
        test_broadcast(f, f, a, b, tp)
        test_broadcast(f, f, x, a, tp)
        test_broadcast(f, f, a, x, tp)
        test_broadcast(f, f, n, x, tp)
        test_broadcast(f, f, x, n, tp)
        test_broadcast(f, f, n, a, tp)
        test_broadcast(f, f, a, n, tp)
    end
end
