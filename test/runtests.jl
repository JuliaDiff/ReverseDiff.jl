using ReverseDiff
using ParallelTestRunner

# Every file under `test/` except this one and `utils.jl` is a test, named by its path,
# e.g. "api/GradientTests". Select tests via `Pkg.test("ReverseDiff"; test_args=["api/GradientTests"])`,
# exclude them with a leading `!`, and list all with `test_args=["--list"]`.
const testsuite = find_tests(@__DIR__)
delete!(testsuite, "utils")

# Workers are single-threaded by default
function test_worker(name)
    if name == "ThreadsTests"
        return addworker(; exeflags = ["--threads=4"])
    else
        return nothing
    end
end

runtests(ReverseDiff, ARGS; testsuite, test_worker)
