module TapeTests

using ReverseDiff, Test
using ReverseDiff: SpecialInstruction, ScalarInstruction, NULL_TAPE

include(joinpath(dirname(@__FILE__), "utils.jl"))

for Instr in (SpecialInstruction, ScalarInstruction)
    x, y, k = rand(3), rand(2, 1), rand()
    z = rand()
    c = rand(1)
    instr = Instr(+, (x, y, k), z, c)
    @test isa(instr, Instr{typeof(+)})
    @test instr.func === +
    @test instr.input[1] !== x
    @test instr.input[2] !== y
    @test instr.input[3] === k
    @test instr.input[1] == x
    @test instr.input[2] == y
    @test instr.output === z
    @test instr.cache === c

    tp = InstructionTape()
    ReverseDiff.record!(tp, Instr, +, (x, y, k), z, c)
    @test_throws ArgumentError length(tp)
    @test_throws ArgumentError first(tp)
    ReverseDiff.finish!(tp)
    @test length(tp) == 1
    recorded = first(tp)
    @test recorded == instr
    @test recorded.func === +
    @test recorded.input[1] !== x
    @test recorded.input[2] !== y
    @test recorded.input[3] === k
    @test recorded.input[1] == x
    @test recorded.input[2] == y
    @test recorded.output === z
    @test recorded.cache === c
    @test startswith(string(tp), "1-element InstructionTape:")
    @test_throws ArgumentError ReverseDiff.record!(tp, Instr, +, (x, y, k), z, c)
    @test_throws ArgumentError ReverseDiff.finish!(tp)

    empty!(tp)
    ReverseDiff.record!(tp, Instr, +, (x, y, k), z, c)
    ReverseDiff.record!(tp, Instr, +, (x, y, k), z, c)
    ReverseDiff.finish!(tp)
    @test length(tp) == 2

    ReverseDiff.record!(NULL_TAPE, Instr, +, (x, y, k), z, c)
    @test isempty(NULL_TAPE)
end

end # module
