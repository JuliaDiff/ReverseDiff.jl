module TapeTests

using ReverseDiff, Base.Test
using ReverseDiff: SpecialInstruction, ScalarInstruction, NULL_TAPE

include(joinpath(dirname(@__FILE__), "utils.jl"))

println("testing RawTape/AbstractInstructions...")
tic()

############################################################################################

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

    tp = RawTape()
    ReverseDiff.record!(tp, Instr, +, (x, y, k), z, c)
    @test tp[1] == instr
    @test tp[1].func === +
    @test tp[1].input[1] !== x
    @test tp[1].input[2] !== y
    @test tp[1].input[3] === k
    @test tp[1].input[1] == x
    @test tp[1].input[2] == y
    @test tp[1].output === z
    @test tp[1].cache === c

    ReverseDiff.record!(NULL_TAPE, Instr, +, (x, y, k), z, c)
    @test isempty(NULL_TAPE)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
