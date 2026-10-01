#######################
# AbstractInstruction #
#######################

abstract type AbstractInstruction end

mutable struct TapeNode
    const instruction::AbstractInstruction
    prev::Union{Nothing,TapeNode}
end

struct Finished end

# A tape is first recorded, possibly from multiple threads at once, and then replayed.
# Reading it while recording throws, and so does recording onto it after `finish!`;
# `empty!` starts recording again.
mutable struct InstructionTape
    # while recording: the most recently recorded node (nodes link backwards);
    # `Finished()` once recording has finished
    @atomic last::Union{Nothing,TapeNode,Finished}
    # recorded instructions in order, filled by `finish!`
    const instructions::Vector{AbstractInstruction}
end

InstructionTape() = InstructionTape(nothing, AbstractInstruction[])

# end recording; all tasks recording onto `tp` must have finished (e.g. via `@sync`)
function finish!(tp::InstructionTape)
    node = @atomicswap tp.last = Finished()
    if node isa Finished || !isempty(tp.instructions)
        throw(ArgumentError("tape has already finished recording"))
    end
    while node !== nothing
        push!(tp.instructions, node.instruction)
        node = node.prev
    end
    reverse!(tp.instructions)
    return tp
end

# the recorded instructions; throws unless recording has finished
function instructions(tp::InstructionTape)
    if !((@atomic tp.last) isa Finished)
        throw(ArgumentError("tape is still recording; call `ReverseDiff.finish!` first"))
    end
    return tp.instructions
end

Base.iterate(tp::InstructionTape) = iterate(instructions(tp))
Base.iterate(tp::InstructionTape, state) = iterate(instructions(tp), state)
Base.eltype(::Type{InstructionTape}) = AbstractInstruction
Base.length(tp::InstructionTape) = length(instructions(tp))

function Base.empty!(tp::InstructionTape)
    empty!(tp.instructions)
    @atomic tp.last = nothing
    return tp
end

@noinline function throw_finished()
    throw(ArgumentError("tape has finished recording; call `empty!` to record again"))
end

@inline function record!(tp::InstructionTape, ::Type{InstructionType}, args...) where {InstructionType<:AbstractInstruction}
    if tp !== NULL_TAPE
        node = TapeNode(InstructionType(args...), nothing)
        # the swap orders concurrent recordings and returns the predecessor
        prev = @atomicswap tp.last = node
        if prev isa Finished
            throw_finished()
        end
        node.prev = prev
    end
    return nothing
end

function Base.:(==)(a::AbstractInstruction, b::AbstractInstruction)
    return (a.func == b.func &&
            a.input == b.input &&
            a.output == b.output &&
            a.cache == b.cache)
end

# Ensure that the external state is "captured" so that external
# reference-breaking (e.g. destructive assignment) doesn't break
# internal instruction state. By default, `capture` is a no-op.
@inline capture(state) = state
@inline capture(state::Tuple) = map(capture, state)

# ScalarInstruction #
#-------------------#

struct ScalarInstruction{F,I,O,C} <: AbstractInstruction
    func::F
    input::I
    output::O
    cache::C
    # disable default outer constructor
    function ScalarInstruction{F,I,O,C}(func, input, output, cache) where {F,I,O,C}
        return new{F,I,O,C}(func, input, output, cache)
    end
end

@inline function _ScalarInstruction(func::F, input::I, output::O, cache::C) where {F,I,O,C}
    return ScalarInstruction{F,I,O,C}(func, input, output, cache)
end

function ScalarInstruction(func, input, output, cache = nothing)
    return _ScalarInstruction(func, capture(input), capture(output), cache)
end

# SpecialInstruction #
#--------------------#

struct SpecialInstruction{F,I,O,C} <: AbstractInstruction
    func::F
    input::I
    output::O
    cache::C
    # disable default outer constructor
    function SpecialInstruction{F,I,O,C}(func, input, output, cache) where {F,I,O,C}
        return new{F,I,O,C}(func, input, output, cache)
    end
end

@inline function _SpecialInstruction(func::F, input::I, output::O, cache::C) where {F,I,O,C}
    return SpecialInstruction{F,I,O,C}(func, input, output, cache)
end

function SpecialInstruction(func, input, output, cache = nothing)
    return _SpecialInstruction(func, capture(input), capture(output), cache)
end

##########
# passes #
##########

function forward_pass!(tape::InstructionTape)
    for instruction in instructions(tape)
        forward_exec!(instruction)
    end
    return nothing
end

@noinline forward_exec!(instruction::ScalarInstruction) = scalar_forward_exec!(instruction)
@noinline forward_exec!(instruction::SpecialInstruction) = special_forward_exec!(instruction)

function reverse_pass!(tape::InstructionTape)
    for instruction in Iterators.reverse(instructions(tape))
        reverse_exec!(instruction)
    end
    return nothing
end

@noinline reverse_exec!(instruction::ScalarInstruction) = scalar_reverse_exec!(instruction)
@noinline reverse_exec!(instruction::SpecialInstruction) = special_reverse_exec!(instruction)

###################
# Pretty Printing #
###################

# extra spaces here accomodates padding in show(::IO, ::AbstractInstruction)
compactrepr(x::Tuple) = "("*join(map(compactrepr, x), ",\n           ")*")"
compactrepr(x::AbstractArray) = length(x) < 5 ? match(r"\[.*?\]", repr(x)).match : summary(x)
compactrepr(x) = repr(x)

function Base.show(io::IO, instruction::AbstractInstruction, pad = "")
    name = isa(instruction, ScalarInstruction) ? "ScalarInstruction" : "SpecialInstruction"
    println(io, pad, "$(name)($(instruction.func)):")
    println(io, pad, "  input:  ", compactrepr(instruction.input))
    println(io, pad, "  output: ", compactrepr(instruction.output))
    print(io,   pad, "  cache:  ", compactrepr(instruction.cache))
end

function Base.show(io::IO, tp::InstructionTape)
    println(io, length(tp), "-element InstructionTape:")
    i = 1
    for instruction in tp
        print(io, "$i => ")
        show(io, instruction)
        println(io)
        i += 1
    end
end
