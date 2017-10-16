#######################
# AbstractInstruction #
#######################

@compat abstract type AbstractInstruction end

@compat const InstructionTape = Vector{AbstractInstruction}

function record!{InstructionType}(tp::InstructionTape, ::Type{InstructionType}, args...)
    tp !== NULL_TAPE && push!(tp, InstructionType(args...))
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

@compat immutable ScalarInstruction{F,I,O,C} <: AbstractInstruction
    func::F
    input::I
    output::O
    cache::C
    # disable default outer constructor
    function (::Type{ScalarInstruction{F,I,O,C}}){F,I,O,C}(func, input, output, cache)
        return new{F,I,O,C}(func, input, output, cache)
    end
end

@inline function _ScalarInstruction{F,I,O,C}(func::F, input::I, output::O, cache::C)
    return ScalarInstruction{F,I,O,C}(func, input, output, cache)
end

function ScalarInstruction(func, input, output, cache = nothing)
    return _ScalarInstruction(func, capture(input), capture(output), cache)
end

# SpecialInstruction #
#--------------------#

@compat immutable SpecialInstruction{F,I,O,C} <: AbstractInstruction
    func::F
    input::I
    output::O
    cache::C
    # disable default outer constructor
    function (::Type{SpecialInstruction{F,I,O,C}}){F,I,O,C}(func, input, output, cache)
        return new{F,I,O,C}(func, input, output, cache)
    end
end

@inline function _SpecialInstruction{F,I,O,C}(func::F, input::I, output::O, cache::C)
    return SpecialInstruction{F,I,O,C}(func, input, output, cache)
end

function SpecialInstruction(func, input, output, cache = nothing)
    return _SpecialInstruction(func, capture(input), capture(output), cache)
end

##########
# passes #
##########

function forward_pass!(tape::InstructionTape)
    for instruction in tape
        forward_exec!(instruction)
    end
    return nothing
end

@noinline forward_exec!(instruction::ScalarInstruction) = scalar_forward_exec!(instruction)
@noinline forward_exec!(instruction::SpecialInstruction) = special_forward_exec!(instruction)

function reverse_pass!(tape::InstructionTape)
    for i in length(tape):-1:1
        reverse_exec!(tape[i])
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
    println("$(length(tp))-element InstructionTape:")
    i = 1
    for instruction in tp
        print(io, "$i => ")
        show(io, instruction)
        println(io)
        i += 1
    end
end
