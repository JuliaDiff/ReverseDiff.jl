#######################
# AbstractInstruction #
#######################

abstract AbstractInstruction

typealias RawTape Vector{AbstractInstruction}

# Define some AbstractInstruction types. They all have the same structure,
# but are defined this way to make dispatch more readable (and possibly
# faster) than dispatching on a type parameter.
for InstructionType in (:ScalarInstruction, :SpecialInstruction)
    _InstructionType = Symbol(string("_", InstructionType))
    @eval begin
        immutable $(InstructionType){F,I,O,C} <: AbstractInstruction
            func::F
            input::I
            output::O
            cache::C
            # disable default outer constructor
            $(InstructionType)(func, input, output, cache) = new(func, input, output, cache)
        end

        @inline function $(_InstructionType){F,I,O,C}(func::F, input::I, output::O, cache::C)
            return $(InstructionType){F,I,O,C}(func, input, output, cache)
        end

        function $(InstructionType)(func, input, output, cache = nothing)
            return $(_InstructionType)(func, capture(input), capture(output), cache)
        end
    end
end

function record!{InstructionType}(tp::RawTape, ::Type{InstructionType}, args...)
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

##########
# passes #
##########

function forward_pass!(tape::RawTape)
    for instruction in tape
        forward_exec!(instruction)
    end
    return nothing
end

@noinline forward_exec!(instruction::ScalarInstruction) = scalar_forward_exec!(instruction)
@noinline forward_exec!(instruction::SpecialInstruction) = special_forward_exec!(instruction)

function reverse_pass!(tape::RawTape)
    for i in length(tape):-1:1
        reverse_exec!(tape[i])
    end
    return nothing
end

@noinline reverse_exec!(instruction::ScalarInstruction) = scalar_reverse_exec!(instruction)
@noinline reverse_exec!(instruction::SpecialInstruction) = special_reverse_exec!(instruction)

####################
# pass compilation #
####################

function generate_forward_pass_method{T}(::Type{T}, tape::RawTape)
    body = Expr(:block)
    for i in 1:length(tape)
        push!(body.args, :(ReverseDiff.forward_exec!(tape[$i]::$(typeof(tape[i])))))
    end
    push!(body.args, :(return nothing))
    return :(ReverseDiff.forward_pass!(tape::$T) = $body)
end

function generate_reverse_pass_method{T}(::Type{T}, tape::RawTape)
    body = Expr(:block)
    for i in length(tape):-1:1
        push!(body.args, :(ReverseDiff.reverse_exec!(tape[$i]::$(typeof(tape[i])))))
    end
    push!(body.args, :(return nothing))
    return :(ReverseDiff.reverse_pass!(tape::$T) = $body)
end

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

Base.display(tp::RawTape) = show(STDOUT, tp)

function Base.show(io::IO, tp::RawTape)
    println("$(length(tp))-element RawTape:")
    i = 1
    for instruction in tp
        print(io, "$i => ")
        show(io, instruction)
        println(io)
        i += 1
    end
end
