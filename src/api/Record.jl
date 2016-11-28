##################
# AbstractRecord #
##################

abstract AbstractRecord

Base.show(io::IO, rec::AbstractRecord) = print(io, typeof(rec).name, "(", rec.func, ")")

# Define a few different R<:AbstractRecord types. All these types share the same structure,
# but feature different constructors and dispatch restrictions in downstream code.
for R in (:GradientRecord, :JacobianRecord, :HessianRecord)
    _R = Symbol(string("_", R))
    @eval begin
        immutable $(R){F,I,O} <: AbstractRecord
            func::F
            input::I
            output::O
            tape::Tape
            # disable default outer constructor
            $(R)(func, input, output, tape) = new(func, input, output, tape)
        end

        # "private" convienence constructor
        $(_R){F,I,O}(func::F, input::I, output::O, tape::Tape) = $(R){F,I,O}(func, input, output, tape)
    end
end

forward_pass!(rec::AbstractRecord) = forward_pass!(rec.tape)

reverse_pass!(rec::AbstractRecord) = reverse_pass!(rec.tape)

function seeded_forward_pass!(rec::AbstractRecord, input)
    value!(rec.input, input)
    forward_pass!(rec)
    return nothing
end

function seeded_reverse_pass!(result, rec::AbstractRecord)
    seeded_reverse_pass!(result, rec.output, rec.input, rec)
    return result
end

############
# Compiled #
############

immutable Compiled{R<:AbstractRecord,F,I,O,FP,RP} <: AbstractRecord
    record_type::Type{R}
    func::F
    input::I
    output::O
    forward_pass!::FP
    reverse_pass!::RP
end

typealias CompiledGradient{R<:GradientRecord,F,I,O,FP,RP} Compiled{R,F,I,O,FP,RP}
typealias CompiledJacobian{R<:JacobianRecord,F,I,O,FP,RP} Compiled{R,F,I,O,FP,RP}
typealias CompiledHessian{R<:HessianRecord,F,I,O,FP,RP}   Compiled{R,F,I,O,FP,RP}

function compile(rec::AbstractRecord)
    return Compiled(typeof(rec), rec.func, rec.input, rec.output,
                    eval(ReverseDiff, :(() -> $(generate_forward_code(rec.tape)))),
                    eval(ReverseDiff, :(() -> $(generate_reverse_code(rec.tape)))))
end

forward_pass!(rec::Compiled) = rec.forward_pass!()

reverse_pass!(rec::Compiled) = rec.reverse_pass!()

##################
# GradientRecord #
##################

function GradientRecord(f, input, cfg::GradientConfig = GradientConfig(input))
    track!(cfg.input, input)
    tracked_ouput = f(cfg.input)
    return _GradientRecord(f, cfg.input, tracked_ouput, cfg.tape)
end

function GradientRecord(f, input::Tuple, cfg::GradientConfig = GradientConfig(input))
    for i in eachindex(cfg.input)
        track!(cfg.input[i], input[i])
    end
    tracked_output = f(cfg.input...)
    return _GradientRecord(f, cfg.input, tracked_output, cfg.tape)
end

##################
# JacobianRecord #
##################

function JacobianRecord(f, input, cfg::JacobianConfig = JacobianConfig(input))
    track!(cfg.input, input)
    tracked_ouput = f(cfg.input)
    return _JacobianRecord(f, cfg.input, tracked_ouput, cfg.tape)
end

function JacobianRecord(f, input::Tuple, cfg::JacobianConfig = JacobianConfig(input))
    for i in eachindex(cfg.input)
        track!(cfg.input[i], input[i])
    end
    tracked_output = f(cfg.input...)
    return _JacobianRecord(f, cfg.input, tracked_output, cfg.tape)
end

function JacobianRecord(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
    track!(cfg.output, output, cfg.tape)
    track!(cfg.input, input)
    f!(cfg.output, cfg.input)
    return _JacobianRecord(f!, cfg.input, cfg.output, cfg.tape)
end

function JacobianRecord(f!, output, input::Tuple, cfg::JacobianConfig = JacobianConfig(output, input))
    track!(cfg.output, output, cfg.tape)
    for i in eachindex(input)
        track!(cfg.input[i], input[i])
    end
    f!(cfg.output, cfg.input...)
    return _JacobianRecord(f!, cfg.input, cfg.output, cfg.tape)
end

#################
# HessianRecord #
#################

function HessianRecord(f, input, cfg::HessianConfig = HessianConfig(input))
    gcfg = cfg.gradient_config
    jcfg = cfg.jacobian_config
    rec = _HessianRecord(f, jcfg.input, similar(deriv(gcfg.input)), jcfg.tape)
    track!(rec.input, input)
    grec = GradientRecord(f, rec.input, gcfg)
    seeded_reverse_pass!(rec.output, grec.output, grec.input, grec.tape)
    return rec
end
