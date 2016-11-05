########################
# AbstractRecord Types #
########################

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
