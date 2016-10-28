##########
# Record #
##########

immutable Record{I,O,T}
    inputs::I
    outputs::O
    tape::T
    # disable default outer constructor
    Record(inputs::I, outputs::O, tape::T) = new(inputs, outputs, tape)
end

# hidden convienence constructor

_Record{I,O,T}(inputs::I, outputs::O, tape::T) = Record{I,O,T}(inputs, outputs, tape)

# public outer constructors

function Record(f, x, opts::Options = Options(x))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    yt = f(xt)
    return _Record(xt, yt, tp)
end

function Record(f, xs::Tuple, opts::Options = Options(xs))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    return _Record(xts, yt, tp)
end

function Record(f!, y, x, opts::Options = Options(y, x))
    yt, xt = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xt, x, tp)
    f!(yt, xt)
    return _Record(xt, yt, tp)
end

function Record(f!, y, xs::Tuple, opts::Options = Options(y, xs))
    yt, xts = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xts, xs, tp)
    f!(yt, xts...)
    return _Record(xt, yt, tp)
end

#################
# HessianRecord #
#################

immutable HessianRecord{J<:Record,G<:Record}
    jacobian_record::J
    gradient_record::G
end

function HessianRecord(f, x)
    opts = HessianOptions(x)
    gopts = gradient_options(opts)
    jopts = jacobian_options(opts)
    xt, jtp = jopts.state, jopts.tape
    xtt, gtp = gopts.state, gopts.tape
    track!(xt, x, jtp)
    track!(xtt, xt, gtp)
    yt = adjoint(xtt)
    ytt = f(xtt)
    return HessianRecord(_Record(xt, yt, jtp), _Record(xtt, ytt, gtp))
end

gradient_record(r::HessianRecord) = r.gradient_record
jacobian_record(r::HessianRecord) = r.jacobian_record
