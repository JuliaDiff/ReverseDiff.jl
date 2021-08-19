const FUNCS_FROM_CHAINRULES = [
    :(LinearAlgebra.norm1),
]

function import_rrules()
    for func in FUNCS_FROM_CHAINRULES
        @eval @grad_from_chainrules $func
    end
end
