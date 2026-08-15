#=
 Backend strategy types.

 Backend properties are encoded in types so dispatch and inference can resolve
 precision and arc-mixing behavior at compile time.
=#

abstract type EMDBackend end

struct NetworkSimplexBackend{T<:AbstractFloat,ArcMixing} <: EMDBackend end

struct SinkhornBackend <: EMDBackend end

# Canonical singleton strategies used by the public backend keywords.
const NS64 = NetworkSimplexBackend{Float64,false}()
const OT64 = NetworkSimplexBackend{Float64,true}()
const NS32 = NetworkSimplexBackend{Float32,false}()
const OT32 = NetworkSimplexBackend{Float32,true}()
const Sinkhorn = SinkhornBackend()

const AVAILABLE_BACKENDS = (:ns64, :ot64, :ns32, :ot32, :sinkhorn)

_as_backend(backend::EMDBackend) = backend

function _as_backend(backend::Symbol)
    if backend === :ns64
        return NS64
    elseif backend === :ot64
        return OT64
    elseif backend === :ns32
        return NS32
    elseif backend === :ot32
        return OT32
    elseif backend === :sinkhorn
        return Sinkhorn
    else
        throw(ArgumentError(
            "unknown backend :$backend; available backends: $(AVAILABLE_BACKENDS)"
        ))
    end
end

_backend_eltype(::NetworkSimplexBackend{T}) where {T} = T
_arc_mixing(::NetworkSimplexBackend{T,A}) where {T,A} = A

_backend_symbol(::typeof(NS64)) = :ns64
_backend_symbol(::typeof(OT64)) = :ot64
_backend_symbol(::typeof(NS32)) = :ns32
_backend_symbol(::typeof(OT32)) = :ot32
_backend_symbol(::SinkhornBackend) = :sinkhorn
