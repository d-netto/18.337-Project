abstract type OneDimQuadrature end

struct GaussLegendreQuad{T1,T2,T3} <: OneDimQuadrature
    n::T1
    nodes::T2
    weights::T3

    function GaussLegendreQuad(n;wType=Float64)
        @assert n > 0
        quad = gausslegendre(n)
        nodes = wType.(quad[1])
        weights = wType.(quad[2])
        return new{typeof(n),typeof(nodes),typeof(weights)}(n,nodes,weights)
    end
end

struct GaussHermiteQuad{T1,T2,T3} <: OneDimQuadrature
    n::T1
    nodes::T2
    weights::T3

    function GaussHermiteQuad(n;wType=Float64)
        @assert n > 0
        quad = gausshermite(n)
        nodes = wType.(quad[1])
        weights = wType.(quad[2])
        return new{typeof(n),typeof(nodes),typeof(weights)}(n,nodes,weights)
    end
end
