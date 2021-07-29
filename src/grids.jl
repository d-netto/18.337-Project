struct TensorIntegrationGrid{T1,T2}
    quads::T1
    quad_indices::T2

    function TensorIntegrationGrid(dim_,number_polys::Tuple,vars;wType=Float64)
        @assert length(number_polys) == length(vars)

        quads_temp = []

        for i in Base.OneTo(length(vars))
            @assert number_polys[i] > 0
            quad_list = []
            var = vars[i]
            if var isa Uniform
                @assert var.a ≈ -1.0 && var.b ≈ 1.0
                push!(quads_temp,GaussLegendreQuad(number_polys[i];wType))
            elseif var isa Normal
                # Using physicist's Hermite polynomial
                @assert var.μ ≈ 0.0 && var.σ ≈ sqrt(2.0) / 2
                push!(quads_temp,GaussHermiteQuad(number_polys[i];wType))
            else
                throw("Not implemented")
            end
        end

        quads = Tuple(quads_temp)
        quad_indices = CartesianIndices(Tuple(number_polys))

        return new{typeof(quads),typeof(quad_indices)}(quads,quad_indices)
    end

end
