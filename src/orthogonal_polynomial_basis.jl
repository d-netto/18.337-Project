abstract type OrthogonalPolyBasis end

struct LegendrePolyBasis{T,A} <: OrthogonalPolyBasis
    n::T
    weights::A

    function LegendrePolyBasis(n; wType = Float64)
        weights_helper = zeros(wType, n, n)
        weights_helper[1, 1] = one(wType)
        weights_helper[2, 2] = one(wType)
        for i = 3:n
            real_index = i - 1
            two_below = ((real_index - 1) / real_index) .* weights_helper[i-2, :]
            one_below =
                ((2 * real_index - 1) / real_index) .*
                [[zero(wType)]; weights_helper[i-1, :][1:n-1]]
            weights_helper[i, :] .= one_below .- two_below
        end
        weights = [Tuple(weights_helper[i, j] for j = 1:i) for i = 1:n]
        return new{typeof(n),typeof(weights)}(n, weights)
    end
end

function (b::LegendrePolyBasis)(x, i::Int64)
    @unpack weights = b
    @evalpoly(x, weights[i+1]...)
end

struct HermitePolyBasis{T,A} <: OrthogonalPolyBasis
    n::T
    weights::A

    function HermitePolyBasis(n; wType = Int64)
        weights_helper = zeros(wType, n, n)
        weights_helper[1, 1] = one(wType)
        weights_helper[2, 2] = 2
        for i = 3:n
            real_index = i - 1
            weights_helper[i, 1] = -weights_helper[i-1, 2]
            for j = 2:i-1
                weights_helper[i, j] =
                    2 * weights_helper[i-1, j-1] - j * weights_helper[i-1, j+1]
            end
            weights_helper[i, i] = 2 * weights_helper[i-1, i-1]
        end
        weights = [Tuple(weights_helper[i, j] for j = 1:i) for i = 1:n]
        return new{typeof(n),typeof(weights)}(n, weights)
    end
end

function (b::HermitePolyBasis)(x, i::Int64)
    @unpack weights = b
    @evalpoly(x, weights[i+1]...)
end
