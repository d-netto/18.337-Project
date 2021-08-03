__precompile__()

module PolyChaosODE

using BenchmarkTools,
    DifferentialEquations,
    Distributions,
    FastGaussQuadrature,
    LaTeXStrings,
    LinearAlgebra,
    Parameters,
    PolyChaos,
    Plots,
    Statistics

include("grids.jl")
include("orthogonal_polynomial_basis.jl")
include("quadrature.jl")
include("stochastic_galerkin_ode.jl")
include("utils.jl")

export TensorIntegrationGrid
export OrthogonalPolyBasis, LegendrePolyBasis, HermitePolyBasis
export StochGalerkinODE
export compute_expectation_and_diag_variance, plot_with_plus_minus_std, compute_total_order_sobol_indices

end
