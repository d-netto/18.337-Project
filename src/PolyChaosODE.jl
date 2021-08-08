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
include("monte_carlo_gsa.jl")
include("utils.jl")

export TensorIntegrationGrid
export OrthogonalPolyBasis, LegendrePolyBasis, HermitePolyBasis
export StochGalerkinODE
export compute_expectation_and_diag_variance,
    mc_gsa_estimate_total_indices,
    plot_with_plus_minus_std,
    compute_total_order_sobol_indices

end
