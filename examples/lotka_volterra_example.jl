using PolyChaosODE, BenchmarkTools, LaTeXStrings, DifferentialEquations, Distributions

### Lotka-Volterra example ###

const e_ = 0.050
const μ = 0.0
const σ = sqrt(2.0) / 2

function lotka_volterra!(du, u, p, x, t)
    du[1] = (p[1] + e_ * x[1]) * u[1] + (-p[2] - e_ * x[2]) * u[1] * u[2]
    du[2] = (-p[3] - e_ * x[3]) * u[2] + (p[4] + e_ * x[4]) * u[1] * u[2]
end

vars = Tuple(Normal(μ, σ) for i = 1:4)
dim_ = 2
stoch_galerkin_ode = StochGalerkinODE(lotka_volterra!, dim_, vars)

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
tstep = 0.1
interval_t = tspan[1]:tstep:tspan[2]
p = [1.5, 1.0, 3.0, 1.0]

sol = stoch_galerkin_ode(u0, tspan, p; alg = VCABM())

const VAR_INDEX = 4
const TEN_K = 10000
const HUNDRED_K = 100000

sobol_indices_pce = mapreduce(
    transpose,
    vcat,
    compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX),
)[
    2:end,
    :,
]

sobol_indices_mc_10k = mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    TEN_K,
    interval_t,
    u0,
    tspan,
    p,
)

sobol_indices_mc_100k = mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    HUNDRED_K,
    interval_t,
    u0,
    tspan,
    p,
)

plot(
    interval_t[2:end],
    ((x, y) -> log10.(abs.(x .- y) ./ y)).(sobol_indices_pce, sobol_indices_mc_100k);
    label = [L"err\left(\hat{S}_{1, 4}^{(PCE)}\right)" L"err\left(\hat{S}_{2, 4}^{(PCE)}\right)"],
    size = (500, 400),
    ylims = (-5, -1),
)

xlabel!(L"t")
ylabel!(L"\log_{10}\left\vert \frac{\hat{S} - S}{S} \right\vert")

plot(
    interval_t[2:end],
    ((x, y) -> log10.(abs.(x .- y) ./ y)).(sobol_indices_mc_10k, sobol_indices_mc_100k),
    label = [L"err\left(\hat{S}_{1, 4}^{(MC)}\right)" L"err\left(\hat{S}_{2, 4}^{(MC)}\right)"],
    size = (500, 400),
    ylims = (-5, -1),
)

xlabel!(L"t")
ylabel!(L"\log_{10}\left\vert \frac{\hat{S} - S}{S} \right\vert")

@btime stoch_galerkin_ode(u0, tspan, p; alg = VCABM())
@btime compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX)
@btime mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    TEN_K,
    interval_t,
    u0,
    tspan,
    p,
)
