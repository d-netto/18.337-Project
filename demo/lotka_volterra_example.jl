using PolyChaosODE, BenchmarkTools, DifferentialEquations, Distributions

### Lotka-Volterra test ###

const e_ = 0.050

function lotka_volterra!(du, u, p, x, t)
    du[1] = (p[1] + e_ * x[1]) * u[1] + (-p[2] - e_ * x[2]) * u[1] * u[2]
    du[2] = (-p[3] - e_ * x[3]) * u[2] + (p[4] + e_ * x[4]) * u[1] * u[2]
end

vars = Tuple(Uniform(-1.0, 1.0) for i = 1:4)
dim_ = 2
stoch_galerkin_ode = StochGalerkinODE(lotka_volterra!, dim_, vars)

u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
tstep = 0.1
interval_t = tspan[1]:tstep:tspan[2]
p = [1.5, 1.0, 3.0, 1.0]

sol = stoch_galerkin_ode(u0, tspan, p; alg = VCABM())

const VAR_INDEX = 3
const NUMBER_SAMPLES = 10000

sobol_indices_ode = mapreduce(
    transpose,
    vcat,
    compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX),
)[
    2:end,
    :,
]

sobol_indices_mc = mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    NUMBER_SAMPLES,
    interval_t,
    u0,
    tspan,
    p,
)

@btime stoch_galerkin_ode(u0, tspan, p; alg = VCABM())
@btime compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX)
@btime mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    NUMBER_SAMPLES,
    interval_t,
    u0,
    tspan,
    p,
)

@show sobol_indices_ode .- sobol_indices_mc
