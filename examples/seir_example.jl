using PolyChaosODE,
    BenchmarkTools, DifferentialEquations, Distributions, LaTeXStrings, Plots

### Test based on the example provided at https://julia.quantecon.org/continuous_time/seir_model.html ###
const _e_1 = rand(Uniform(0.0005, 0.001))
const _e_2 = rand(Uniform(0.0005, 0.001))
const _e_3 = rand(Uniform(0.0005, 0.001))

μ = 0.0
σ = sqrt(2.0) / 2

function seir!(du, u, p, x, t)
    γ = p[1] + _e_1 * x[1]
    σ = p[2] + _e_2 * x[2]
    R_0 = p[3] + _e_3 * x[3]
    du[1] = -γ * R_0 * u[1] * u[3]
    du[2] = γ * R_0 * u[1] * u[3] - σ * u[2]
    du[3] = σ * u[2] - γ * u[3]
    du[4] = γ * u[3]
end

vars = Tuple(Normal(μ, σ) for i = 1:3)
dim_ = 4
stoch_galerkin_ode = StochGalerkinODE(seir!, dim_, vars, num_polys = (2, 2, 2))

i_0 = 1E-7
e_0 = 4.0 * i_0
s_0 = 1.0 - i_0 - e_0
r_0 = 0.0

u0 = [s_0, e_0, i_0, r_0]
tspan = (0.0, 350.0)
p = [1 / 18, 1 / 5.2, 3.0]

interval_t_start = 300.0
tstep = 0.1
interval_t = interval_t_start:tstep:tspan[2]

sol = stoch_galerkin_ode(u0, tspan, p; alg = VCABM())

const VAR_INDEX = 3
const ONE_K = 1000
const TWENTY_K = 20000

sobol_indices_pce = mapreduce(
    transpose,
    vcat,
    compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX),
)[
    2:end,
    :,
]

sobol_indices_mc_1k = mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    ONE_K,
    interval_t,
    u0,
    tspan,
    p,
)

sobol_indices_mc_20k = mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    TWENTY_K,
    interval_t,
    u0,
    tspan,
    p,
)

plot(
    interval_t[2:end],
    ((x, y) -> log10.(abs.(x .- y) ./ y)).(sobol_indices_pce, sobol_indices_mc_20k);
    label = [L"err\left(\hat{S}_{1, 3}^{(PCE)}\right)" L"err\left(\hat{S}_{2, 3}^{(PCE)}\right)" L"err\left(\hat{S}_{3, 3}^{(PCE)}\right)" L"err\left(\hat{S}_{4, 3}^{(PCE)}\right)"],
    legend = :outertopright,
    size = (500, 400),
    ylims = (-5, 0),
)

xlabel!(L"t")
ylabel!(L"\log_{10}\left\vert \frac{\hat{S} - S}{S} \right\vert")

# savefig("seir_pce.png")

plot(
    interval_t[2:end],
    ((x, y) -> log10.(abs.(x .- y) ./ y)).(sobol_indices_mc_1k, sobol_indices_mc_20k),
    label = [L"err\left(\hat{S}_{1, 3}^{(MC)}\right)" L"err\left(\hat{S}_{2, 3}^{(MC)}\right)" L"err\left(\hat{S}_{3, 3}^{(MC)}\right)" L"err\left(\hat{S}_{4, 3}^{(MC)}\right)"],
    legend = :outertopright,
    size = (500, 400),
    ylims = (-5, 0),
)

xlabel!(L"t")
ylabel!(L"\log_{10}\left\vert \frac{\hat{S} - S}{S} \right\vert")

# savefig("seir_mc.png")

@btime stoch_galerkin_ode(u0, tspan, p; alg = VCABM())
@btime compute_total_order_sobol_indices(stoch_galerkin_ode, sol, interval_t, VAR_INDEX)
@btime mc_gsa_estimate_total_indices(
    stoch_galerkin_ode,
    VAR_INDEX,
    ONE_K,
    interval_t,
    u0,
    tspan,
    p,
)
