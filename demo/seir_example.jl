using PolyChaosODE, DifferentialEquations

### Test based on the example provided at https://julia.quantecon.org/continuous_time/seir_model.html ###
const _e = 0.050
μ = 0.0
σ = sqrt(2.0) / 2

function seir!(du, u, p, x, t)
    γ = p[1] * (1 + _e * x[1])
    σ = p[2] * (1 + _e * x[2])
    R_0 = p[3] * (1 + _e * x[3])
    du[1] = -γ * R_0 * u[1] * u[3]
    du[2] = γ * R_0 * u[1] * u[3] - σ * u[2]
    du[3] = σ * u[2] - γ * u[3]
    du[4] = γ * u[3]
end

vars = Tuple(Normal(μ, σ) for i = 1:3)
num_polys = Tuple(5 for i = 1:3)
dim_ = 4
stoch_galerkin_ode = StochGalerkinODE(seir!, dim_, vars; num_polys = num_polys)

i_0 = 1E-7
e_0 = 4.0 * i_0
s_0 = 1.0 - i_0 - e_0
r_0 = 0.0

u0 = [s_0, e_0, i_0, r_0]
tspan = (0.0, 350.0)
tstep = 0.001
interval_t = tspan[1]:tstep:tspan[2]
p = [1 / 18, 1 / 5.2, 3.0]

sol = stoch_galerkin_ode(u0, tspan, p; alg = VCABM())
plot_with_plus_minus_std(stoch_galerkin_ode, interval_t, sol)
