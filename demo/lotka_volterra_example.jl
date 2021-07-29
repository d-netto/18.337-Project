using PolyChaosODE, Distributions, DifferentialEquations

### Lotka-Volterra test ###

const e_ = 0.050

function lotka_volterra!(du,u,p,x,t)
  du[1] = (p[1] + e_ * x[1]) * u[1] + (-p[2] - e_ * x[2]) * u[1] * u[2]
  du[2] = (-p[3] - e_ * x[3]) * u[2] + (p[4] + e_ * x[4]) * u[1] * u[2]
end

vars = Tuple(Uniform(-1.0,1.0) for i in 1:4)
dim_ = 2
stoch_galerkin_ode = StochGalerkinODE(lotka_volterra!,dim_,vars)

u0 = [1.0,1.0]
tspan = (0.0,10.0)
tstep = 0.1
interval_t = tspan[1]:tstep:tspan[2]
p = [1.5,1.0,3.0,1.0]

sol = stoch_galerkin_ode(u0,tspan,p;alg=VCABM())
plot_with_plus_minus_std(stoch_galerkin_ode,interval_t,sol; display_plot=true)

sobol_indices = compute_total_order_sobol_indices(stoch_galerkin_ode,sol,interval_t,4)
@show sobol_indices
