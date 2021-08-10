using PolyChaosODE, BenchmarkTools, DifferentialEquations, Distributions, PolyChaos, Plots

### This package ###

x0 = 2.0
μ, σ = -0.5, 0.05
tend = 10.0
L = 6

function exp_ode!(du, u, p, x, t)
    du[1] = (p[1] + p[2] * x[1]) * u[1]
end

prob_dim = 1
vars = (Normal(0.0, sqrt(2) / 2),)
num_polys = (L + 1,)
stoch_galerkin_ode = StochGalerkinODE(exp_ode!, prob_dim, vars; num_polys = num_polys)

x0_ = [x0]
tspan = (0, tend)
# sqrt(2) factor in the second term because we are using the Physicist's Hermite polynomials instead of the Probabilist's
p = [-0.5, 0.05 * sqrt(2)]
sol = stoch_galerkin_ode(x0_, tspan, p; alg = VCABM())

t_step = 0.01
interval_t = 0.0:t_step:tend
std_this =
    (
        x -> sqrt.(x)
    ).(
        compute_expectation_and_diag_variance(
            stoch_galerkin_ode,
            interval_t,
            [sol(t) for t in interval_t],
        )[2],
    )

@btime stoch_galerkin_ode($x0_, $tspan, $p; alg = VCABM())
@btime (
    x -> sqrt.(x)
).(
    compute_expectation_and_diag_variance(
        stoch_galerkin_ode,
        interval_t,
        [sol(t) for t in interval_t],
    )[2],
)

### PolyChaos (based on https://timueh.github.io/PolyChaos.jl/stable/random_ode/)###

Nrec = 40
opq = GaussOrthoPoly(L; Nrec = Nrec, addQuadrature = true)

a = [convert2affinePCE(μ, σ, opq); zeros(Float64, L - 1)]
xinit = [x0; zeros(Float64, L)]

const t2_ = Tensor(2, opq);
const t3_ = Tensor(3, opq);

function ODEgalerkin(du, u, p, t)
    du[:] = [
        sum(p[j+1] * u[k+1] * t3_.get([j, k, m]) / t2_.get([m, m]) for j = 0:L for k = 0:L) for m = 0:L
    ]
end

probgalerkin = ODEProblem(ODEgalerkin, xinit, tspan, a)
solgalerkin = solve(probgalerkin; alg = VCABM())

t, x = solgalerkin.t, solgalerkin.u;
mean_poly_chaos = [mean(x_, opq) for x_ in x]
std_poly_chaos = [std(x_, opq) for x_ in x]

@btime solve(probgalerkin; alg = VCABM())
@btime [mean(x_, opq) for x_ in x]
@btime [std(x_, opq) for x_ in x]

plot(t, mean_pce .- std_pce)
plot!(t, mean_pce)
plot!(t, mean_pce .+ std_pce)
