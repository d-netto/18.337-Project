cd(@__DIR__)
cd("..")
include(joinpath(pwd(), "src/stochastic_galerkin_odes.jl"))
include(joinpath(pwd(), "src/utils.jl"))

### Performance vs dimensionality test ###

struct LinearODEComp{T1,T2}
    Λ::T1
    error_vec::T2
end

function ode_func!(du,u,Λ,error_vec,x,t)
    du .= (Λ .+ error_vec .* x) .* u
end

function ode_func_closure!(du,u,comp::LinearODEComp,x,t)
    @unpack Λ,error_vec = comp
    ode_func!(du,u,Λ,error_vec,x,t)
end

### benchmark params
max_dim = 5
μ = 0.0
σ = sqrt(2.0) / 2

### run benchmark
for prob_dim in 1:max_dim
    vars = Tuple(Normal(μ,σ) for i in 1:prob_dim)
    Λ = rand(Uniform(-0.50,-0.25),prob_dim)
    error_vec = rand(Uniform(0.025,0.050),prob_dim)
    ### ODE params
    u0 = 3.0 .* randn(prob_dim)
    tspan = (0.0,5.0)
    compartment = LinearODEComp(Λ,error_vec)
    ### interval for plotting
    interval_t = tspan[1]:0.01:tspan[2]
    stoch_galerkin_ode = PolyChaosODE(ode_func_closure!,prob_dim,vars)
    println("~~~~~~~~~~~~~")
    @show prob_dim
    ### compile first
    sol = stoch_galerkin_ode(u0,tspan,compartment;alg=VCABM(),alias_u0=true)
    ### after it has been compiled, time it
    @btime stoch_galerkin_ode($u0,$tspan,$compartment;alg=VCABM(),alias_u0=true)
    plot_with_pm_std(stoch_galerkin_ode,[sol(t) for t in interval_t],interval_t;display_plot=true)
end
