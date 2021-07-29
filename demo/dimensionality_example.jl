using PolyChaosODE, DifferentialEquations, Parameters, Distributions, BenchmarkTools

### Performance vs dimensionality example ###

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

max_dim = 5
μ = 0.0
σ = sqrt(2.0) / 2

# run benchmark
for prob_dim in 1:max_dim
    vars = Tuple(Normal(μ,σ) for i in 1:prob_dim)
    Λ = rand(Uniform(-0.50,-0.25),prob_dim)
    error_vec = rand(Uniform(0.025,0.050),prob_dim)
    # ODE params
    u0 = 10.0 .* randn(prob_dim)
    tspan = (0.0,5.0)
    compartment = LinearODEComp(Λ,error_vec)
    # interval for plotting
    step_t = 0.01
    interval_t = tspan[1]:step_t:tspan[2]
    stoch_galerkin_ode = StochGalerkinODE(ode_func_closure!,prob_dim,vars)
    println("~~~~~~~~~~~~~")
    @show prob_dim
    # compile first
    sol = stoch_galerkin_ode(u0,tspan,compartment)
    # after it has been compiled, time it
    @btime $stoch_galerkin_ode($u0,$tspan,$compartment;alg=VCABM())
    plot_with_plus_minus_std(stoch_galerkin_ode,interval_t,sol;display_plot=true)
end
