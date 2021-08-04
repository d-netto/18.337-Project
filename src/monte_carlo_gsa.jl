using LinearAlgebra

function mc_gsa_estimate_total_indices(
    s::StochGalerkinODE,
    var_index,
    num_samples,
    interval_t,
    u0,
    tspan,
    p;
    kwargs...,
)
    @unpack ode_func!, dim_, vars = s
    A = zeros(num_samples, length(vars))
    B = zeros(num_samples, length(vars))
    for j in Base.OneTo(length(vars))
        for i in Base.OneTo(num_samples)
            A[i, j] = rand(vars[j])
            B[i, j] = rand(vars[j])
        end
    end
    C = deepcopy(A)
    C[:, var_index] .= B[:, var_index]
    Y_A = zeros(num_samples, length(interval_t), dim_)
    Y_C = zeros(num_samples, length(interval_t), dim_)
    for i in Base.OneTo(num_samples)
        sol_A = solve(
            ODEProblem((du, u, p, t) -> ode_func!(du, u, p, A[i, :], t), u0, tspan, p),
            kwargs...,
        )
        sol_B = solve(
            ODEProblem((du, u, p, t) -> ode_func!(du, u, p, B[i, :], t), u0, tspan, p),
            kwargs...,
        )
        sol_C = solve(
            ODEProblem((du, u, p, t) -> ode_func!(du, u, p, C[i, :], t), u0, tspan, p),
            kwargs...,
        )
        for j in Base.OneTo(length(interval_t))
            Y_A[i, j, :] .= sol_A(interval_t[j])
            Y_C[i, j, :] .= sol_C(interval_t[j])
        end
    end
    sobol_indices = zeros(length(interval_t), dim_)
    for t_index in Base.OneTo(length(interval_t))
        for dim_index in Base.OneTo(dim_)
            Y_A_ = @view Y_A[:, t_index, dim_index]
            Y_C_ = @view Y_C[:, t_index, dim_index]
            expectation_not_xi = dot(Y_A_ - Y_C_, Y_A_ - Y_C_) / (2 * num_samples)
            variance = dot(Y_A_, Y_A_) / num_samples - (sum(Y_A_) / num_samples)^2
            sobol_indices[t_index, dim_index] = expectation_not_xi / variance
        end
    end
    return sobol_indices[2:end, :]
end
