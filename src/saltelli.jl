using LinearAlgebra

function saltelli_estimate(s::StochGalerkinODE, var_index, interval_t, num_samples, u0, tspan, p; kwargs...)
    @unpack ode_func!, dim_, vars = s
    A = zeros(num_samples, length(vars))
    B = zeros(num_samples, length(vars))
    for j in Base.OneTo(length(vars))
        for i in Base.OneTo(num_samples)
            A[i, j] = rand(vars[j])
            B[i, j] = rand(vars[j])
        end
    end
    C = deepcopy(B)
    C[:, var_index] .= A[:, var_index]
    Y_A = zeros(num_samples, length(interval_t), dim_)
    Y_B = zeros(num_samples, length(interval_t), dim_)
    Y_C = zeros(num_samples, length(interval_t), dim_)
    for i in Base.OneTo(num_samples)
        x_A = A[i, :]
        x_B = B[i, :]
        x_C = C[i, :]
        prob_A = ODEProblem((du, u, p, t) -> ode_func!(du, u, p, x_A, t), u0, tspan, p)
        prob_B = ODEProblem((du, u, p, t) -> ode_func!(du, u, p, x_B, t), u0, tspan, p)
        prob_C = ODEProblem((du, u, p, t) -> ode_func!(du, u, p, x_C, t), u0, tspan, p)
        sol_A = solve(prob_A; saveat = interval_t, kwargs...)
        sol_B = solve(prob_B; saveat = interval_t, kwargs...)
        sol_C = solve(prob_C; saveat = interval_t, kwargs...)
        for j in Base.OneTo(length(interval_t))
            Y_A[i, j, :] .= sol_A(interval_t[j])
            Y_B[i, j, :] .= sol_B(interval_t[j])
            Y_C[i, j, :] .= sol_C(interval_t[j])
        end
    end
    sobol_indices = zeros(length(interval_t), dim_)
    for t_index in Base.OneTo(length(interval_t))
        for dim_index in Base.OneTo(dim_)
            Y_A_ = Y_A[:, t_index, dim_index]
            Y_B_ = Y_B[:, t_index, dim_index]
            Y_C_ = Y_C[:, t_index, dim_index]
            f_0 = sum(Y_A_) / num_samples
            numerator_ = (dot(Y_B_, Y_C_) / num_samples) - f_0^2
            denominator_ = (dot(Y_A_, Y_A_) / num_samples) - f_0^2
            sobol_indices[t_index, dim_index] = 1 - numerator_ / denominator_
        end
    end
    return sobol_indices[2:end, :]
end
