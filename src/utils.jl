const VAR_THRESHOLD = -10^-10

function compute_cov_matrix!(
    s::StochGalerkinODE,
    c::StochGalerkinODECache,
    matrix_cache,
    vector_transpose_cache,
    dim_,
    variance_map,
    u,
)
    @unpack indices_polys = s
    @unpack y_hat_for_grid, integral_est = c
    cov_matrix = zeros(eltype(u), dim_, dim_)
    for index_ in indices_polys
        transpose!(vector_transpose_cache, (@view u[:, index_]))
        mul!(matrix_cache, (@view u[:, index_]), vector_transpose_cache)
        cov_matrix .+= matrix_cache .* variance_map[index_]
    end
    return cov_matrix
end

function compute_expectation_and_diag_variance(s::StochGalerkinODE, interval_t, sol_u)
    c = StochGalerkinODECache(s, first(sol_u))
    @unpack dim_, num_polys, indices_polys, poly_basis, grid, normalization_map = s
    @unpack quads, quad_indices = grid
    uType = eltype(eltype(sol_u))
    # stores expectation for each polynomial with a given multi-index
    expectation_map = zeros(uType, num_polys)
    # stores variance for each polynomial with a given multi-index
    variance_map = zeros(uType, num_polys)
    # normalization constant (quadrature weights don't include constant factors from pdf's)
    norm_const = mapreduce(
        index_ -> mapreduce(i -> quads[i].weights[index_[i]], *, 1:length(index_)),
        +,
        quad_indices,
    )
    for index in indices_polys
        expectation_map[index] =
            mapreduce(
                i -> dot(
                    quads[i].weights,
                    (x -> poly_basis[i](x, index[i] - 1)).(quads[i].nodes),
                ),
                *,
                Base.OneTo(length(index)),
            ) / norm_const
        variance = normalization_map[index] / norm_const - expectation_map[index]^2
        # Hack for handling very small (absolute val < 10^-10) and negative variances
        variance >= VAR_THRESHOLD ? variance_map[index] = abs(variance) :
        throw("Variance is negative and has absolute value larger than VAR_THRESHOLD")
    end
    # pre-allocated caches
    matrix_cache = zeros(uType, dim_, dim_)
    vector_transpose_cache = zeros(uType, 1, dim_)
    # arrays for plotting
    expectations = []
    cov_matrices = []
    for i in Base.OneTo(length(interval_t))
        push!(
            expectations,
            mapreduce(
                index -> (@view sol_u[i][:, index]) .* expectation_map[index],
                +,
                indices_polys,
            ),
        )
        push!(
            cov_matrices,
            compute_cov_matrix!(
                s,
                c,
                matrix_cache,
                vector_transpose_cache,
                dim_,
                variance_map,
                sol_u[i],
            ),
        )
    end
    # variances --> from diagonal of cov matrix
    return expectations, diag.(cov_matrices)
end

function plot_with_plus_minus_std(
    s::StochGalerkinODE,
    interval_t,
    sol;
    display_plot = false,
)
    @unpack dim_ = s
    sol_u = [sol(t) for t in interval_t]
    expectations, variances = compute_expectation_and_diag_variance(s, interval_t, sol_u)
    lower = expectations .- (v -> sqrt.(v)).(variances)
    upper = expectations .+ (v -> sqrt.(v)).(variances)
    # initialized plot which will be mutated later
    pl = plot(; size = (500, 400))
    results = (lower, expectations, upper)
    colors = [:blue, :green, :red]
    labels = [
        L"\overline{u}(t) - \sqrt{diag(\Sigma(t))}",
        L"\overline{u}(t)",
        L"\overline{u}(t) + \sqrt{diag(\Sigma(t))}",
    ]
    for i in Base.OneTo(length(results))
        result = results[i]
        color = colors[i]
        for j in Base.OneTo(dim_)
            ## put label only onde
            label = j == 1 ? labels[i] : nothing
            plot!(
                pl,
                interval_t,
                (r -> getindex(r, j)).(result),
                color = color,
                label = label,
                legend = :outertopright,
            )
        end
    end
    xlabel!(L"t")
    ylabel!(L"u(t)")
    display_plot && display(pl)
    return pl
end

# TODO: make this less allocation heavy
function compute_expected_variance_exluding_one_variable!(
    s::StochGalerkinODE,
    variance_quad_index_cache,
    variance_index_cache,
    poly_prod,
    u,
    variable_index,
)
    @unpack dim_, num_polys, poly_basis, grid = s
    @unpack quads = grid
    uType = eltype(u)
    num_vars = length(num_polys)
    # number of polynomials for the dimension corresponding to "variable_index"
    num_polys_var_index_dim = num_polys[variable_index]
    # number of polynomials for each dimension other than the one corresponding to "variable_index"
    sz_others = Tuple(num_polys[i] for i in Base.OneTo(num_vars) if i != variable_index)
    # map j -> Var_(xi)[ψ^(i)_{j}(xi)], 1 <= j <= num_polys_var_index_dim, where ψ^(i) is the orthogonal
    #  polynomial family on the dimension corresponding to "variance_index"
    variance_var_index_poly_family_map = zeros(uType, num_polys[variable_index])
    # normalization constant considering distribution of the variable on the "variable_index" dimension
    # (quadrature weights don't include constant factors from pdf's)
    norm_const_var_index = mapreduce(
        j -> quads[variable_index].weights[j],
        +,
        Base.OneTo(num_polys_var_index_dim),
    )
    for j in Base.OneTo(num_polys_var_index_dim)
        variance_var_index_jth_poly =
        # First term is E_{xi}[(ψ^{i}_{j}(xi))^2]
            dot(
                quads[variable_index].weights,
                (x -> poly_basis[variable_index](x, j - 1)^2).(quads[variable_index].nodes),
            ) / norm_const_var_index -
            # Second term is (E_{xi}[ψ^{i}_{j}(xi)])^2
            (
                dot(
                    quads[variable_index].weights,
                    (
                        x -> poly_basis[variable_index](x, j - 1)
                    ).(quads[variable_index].nodes),
                ) / norm_const_var_index
            )^2
        variance_var_index_poly_family_map[j] = variance_var_index_jth_poly
    end
    # helper funcion to shift all indices greater than variable_index one position to the left
    shift_index = j -> j < variable_index ? j : j - 1
    # helper function to attach the value i at the position "variable_index" of index and keeps the rest
    # example: variable_index = 2 --> join_index(7, CartesianIndex(2,3)) = CartesianIndex(2,7,3)
    join_index =
        (i, index) -> CartesianIndex(
            Tuple(
                j == variable_index ? i : index[shift_index(j)] for
                j in Base.OneTo(length(index) + 1)
            ),
        )
    running_sum_expectation_of_variance = zeros(uType, dim_)
    # normalization constant considering distribution of other vars
    # (quadrature weights don't include constant factors from pdf's)
    norm_const_other_vars = zero(uType)
    for quad_index in CartesianIndices(sz_others)
        fill!(variance_quad_index_cache, zero(uType))
        # computes product of weights at the position determined by "quad_index"
        prod_weights = mapreduce(
            j ->
                j == variable_index ? one(uType) :
                quads[j].weights[quad_index[shift_index(j)]],
            *,
            Base.OneTo(num_vars),
        )
        norm_const_other_vars += prod_weights
        # pre-compute product of polynomials of multi-index "index" at the value determined by "quad_index"
        for index in CartesianIndices(sz_others)
            poly_prod[index] = mapreduce(
                j ->
                    j == variable_index ? one(uType) :
                    poly_basis[j](
                        quads[j].nodes[quad_index[shift_index(j)]],
                        index[shift_index(j)] - 1,
                    ),
                *,
                Base.OneTo(num_vars),
            )
        end
        # now compute variance considering other variables
        # we use the following identities in the loop below (note that polynomials of different indices are uncorrelated and i = variable_index,
        # to make the notation easier)
        # Y = ∑_{j <= num_polys_i}[(∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)]) * ψ^{i}_{j}(xi)]
        # --> Var_{xi}(Y|x~i) = ∑_{j <= num_polys_i}[(∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)])^2 * var_{xi}(ψ^{i}_{j}(xi))]
        # where index_ spans over CartesianIndices(sz_others) (the number of polynomials in the other dimensions)
        for j in Base.OneTo(num_polys_var_index_dim)
            # variance_index_cache receives the term ∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)]
            fill!(variance_index_cache, zero(uType))
            variance_index_cache .= mapreduce(
                index_ -> u[:, join_index(j, index_)] .* poly_prod[index_],
                .+,
                CartesianIndices(sz_others),
            )
            # now that we have ∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)],
            # compute (∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)])^2 * var_{xi}(ψ^{i}_{j}(xi))
            # and add to the running sum stored in "variance_quad_index_cache"
            variance_quad_index_cache .+=
                variance_index_cache .^ 2 .* (variance_var_index_poly_family_map[j])
        end
        # now that we have, ∑_{j <= num_polys_i}[(∑_{index_} [C{join_index(i, index_)} * ψ_{index_}(x~i)])^2 * var_{xi}(ψ^{i}_{j}(xi))]
        # multiply by the product of weights (excluding the quadrature on the "variable_index" dimension) and add
        # to the running sum of the expectation of the variance to compute the outer integral (with respect to x~i)
        running_sum_expectation_of_variance .+= variance_quad_index_cache .* prod_weights
    end
    # finally, normalize (norm_const_other_vars is just the sum of the product of weights excluding
    # the quadrature on the "variable_index" dimension)
    running_sum_expectation_of_variance ./= norm_const_other_vars
    return running_sum_expectation_of_variance
end

# TODO: make this less allocation heavy
function compute_total_order_sobol_indices(
    s::StochGalerkinODE,
    sol,
    interval_t,
    variable_index,
)
    sol_u = [sol(t) for t in interval_t]
    c = StochGalerkinODECache(s, first(sol_u))
    @unpack dim_, num_polys, indices_polys, poly_basis, grid = s
    @unpack quads, quad_indices = grid
    uType = eltype(eltype(sol_u))
    # number of polys for all variables except for the one we are computing the Sobol index
    sz_others =
        Tuple(num_polys[i] for i in Base.OneTo(length(num_polys)) if i != variable_index)
    # stores E_{x~i}[Var_{xi}(y|x~i)]
    var_i = []
    # pre-allocated caches
    variance_quad_index_cache = zeros(uType, dim_)
    variance_index_cache = zeros(uType, dim_)
    poly_prod = zeros(uType, sz_others)
    # normalization constant considering joint distribution of other vars
    # (quadrature weights don't include constant factors from pdf's)
    for u in sol_u
        push!(
            var_i,
            compute_expected_variance_exluding_one_variable!(
                s,
                variance_quad_index_cache,
                variance_index_cache,
                poly_prod,
                u,
                variable_index,
            ),
        )
    end
    _, variance = compute_expectation_and_diag_variance(s, interval_t, sol_u)
    return [var_i[j] ./ variance[j] for j in Base.OneTo(length(interval_t))]
end
