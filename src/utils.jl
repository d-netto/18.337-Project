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

function compute_diag_variance(s::StochGalerkinODE, interval_t, sol_u)
    c = StochGalerkinODECache(s, first(sol_u))
    @unpack dim_, num_polys, indices_polys, poly_basis, grid, normalization_map = s
    @unpack quads, quad_indices = grid
    uType = eltype(eltype(sol_u))
    # Expectation and variance of for each polynomial with a given multi-index will be stored here
    expectation_map = zeros(uType, num_polys)
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
        # Hack for handling very small (abs val < 10^-10) and negative variances
        variance_map[index] =
            abs(normalization_map[index] / norm_const - expectation_map[index]^2)
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
    expectations, variances = compute_diag_variance(s, interval_t, sol_u)
    lower = expectations .- (v -> sqrt.(v)).(variances)
    upper = expectations .+ (v -> sqrt.(v)).(variances)
    # initialized plot that will be mutated later
    pl = plot(; size = (500, 400))
    plts = (lower, expectations, upper)
    colors = [:blue, :green, :red]
    labels = [
        L"\overline{u}(t) - \sqrt{diag(\Sigma(t))}",
        L"\overline{u}(t)",
        L"\overline{u}(t) + \sqrt{diag(\Sigma(t))}",
    ]
    for i in Base.OneTo(length(plts))
        result = plts[i]
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
    display_plot && display(pl)
    return pl
end

function compute_expected_variance_exluding_one_variable!(
    s::StochGalerkinODE,
    variance_quad_index_cache,
    variance_index_cache,
    poly_prod,
    u,
    t,
    variable_index,
)
    @unpack dim_, num_polys, poly_basis, grid = s
    @unpack quads = grid
    uType = eltype(u)
    num_vars = length(num_polys)
    # number of polys for all variables except the one we are computing the Sobol index
    sz_others =
        Tuple(num_polys[i] for i in Base.OneTo(length(num_polys)) if i != variable_index)
    # map j -> Var_(xi)[ψi_{j}(xi)], 1 <= j <= variable_index, where ψi is the polynomial family corresponding to the variable
    # on the "variance_index_cache" dimension
    variance_this_poly_family_map = zeros(uType, num_polys[variable_index])
    # normalization constant considering distribution of the variable on the "variable_index" dimension
    norm_const_this_var = mapreduce(
        j -> quads[variable_index].weights[j],
        +,
        Base.OneTo(num_polys[variable_index]),
    )
    for j in Base.OneTo(num_polys[variable_index])
        variance_this_var =
            dot(
                quads[variable_index].weights,
                (x -> poly_basis[variable_index](x, j - 1)^2).(quads[variable_index].nodes),
            ) / norm_const_this_var -
            (
                dot(
                    quads[variable_index].weights,
                    (
                        x -> poly_basis[variable_index](x, j - 1)
                    ).(quads[variable_index].nodes),
                ) / norm_const_this_var
            )^2
        variance_this_poly_family_map[j] = variance_this_var
    end
    # helper funcion
    effective_index = j -> j < variable_index ? j : j - 1
    # attaches the value i at the position "variable_index" of index and keeps the rest
    # example: variable_index = 2 --> join_index(7, CartesianIndex(2,3)) = CartesianIndex(2,7,3)
    join_index =
        (i, index) -> CartesianIndex(
            Tuple(
                j == variable_index ? i : index[effective_index(j)] for
                j in Base.OneTo(length(index) + 1)
            ),
        )
    expectation_of_variance_running_sum = zeros(uType, dim_)
    # normalization constant considering distribution of other vars
    norm_const_other_vars = zero(uType)
    for quad_index in CartesianIndices(sz_others)
        fill!(variance_quad_index_cache, zero(uType))
        # computes product of weights at the position determined by quad_index
        prod_weights = mapreduce(
            j ->
                j == variable_index ? one(uType) :
                quads[j].weights[quad_index[effective_index(j)]],
            *,
            Base.OneTo(num_vars),
        )
        norm_const_other_vars += prod_weights
        # pre-compute product of polynomials of multi-index index at the value determined by quad_index
        for index in CartesianIndices(sz_others)
            poly_prod[index] = mapreduce(
                j ->
                    j == variable_index ? one(uType) :
                    poly_basis[j](
                        quads[j].nodes[quad_index[effective_index(j)]],
                        index[effective_index(j)] - 1,
                    ),
                *,
                Base.OneTo(num_vars),
            )
        end
        # now compute variance considering other variables
        # we use the following identity in the loop below (polynomials of different indices --> uncorrelated):
        # Var_{xi}(Y|x~i) = ∑_{j <= num_poly_i}(∑_{index_} C{index_ ∪ i} * ψ_{index_}(x~i))^2 * var_{xi}(ψ_{j}(xi)), where
        # index_ spans over the number of polynomials in the other dimensions
        for j in Base.OneTo(num_polys[variable_index])
            # variance_index_cache receives the term ∑_{index_} C{index_ ∪ i} * ψindex_(x~i)
            fill!(variance_index_cache, zero(uType))
            for index_ in CartesianIndices(sz_others)
                index = join_index(j, index_)
                variance_index_cache .+= u[:, index] .* poly_prod[index_]
            end
            # now that we have ∑_{index_} C{index_ ∪ i} * ψindex_(x~i), compute (∑_{index_} C{index_ ∪ i} * ψindex_(x~i))^2 * var_{xi}(ψj(xi))
            # and add to the running sum
            variance_quad_index_cache .+=
                variance_index_cache .^ 2 .* (variance_this_poly_family_map[j])
        end
        # now that we have, ∑_{j <= num_poly_i}(∑_{index_} C{index_ ∪ i} * ψ_{index_}(x~i))^2 * var_{xi}(ψ_{j}(xi))
        # multiply by the product of weights (excluding the quadrature on the "variable_index" dimension) and add
        # to the running sum of the expectation of the variance to compute the outer integral (with respect to x~i)
        expectation_of_variance_running_sum .+= variance_quad_index_cache .* prod_weights
    end
    # finally, normalize (norm_const_other_vars is just the sum of the product of weight excluding
    # the quadrature on the "variable_index" dimension
    expectation_of_variance_running_sum ./= norm_const_other_vars
    return sqrt.(expectation_of_variance_running_sum)
end

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
    # will store E_{x~i}[Var_{xi}(y|x~i)] here
    std_i = []
    # pre allocated caches
    variance_quad_index_cache = zeros(uType, dim_)
    variance_index_cache = zeros(uType, dim_)
    poly_prod = zeros(uType, sz_others)
    # normalization constant considering joint distribution of other vars
    for (u, t) in zip(sol_u, interval_t)
        push!(
            std_i,
            compute_expected_variance_exluding_one_variable!(
                s,
                variance_quad_index_cache,
                variance_index_cache,
                poly_prod,
                u,
                t,
                variable_index,
            ),
        )
    end
    _, variance = compute_diag_variance(s, interval_t, sol_u)
    return [std_i[j] ./ sqrt.(variance[j]) for j in Base.OneTo(length(interval_t))]
end
