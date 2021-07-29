function compute_cov_matrix!(s::StochGalerkinODE,c::StochGalerkinODECache,cov_matrix,matrix_cache,vector_transpose_cache,var_map,u)
    @unpack indices_polys = s
    @unpack y_hat_for_grid,integral_est = c
    fill!(cov_matrix,zero(eltype(y_hat_for_grid)))
    for index_ in indices_polys
      transpose!(vector_transpose_cache,(@view u[:,index_]))
      mul!(matrix_cache,(@view u[:,index_]),vector_transpose_cache)
      cov_matrix .+= matrix_cache .* var_map[index_]
    end
end

function compute_diag_variance(s::StochGalerkinODE,sol_u,sol_t)
    c = StochGalerkinODECache(s,first(sol_u))
    @unpack dim_,num_polys,indices_polys,poly_basis,grid,normalization_map = s
    @unpack quads,quad_indices = grid
    uType = typeof(first(first(sol_u)))
    # Expectation and variance of products of polynomials will be stored here
    exp_map = zeros(uType,num_polys)
    var_map = zeros(uType,num_polys)
    # Normalization constant (quadrature weights don't include constant factors from pdf's)
    norm_const = mapreduce(index_ -> mapreduce(i -> quads[i].weights[index_[i]],*,1:length(index_)),+,quad_indices)
    for index_ in indices_polys
        exp_map[index_] = mapreduce(i -> dot(quads[i].weights, (x -> poly_basis[i](x,index_[i] - 1)).(quads[i].nodes)),*,Base.OneTo(length(index_))) / norm_const
        temp = normalization_map[index_] / norm_const -
                          exp_map[index_]^2
        # Hack for handling very small (abs val < 10^-10) and negative variances
        var_map[index_] = abs(temp)
    end
    # Pre-allocated caches
    transp_cache = zeros(uType,1,dim_)
    matrix_cache = zeros(uType,dim_,dim_)
    cov_matrix = zeros(uType,dim_,dim_)
    # Arrays for plotting
    expectations = [zeros(uType,dim_) for t in sol_t]
    cov_matrices = [zeros(uType,dim_,dim_) for t in sol_t]
    for i in Base.OneTo(length(sol_t))
        expectations[i] = mapreduce(index_ -> (@view sol_u[i][:,index_]) .* exp_map[index_],+,indices_polys)
        compute_cov_matrix!(s,c,var_map,transp_cache,matrix_cache,cov_matrix,sol_u[i])
        cov_matrices[i] .= cov_matrix
    end
    # variances --> from diagonal of cov matrix
    std = (v -> sqrt.(v)).(diag.(cov_matrices))
    return expectations, std
end

function plot_with_plus_minus_std(s::StochGalerkinODE,sol_u,sol_t;display_plot=false)
    @unpack dim_ = s
    expectations, var = compute_var(s,sol_u,sol_t)
    lower = expectations .- var
    upper = expectations .+ var
    # initialized plot that will be mutated later
    pl = plot()
    plts = (lower,expectations,upper)
    colors = [:blue,:green,:red]
    labels = [L"\overline{u}(t) - \sqrt{diag(\Sigma(t))}",L"\overline{u}(t)",L"\overline{u}(t) + \sqrt{diag(\Sigma(t))}"]
    for i in Base.OneTo(length(plts))
        result = plts[i]
        color = colors[i]
        for j in Base.OneTo(dim_)
            ### put label only onde
            label = j == 1 ? labels[i] : nothing
            plot!(pl,sol_t,(r -> getindex(r,j)).(result),color=color,label=label,legend=:outertopright)
        end
    end
    xlabel!("t")
    if display_plot
        display(pl)
    end
    return pl
end

function _compute_total_order_sobol_indices!(s::StochGalerkinODE,var_this_quad_index,var_this_i,poly_prod,u,t,var_index)
    @unpack num_polys,poly_basis,grid = s
    @unpack quads = grid
    uType = typeof(first(u))
    num_vars = length(num_polys)
    # number of polys for all variables except the one we are computing the Sobol index
    sz_others = Tuple(num_polys[i] for i in Base.OneTo(length(num_polys)) if i != var_index)
    # Map i -> Var_(xi)[ψi]
    var_this_map = zeros(uType,num_polys[var_index])
    # Normalization constant considering distribution of this var
    norm_const_this_var = mapreduce(i -> quads[var_index].weights[i],+,Base.OneTo(num_polys[var_index]))
    for i in Base.OneTo(num_polys[var_index])
        var_this = dot(quads[var_index].weights,(x -> poly_basis[var_index](x,i - 1)^2).(quads[var_index].nodes)) / norm_const_this_var -
                   (dot(quads[var_index].weights,(x -> poly_basis[var_index](x,i - 1)).(quads[var_index].nodes)) / norm_const_this_var)^2
        var_this_map[i] = var_this
    end
    # Helper funcs
    eff_index = j -> j < var_index ? j : j - 1
    join_index = (i,index_) -> CartesianIndex(Tuple(j == var_index ? i : index_[eff_index(j)] for j in Base.OneTo(length(index_) + 1)))
    var = zeros(uType,dim_)
    # Normalization constant considering distribution of other vars
    norm_const_other_vars = zero(uType)
    for quad_index in CartesianIndices(sz_others)
        fill!(var_this_quad_index,zero(uType))
        prod_weights = mapreduce(j -> j == var_index ? one(uType) : quads[eff_index(j)].weights[quad_index[eff_index(j)]],*,Base.OneTo(num_vars))
        norm_const_other_vars += prod_weights
        ## Pre-compute polynomial products
        for index_ in CartesianIndices(sz_others)
            poly_prod[index_] = mapreduce(j -> j == var_index ? one(uType) : poly_basis[eff_index(j)](quads[eff_index(j)].
                                nodes[quad_index[eff_index(j)]],index_[eff_index(j)] - 1),*,Base.OneTo(num_vars))
        end
        # Now compute variance considering other vars
        for i in Base.OneTo(num_polys[var_index])
            fill!(var_this_i,zero(uType))
            for index_ in CartesianIndices(sz_others)
                index = join_index(i,index_)
                var_this_i .+= u[:,index] .* poly_prod[index_]
            end
            var_this_quad_index .+= var_this_i.^2 .* (var_this_map[i])
        end
        var .+= var_this_quad_index .* prod_weights
    end
    var ./= norm_const_other_vars
    return sqrt.(var)
end

function compute_total_order_sobol_indices(s::StochGalerkinODE,sol_u,sol_t,var_index)
    c = StochGalerkinODECache(s,first(sol_u))
    @unpack dim_,num_polys,indices_polys,poly_basis,grid = s
    @unpack quads,quad_indices = grid
    uType = typeof(first(first(sol_u)))
    # number of polys for all variables except the one we are computing the Sobol index
    sz_others = Tuple(num_polys[i] for i in Base.OneTo(length(num_polys)) if i != var_index)
    # Will store E[Var(y|x~i)] here
    std_i = []
    # Pre allocated caches
    var_this_quad_index = zeros(uType,dim_)
    var_this_i = zeros(uType,dim_)
    poly_prod = zeros(uType,sz_others)
    # Normalization constant considering joint distribution of other vars
    for (u,t) in zip(sol_u,sol_t)
        std = _compute_total_order_sobol_indices!(s,var_this_quad_index,var_this_i,poly_prod,u,t,var_index)
        push!(std_i,std)
    end
    _, std = compute_var(s,sol_u,sol_t)
    return [std_i[j] ./ std[j] for j in Base.OneTo(length(sol_t))]
end
